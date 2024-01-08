import argparse
import logging
import os
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import csv
from utils.loss_benchmark import FocalLoss
import shutil
from scipy.optimize import curve_fit

import wandb
from evaluate import evaluate
from unet import UNet, UNet_with_CRF
from utils.data_loading import BasicDataset, CarvanaDataset, ImbalancedDatasetSampler
from utils.dice_score import dice_loss
from utils.JSD_loss import calc_jsd_multiscale as calc_jsd_temp
from utils.iou_computation import update_iou_stat, compute_iou, iter_iou_stat, get_mask, iter_fraction_pixelwise

# dir_img = os.environ.get('SM_CHANNEL_TRAINING') + '/imgs_wv3/'
# dir_mask = os.environ.get('SM_CHANNEL_TRAINING') + '/masks_wv3/'
# dir_checkpoint = '/opt/ml/checkpoints'
# dir_model = os.environ.get('SM_MODEL_DIR')
dir_img = Path('../data/imgs/')
dir_mask = Path('../data/masks/')
dir_checkpoint = Path('./checkpoints/')
dir_model =  Path('./')

def curve_func(x, a, b, c):
    return a * (1 - np.exp(-1 / c * x ** b))

def fit(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0=(1, 1, 1), method='trf', sigma=np.geomspace(1, .1, len(y)), \
                           absolute_sigma=True, bounds=([0, 0, 0], [1, 1, np.inf]))
    return tuple(popt)

def derivation(x, a, b, c):
    x = x + 1e-6  # numerical robustness
    return a * b * 1 / c * np.exp(-1 / c * x ** b) * (x ** (b - 1))

def label_update_epoch(ydata_fit, threshold=0.9, eval_interval=100, num_iter_per_epoch=522 / 8):
    #xdata_fit = np.linspace(0, len(ydata_fit) * eval_interval / num_iter_per_epoch, len(ydata_fit))
    xdata_fit = np.linspace(0, len(ydata_fit) - 1, len(ydata_fit))
    a, b, c = fit(curve_func, xdata_fit, ydata_fit)
    epoch = np.arange(1, 16)
    y_hat = curve_func(epoch, a, b, c)
    relative_change = abs(abs(derivation(epoch, a, b, c)) - abs(derivation(1, a, b, c))) / abs(derivation(1, a, b, c))
    relative_change[relative_change > 1] = 0
    update_epoch = np.sum(relative_change <= threshold) + 1
    return update_epoch  # , a, b, c

def if_update(iou_value, current_epoch, threshold=0.90):
    update_epoch = label_update_epoch(iou_value, threshold=threshold)
    print(update_epoch)
    return current_epoch >= update_epoch  # , update_epoch

def train_model(
        model,
        device,
        Lambda1,
        threshold,
        epochs: int = 5,
        start_epoch: int = 1,
        batch_size: int = 1,
        patch_size: int = 128,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        update_interval: int = 1,
        r_threshold: float = 0.9,
        mask_threshold: float = 0.8
):
    # 1. Create dataset
    #try:
    #    dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    #except (AssertionError, RuntimeError, IndexError):
    #    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    #test_set = BasicDataset(dir_img, dir_mask, None, img_scale, val_per=val_percent)
    test_set = BasicDataset(dir_img, dir_mask, patch_size, img_scale, subset="Test", val_per=val_percent)
    val_set = BasicDataset(dir_img, dir_mask, patch_size, img_scale, val_per=val_percent)
    train_set = BasicDataset(dir_img, dir_mask, patch_size, img_scale, subset="Train", val_per=val_percent)
    n_val = len(val_set)
    n_train = len(train_set)
    #train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # stop using data-augmentation during testing
    print(val_set)
    #val_set.val_setting()
    
    # use to record the updated class, so that it won't be updated again
    Updated_class_list = []
    # record the noisy pseudo label fitting IoU for each class
    IoU_npl_dict = {}
    for i in range(5):
        IoU_npl_dict[i] = []
        
    IoU_npl_indx = np.array([])
    
    weight = nn.Parameter(torch.Tensor(3))
    weight.data.fill_(1)
    weight.to(device)
    
    itr = (start_epoch - 1) * len(train_set) // (batch_size)
    max_itr = epochs * len(train_set) // (batch_size)
    max_epoch = epochs

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, sampler=ImbalancedDatasetSampler(train_set), **loader_args)
    #train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader_args = dict(batch_size=int(batch_size//4), num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **test_loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5, last_epoch=-1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)    
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    if (start_epoch - 1) - 5 > 0:
        for i in range((start_epoch - 1) - 5):
            scheduler.step()
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            # noisy pseudo label fit
            TP_npl = [0] * 5
            P_npl = [0] * 5
            T_npl = [0] * 5
            
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                
#                 images_small = F.interpolate(images, scale_factor=0.9, mode='bilinear',
#                                              align_corners=True,
#                                              recompute_scale_factor=True)

#                 images_large = F.interpolate(images, scale_factor=1.1, mode='bilinear',
#                                              align_corners=True,
#                                              recompute_scale_factor=True)
                
#                 images_small = images_small.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
#                 images_large = images_large.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    #masks_pred = model(images)
                    masks_pred = model(images)['out']
                    masks_pred_mix = masks_pred
                    # masks_pred_small = model(images_small)
                    # masks_pred_large = model(images_large)
                    # masks_pred_small = F.interpolate(masks_pred_small['out'], size=(patch_size, patch_size), mode='bilinear', align_corners=True)
                    # masks_pred_large = F.interpolate(masks_pred_large['out'], size=(patch_size, patch_size), mode='bilinear', align_corners=True)
                    # masks_pred_mix = (masks_pred + masks_pred_small + masks_pred_large) / 3.
                    
                    # the statistics about noise segmentation label fitting
                    pred_np = torch.argmax(masks_pred, dim=1).detach().cpu().numpy()  # b, h, w
                    pred_np_mix = torch.argmax(masks_pred_mix, dim=1).detach()  # b, h, w
                    gt_np = true_masks.detach().cpu().numpy()
                    
                    pseudo_masks = train_loader.dataset.update_allclass(IoU_npl_indx, true_masks.cpu(), pred_np_mix.cpu(), masks_pred_mix.cpu().max(1)[0])
                    
                    if model.n_classes == 1:
                        #focalloss = FocalLoss(gamma=0.5)
                        #loss = focalloss(masks_pred.squeeze(1).log_softmax(dim=1).exp(), true_masks.float()).mean()
                        #loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        #loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        pass
                    else:
                        #loss_ce, consistency, variance, mixture_label = calc_jsd_temp(weight.to(device), pseudo_masks.to(device), masks_pred, masks_pred_small, masks_pred_large, threshold=threshold)
                        #loss = loss_ce + Lambda1 * consistency
                        #loss = criterion(masks_pred, true_masks)
                        loss = criterion(masks_pred, pseudo_masks.to(device))
                        #loss += dice_loss(
                            #F.softmax(masks_pred, dim=1).float(),
                            #F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            #multiclass=True
                        #)
                # the statistics about noise segmentation label fitting
                label_np_updated = pseudo_masks.numpy()
                TP_npl, P_npl, T_npl = update_iou_stat(pred_np, label_np_updated, TP_npl, P_npl, T_npl)
                    
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                #for i, idx in enumerate(idxs.detach().cpu().tolist()):
                #    train_loader.dataset.prev_pred_dict[idx[0]] = [pred_np_mix[i, :, :].cpu(), masks_pred_mix.cpu().max(1)[0][i, :, :]]

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                itr += 1
                #if itr >= max_itr:
                #    break
                
                IoU_npl = compute_iou(TP_npl, P_npl, T_npl)
                if itr % 30 == 0:
                    for i in range(5):
                        IoU_npl_dict[i].append(IoU_npl[i])
                        TP_npl = [0] * 5
                        P_npl = [0] * 5
                        T_npl = [0] * 5

            
            #for i in range(5):
            #    IoU_npl_dict[i].append(IoU_npl[i])
            if epoch % 4 == 0:
                IoU_npl_indx = Updated_class_list

                for class_idx in range(5):
                    # current code only support update each class once, if updated, it won't be updated again
                    if not class_idx in Updated_class_list:
                        update_sign = if_update(np.array(IoU_npl_dict[class_idx]), epoch, threshold=r_threshold)
                        if update_sign:
                            IoU_npl_indx.append(class_idx)
                            Updated_class_list.append(class_idx)
                #TP_npl = [0] * 5
                #P_npl = [0] * 5
                #T_npl = [0] * 5

                # the classes that need to be updated in the current epoch
                IoU_npl_indx = np.array(IoU_npl_indx)

            #if epoch % args.update_interval == 0 and len(IoU_npl_indx) > 0:
            #    # update the segmentation label
            #    train_loader.dataset.update_seg_dict(IoU_npl_indx, 
            #                                        mask_threshold=mask_threshold)
            # clean the prev_pred_dict to save CPU memory
            #train_loader.dataset.prev_pred_dict.clear()
                
            scheduler.step()
            #if epoch > 5:
            #    scheduler.step()

            # Evaluation round #stop using sk_learn!
            # histograms = {}
            # for tag, value in model.named_parameters():
            #     tag = tag.replace('/', '.')
            #     if not (torch.isinf(value) | torch.isnan(value)).any():
            #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
            #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score, _ = evaluate(model, val_loader, device, amp)
            #scheduler.step(val_score)
            
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            
            logging.info('Validation Dice score: {}'.format(val_score))
            with open(str(dir_checkpoint) + '/latest_score_cm.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerows(val_score)
                writer.writerows([['Epoch: {} \n'.format(epoch+1)]])
            try:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'validation Jaccard': val_score.mean(),
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                    },
                    'step': global_step,
                    'epoch': epoch,
                    **histograms
                })
            except:
                pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            #state_dict['mask_values'] = dataset.mask_values
            state_dict['mask_values'] = train_set.mask_values
            state_dict['end_epoch'] = epoch
            torch.save(state_dict, str(dir_checkpoint) + '/checkpoint.pth'.format(epoch))
            logging.info(f'Checkpoint {epoch} saved!')
    
    try:
        shutil.move(str(dir_checkpoint), str(dir_model))
    except:
        pass
    
    _, cm = evaluate(model, test_loader, device, amp, last=True)
        
    with open(str(dir_model) + '/latest_cm.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(cm)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--patch-size', '-p', dest='patch_size', metavar='P', type=int, default=128, help='Patch size')
    parser.add_argument('--n_channels', '-n', dest='n_channels', metavar='N', type=int, default=3, help='Number of channels')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    #parser.add_argument('--classes', '-c', type=int, default=22, help='Number of classes')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
    #parser.add_argument("--Lambda1", type=float, default=1,
    #                    help="to balance the loss between CE and Consistency loss")
    parser.add_argument("--Lambda1", type=float, default=1.0,
                        help="to balance the loss between CE and Consistency loss")
    parser.add_argument('--threshold', type=float, default=0.8,
                        help="threshold to select the mask, ")
    parser.add_argument('--update_interval', type=int, default=1,
                        help="evaluate the prediction every 1 epoch")
    parser.add_argument('--r_threshold', type=float, default=0.95,
                        help="the r threshold to decide if_update")
    parser.add_argument('--mask_threshold', type=float, default=0.95,
                        help="only the region with high probability and disagree with Pseudo label be updated")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # model = UNet(n_channels=args.n_channels, n_classes=args.classes, bilinear=args.bilinear)
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead
    from torchvision import models
    model = models.segmentation.deeplabv3_resnet50(pretrained=False,
                                                    progress=True)
    model.aux_classifier = None
    model.classifier = DeepLabHead(2048, args.classes)
    model = model.to(memory_format=torch.channels_last)
    model.n_channels = 3
    model.n_classes = 5
    model.bilinear = None
    
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    try:
        state_dict = torch.load(str(dir_checkpoint) + '/checkpoint.pth', map_location=device)
        start_epoch = state_dict['end_epoch'] + 1
        del state_dict['mask_values']
        del state_dict['end_epoch']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
    except:
        start_epoch = 1

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            start_epoch=start_epoch,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            Lambda1=args.Lambda1,
            threshold=args.threshold,
            update_interval=args.update_interval,
            r_threshold=args.r_threshold,
            mask_threshold=args.mask_threshold
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            start_epoch=start_epoch,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            Lambda1=args.Lambda1,
            threshold=args.threshold,
            update_interval=args.update_interval,
            r_threshold=args.r_threshold,
            mask_threshold=args.mask_threshold
        )
