import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from utils.dice_score import multiclass_dice_coeff, dice_coeff, Jaccard_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, tta=False, post_p=False, last=False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        jaccard_score = torch.zeros([net.n_classes])
        mask = torch.zeros([net.n_classes])
        cm = torch.zeros([net.n_classes, net.n_classes])
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            if image.dim() != 4:
                b, nc, c, h, w = image.size()
                image = image.view(-1, c, h, w)
                mask_true = mask_true.view(-1, h, w)
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            #mask_pred = net(image)
            mask_pred = net(image)['out']
            
            if tta == True:
                from torchvision.transforms import functional as T
                mask_list = [None, None, None, None]
                mask_list[0] = mask_pred
                mask_list[1] = T.hflip(net(T.hflip(image)))
                mask_list[2] = T.vflip(net(T.vflip(image)))
                mask_list[3] = net(image.transpose(-2, -1)).transpose(-2, -1)
                mask_pred = ((mask_list[0] + mask_list[1]) / 2 + (mask_list[0] + mask_list[2]) / 2 + (mask_list[0] + mask_list[3]) / 2) / 3
            
            # post processing
            if post_p == True:
                from utils.post_processing import crf
                image = image.detach().cpu().numpy()
                mask_pred = F.softmax(mask_pred, dim=1).detach().cpu().numpy()
                for i in range(len(image)):
                    mask_pred[i,:,:,:] = crf(image[i,:,:,:], mask_pred[i,:,:,:])
                mask_pred = torch.from_numpy(mask_pred).clone().to(device=device)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                
                jaccard_score += Jaccard_coeff(mask_pred.detach(), mask_true, last)[0]
                mask += Jaccard_coeff(mask_pred.detach(), mask_true, last)[2]
                cm += Jaccard_coeff(mask_pred.detach(), mask_true, last)[1]
    
    jaccard_score /= mask
    jaccard_score = jaccard_score.unsqueeze(0).cpu().numpy()
    cm = cm.type(mask_true.dtype).cpu().numpy()
    net.train()
    return jaccard_score, cm
