import argparse
import logging
import os
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
import numpy as np
import csv

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

dir_img = Path('../data/imgs/')
dir_mask = Path('../data/masks/')
dir_checkpoint = Path('./checkpoints/')


def eval_model(\
               model,\
               device,\
               img_scale: float = 0.5,\
               batch_size: int = 1,\
               patch_size: int = 128,\
               val_percent: float = 0.1,\
               tta: bool = False,\
               post_p: bool = False,\
              ):
    #test_set = BasicDataset(dir_img, dir_mask, None, img_scale, val_per=val_percent)
    test_set = BasicDataset(dir_img, dir_mask, patch_size, img_scale, subset="Test", val_per=val_percent)
    n_test = len(test_set)
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    val_score, cm = evaluate(model, test_loader, device, amp=False, tta=tta, post_p=post_p, last=True)
    correct = 0
    for i in range(len(cm[0,:])):
        correct += cm[i, i]
    print('Acc:' + str(correct / np.sum(cm)))
    with open('./latest_cm.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(cm)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--patch-size', '-p', dest='patch_size', metavar='P', type=int, default=128, help='Patch size')
    parser.add_argument('--load', '-f', type=str, default='./checkpoints/checkpoint_1.pth', help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    #parser.add_argument('--classes', '-c', type=int, default=22, help='Number of classes')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
    parser.add_argument('--tta', action='store_true', default=False, help='use test time augmentation')
    parser.add_argument('--post_p', action='store_true', default=False, help='use conditional random field')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = models.segmentation.deeplabv3_resnet50(pretrained=False,
                                                  progress=True)
    model.aux_classifier = None
    model.classifier = DeepLabHead(2048, args.classes)
    model = model.to(memory_format=torch.channels_last)
    model.n_channels = 3
    model.n_classes = 5
    model.bilinear = None
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        del state_dict['end_epoch']
        model.load_state_dict(state_dict)
    model.to(device=device)
    try:
        eval_model(
            model=model,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            tta=args.tta,
            post_p=args.post_p
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        model.use_checkpointing()
        eval_model(
            model=model,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            tta=args.tta,
            post_p=args.post_p
        )
