import random

import numpy as np
import torch
#from torchvision import transforms as T
from torchvision.transforms import functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class IdentityTrans:
    def __init__(self):
        pass

    def __call__(self, image, target):
        return image, target
    
class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomVerticalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class RandomCrop:
    def __init__(self, patchsize):
        self.size = patchsize

    def __call__(self, image, target):
        imagesize = image.size()[-1]
        top = random.randrange(imagesize-self.size)
        left = random.randrange(imagesize-self.size)
        image = F.crop(image, top, left, self.size, self.size)
        target = F.crop(target, top, left, self.size, self.size)
        return image, target

class CenterCrop:
    def __init__(self, patchsize):
        self.size = patchsize

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

class RandomRotation:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = image.transpose(-2, -1)
            target = target.transpose(-2, -1)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target