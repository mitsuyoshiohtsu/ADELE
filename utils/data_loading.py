import logging
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from utils.transforms import *
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from osgeo import gdal


def load_image(filename, rand=None, cropsize=None, test=False):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext == '.tif':
        if cropsize is not None:
            if rand is not None:
                #raster = gdal.Open('./' + str(filename))
                raster = gdal.Open(str(filename))
                rows = raster.RasterXSize
                cols = raster.RasterYSize
                xoff = int((rows - cropsize) * rand[0])
                yoff = int((cols - cropsize) * rand[1])
                src = raster.ReadAsArray(xoff, yoff, cropsize, cropsize)
            elif test == True:
                raster = gdal.Open(str(filename))
                rows = raster.RasterXSize
                cols = raster.RasterYSize
                src = []
                for xoff in range(0, rows, cropsize):
                    for yoff in range(0, rows, cropsize):
                        #print(raster.ReadAsArray(xoff, yoff, cropsize, cropsize).dtype)
                        src.append(raster.ReadAsArray(xoff, yoff, cropsize, cropsize))
                src = np.array(src)
            else:
                #raster = gdal.Open('./' + str(filename))
                raster = gdal.Open(str(filename))
                rows = raster.RasterXSize
                cols = raster.RasterYSize
                xoff = rows // 2 - cropsize // 2
                yoff = rows // 2 - cropsize // 2
                src = raster.ReadAsArray(xoff, yoff, cropsize, cropsize)
        else:
            #src = gdal.Open('./' + str(filename)).ReadAsArray()
            src = gdal.Open(str(filename)).ReadAsArray()
        return src
    #else:
    #   return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.tif'))[0]
    #mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, patch_size: int, scale: float = 1.0, mask_suffix: str = '', subset=None, val_per=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.subset = subset
        self.patchsize = patch_size
        random.seed(42)

        self.ids = np.array(sorted([splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.') and splitext(file)[1] == '.tif']))
        np.random.seed(0)
        indices = np.arange(len(self.ids))
        np.random.shuffle(indices)
        self.ids = self.ids[indices]
        n_train = len(self.ids) - int(len(self.ids) * val_per)
        if self.subset == "Train":
            self.ids = list(self.ids[:n_train])
        else:
            self.ids = list(self.ids[n_train:])
        #self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        if self.subset == "Train":
            self.transforms = Compose([
                IdentityTrans(),
                #RandomHorizontalFlip(0.5),
                #RandomVerticalFlip(0.5),
                #RandomRotation(0.5),
                #Normalize((0, 0, 0), (1, 1, 1)),
            ])
        else:
            self.transforms = Compose([
                IdentityTrans(),
                #CenterCrop(patch_size),
                #Normalize((0, 0, 0), (1, 1, 1)),
            ])

        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)
    

    @staticmethod
    #def preprocess(mask_values, pil_img, scale, is_mask):
    def preprocess(mask_values, img, scale, is_mask):
        #w, h = pil_img.size
        #newW, newH = int(scale * w), int(scale * h)
        #assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        #pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        #img = np.asarray(pil_img)
        #img = pil_img
        #newH = 256
        #newW = 256
        H = img.shape[-1]
        W = img.shape[-2]
        
        if is_mask:
            #mask = np.zeros((newH, newW), dtype=np.int64)
            mask = np.zeros((H, W), dtype=np.int64)
            #for i, v in enumerate(mask_values):
                #if img.ndim == 2:
                #    mask[img == v] = i
                #else:
                #    mask[(img == v).all(-1)] = i
            for v in mask_values:
                if img.ndim == 2:
                    if v < 21:
                        mask[img == v] = 0
                    elif v < 26:
                        mask[img == v] = 1
                    elif v < 42:
                        mask[img == v] = 2
                    elif v < 43:
                        mask[img == v] = 3
                    else:
                        mask[img == v] = 4
                else:
                    if v < 21:
                        mask[(img == v).all(-1)] = 0
                    elif v < 26:
                        mask[(img == v).all(-1)] = 1
                    elif v < 42:
                        mask[(img == v).all(-1)] = 2
                    elif v < 43:
                        mask[(img == v).all(-1)] = 3
                    else:
                        mask[(img == v).all(-1)] = 4  
            return mask
            
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img

            if (img > 1).any():
                img = img / 255.0
            return img
            
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.tif'))
        img_file = list(self.images_dir.glob(name + '.tif'))
        #mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        #img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        # mask = load_image(mask_file[0])
        # img = load_image(img_file[0])
        if self.subset == "Train":
            rand = [random.random(), random.random()]
            mask = load_image(mask_file[0], rand=rand, cropsize=self.patchsize)
            img = load_image(img_file[0], rand=rand, cropsize=self.patchsize)
        elif self.subset == "Test":
            mask = load_image(mask_file[0], rand=None, cropsize=self.patchsize, test=True)
            img = load_image(img_file[0], rand=None, cropsize=self.patchsize, test=True)
        else:
            mask = load_image(mask_file[0], rand=None, cropsize=self.patchsize)
            img = load_image(img_file[0], rand=None, cropsize=self.patchsize)
        
        assert img.shape[-1] * img.shape[-2] == mask.shape[-1] * mask.shape[-2], \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        #assert img.size == mask.size, \
        #    f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
        if self.subset == 'Test':
            for i in range(img.shape[0]):
                img = img.astype('float64') 
                img[i, :, :, :] = self.preprocess(self.mask_values, img[i, :, :, :], self.scale, is_mask=False)
                mask[i, :, :] = self.preprocess(self.mask_values, mask[i, :, :], self.scale, is_mask=True)
        else:
            img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
            mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        
        img = torch.as_tensor(img.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()
        
        img, mask = self.transforms(img, mask)
        
        return {
            'image': img,
            'mask': mask,
        }
            
    #def update_allclass(self, idx, IoU_npl_indx, mask_threshold, IoU_npl_constraint, class_constraint=True, update_or_mask='update', update_all_bg_img=False):
    def update_allclass(self, IoU_npl_indx, seg_label, seg_argmax, seg_prediction_max_prob, mask_threshold=0.8, IoU_npl_constraint='single', class_constraint=True, update_or_mask='update', update_all_bg_img=False):
        #seg_label = self.seg_dict[idx].unsqueeze(0)  # 1,h,w
        b, h, w = seg_label.size()  # b,h,w

        # if seg label does not belong to the set of class that needs to be updated (exclude the background class), return
        #if set(np.unique(seg_label.numpy())).isdisjoint(set(IoU_npl_indx[1:])):
            # only the background in the pseudo label
            # if update_all_bg_img and len(np.unique(seg_label.numpy()))==1 and np.unique(seg_label.numpy())[0]==0:
            #if update_all_bg_img and not (set(np.unique(seg_label.numpy())) == set([])):
            #    pass
            #else:
            #    return
        
        #seg_argmax, seg_prediction_max_prob = self.prev_pred_dict[idx]
        #seg_argmax = seg_argmax
        #seg_prediction_max_prob = seg_prediction_max_prob

        # if the class_constraint==True and seg label has foreground class
        # we prevent using predicted class that is not in the pseudo label to correct the label
        if class_constraint == True and (set(np.unique(seg_label.numpy())) == set([])):
            for i_batch in range(b):
                unique_class = torch.unique(seg_label[i_batch])
                # print(unique_class)
                indx = torch.zeros((h, w), dtype=torch.long)
                for element in unique_class:
                    indx = indx | (seg_argmax[i_batch] == element)
                seg_argmax[i_batch][(indx == 0)] = 255

        seg_mask_255 = (seg_argmax == 255)

        # seg_change_indx means which pixels need to be updated,
        # find index where prediction is different from label,
        # and  it is not a ignored index and confidence is larger than threshold
        seg_change_indx = (seg_label != seg_argmax) & (~seg_mask_255) & (
                seg_prediction_max_prob > mask_threshold)

        # when set to "both", only when predicted class and pseudo label both existed in the set, the label would be corrected
        # this is a conservative way, during our whole experiments, IoU_npl_constraint is always set to be "single",
        # this is retained here in case user may find in useful for their dataset
        if IoU_npl_constraint == 'both':
            class_indx_seg_argmax = torch.zeros((b, h, w), dtype=torch.bool)
            class_indx_seg_label = torch.zeros((b, h, w), dtype=torch.bool)

            for element in IoU_npl_indx:
                class_indx_seg_argmax = class_indx_seg_argmax | (seg_argmax == element)
                class_indx_seg_label = class_indx_seg_label | (seg_label == element)
            seg_change_indx = seg_change_indx & class_indx_seg_label & class_indx_seg_argmax

        #  when set to "single", when predicted class existed in the set, the label would be corrected, no need to consider pseudo label
        # e.g. when person belongs to the set, motor pixels in the pseudo label can be updated to person even if motor is not in set
        elif IoU_npl_constraint == 'single':
            class_indx_seg_argmax = torch.zeros((b, h, w), dtype=torch.bool)

            for element in IoU_npl_indx:
                class_indx_seg_argmax = class_indx_seg_argmax | (seg_argmax == element)
            seg_change_indx = seg_change_indx & class_indx_seg_argmax

        # if the foreground class portion is too small, do not update
        # seg_label_clone = seg_label.clone()
        # seg_label_clone[seg_change_indx] = seg_argmax[seg_change_indx]
        # if torch.sum(seg_label_clone!=0) < 0.5 * torch.sum(seg_label!=0) and torch.sum(seg_label_clone==0)/(b*h*w)>0.95:
        #     return

        # update or mask 255
        if update_or_mask == 'update':
            seg_label[seg_change_indx] = seg_argmax[seg_change_indx]  # update all class of the pseudo label
        else:
            # mask the pseudo label for 255 without computing the loss
            seg_label[seg_change_indx] = (torch.ones((b, h, w), dtype=torch.long) * 255)[
                seg_change_indx]  # the updated pseudo label
        return seg_label
        #self.seg_dict[idx] = seg_label.squeeze()


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
        
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None, num_class=5):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_rate = [0. for i in range(num_class)]
        for idx in self.indices:
            label_dist = self._get_dist(dataset, idx).view(-1)
            for label in range(len(label_dist)):
                label_to_rate[label] += label_dist[label]
        for label in range(num_class):
            label_to_rate[label] = label_to_rate[label] / len(self.indices)
                
        # weight for each sample
        weights = [self._calc_weights(label_to_rate, dataset, idx) \
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_dist(self, dataset, idx):
        mask = dataset[idx]['mask'].view(-1)
        num_class = mask.max().item() + 1
        label_dist = nn.functional.one_hot(mask, num_class).to(float)
        return label_dist.mean(0)
    
    def _calc_weights(self, l_rate, dataset, idx):
        weight = 0.
        l_dist = self._get_dist(dataset, idx)
        for i in range(len(l_dist)):
            weight += 1. / l_rate[i] * l_dist[i]
        return weight
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
