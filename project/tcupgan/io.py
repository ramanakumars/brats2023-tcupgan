import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torch.nn.functional import one_hot
from torchvision import transforms
from einops import rearrange
import os
import tqdm
import nibabel as nib


class BraTSDataGenerator(Dataset):

    def __init__(self, root_folder, num_classes=4, size=256, normalize=True, add_noise = False):
        self.root_folder = root_folder
        self.sub_folders = [x for x in sorted(glob.glob(self.root_folder + "/*"))]
        self.IDs = ['-'.join(os.path.basename(folder).split('-')[2:4]) for folder in self.sub_folders]
        self.num_classes = num_classes
        self.resize = transforms.Resize((size, size), antialias=True)
        self.normalize = normalize
        self.add_noise = add_noise
        self.im_size = size

        print(f"Found {len(self)} images")

    def __len__(self):
        return len(self.sub_folders)

    def __getitem__(self, index):
        folder = self.sub_folders[index]

        t1 = nib.load(glob.glob(folder + "/*-t1n.nii.gz")[0]).get_fdata()
        t1ce = nib.load(glob.glob(folder + "/*-t1c.nii.gz")[0]).get_fdata()
        t2 = nib.load(glob.glob(folder + "/*-t2w.nii.gz")[0]).get_fdata()
        flair = nib.load(glob.glob(folder + "/*-t2f.nii.gz")[0]).get_fdata()
        
        
        if self.normalize:
            img = torch.zeros((4, 240, 240, 155), dtype=torch.float)
            img[0, :, :] = torch.as_tensor(t1 / np.percentile(t1.flatten(), 99))
            img[1, :, :] = torch.as_tensor(t1ce / np.percentile(t1ce.flatten(), 99))
            img[2, :, :] = torch.as_tensor(t2 / np.percentile(t2.flatten(), 99))
            img[3, :, :] = torch.as_tensor(flair / np.percentile(flair.flatten(), 99))
        else:
            img = torch.zeros((4, 240, 240, 155), dtype=torch.float)
            img[0, :, :] = torch.as_tensor(t1)
            img[1, :, :] = torch.as_tensor(t1ce)
            img[2, :, :] = torch.as_tensor(t2)
            img[3, :, :] = torch.as_tensor(flair)
        img = self.resize(rearrange(img, "c h w z -> z c h w")).cpu().numpy()
        
        if self.add_noise:
            noise_vector = np.random.normal(0.0,0.1, size=(155,4,self.im_size,self.im_size))
            zero_condition = np.where(img!= 0)
            noised_img = np.zeros_like(img)
            img[zero_condition] = img[zero_condition] + noise_vector[zero_condition]
        
        img = torch.Tensor(img)

        try:
            seg = np.asarray(nib.load(glob.glob(folder + "/*-seg.nii.gz")[0]).get_fdata(), dtype=int)
            seg_t = torch.LongTensor(seg)
            mask = self.resize(rearrange(one_hot(seg_t, num_classes=self.num_classes).to(torch.float), 'h w t c -> t c h w'))[:,1:,:,:]
        except IndexError:
            mask = None

        return img, mask

    def get_from_ID(self, ID, timepoint):
        index = self.IDs.index(f'{ID:05d}-{timepoint:03d}')
        return self[index]



