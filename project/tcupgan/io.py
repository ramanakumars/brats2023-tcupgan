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

    def __init__(self, root_folder, num_classes=4, size=256):
        self.root_folder = root_folder
        self.sub_folders = [x for x in sorted(glob.glob(self.root_folder + "/*"))]
        self.case_ID = [int(os.path.basename(folder).split('-')[2]) for folder in self.sub_folders]
        self.num_classes = num_classes
        self.resize = transforms.Resize((size, size), antialias=True)

        print(f"Found {len(self)} images")

    def __len__(self):
        return len(self.sub_folders)

    def __getitem__(self, index):
        folder = self.sub_folders[index]

        t1 = nib.load(glob.glob(folder + "/*-t1n.nii.gz")[0]).get_fdata()
        t1ce = nib.load(glob.glob(folder + "/*-t1c.nii.gz")[0]).get_fdata()
        t2 = nib.load(glob.glob(folder + "/*-t2w.nii.gz")[0]).get_fdata()
        flair = nib.load(glob.glob(folder + "/*-t2f.nii.gz")[0]).get_fdata()

        img = torch.zeros((4, 240, 240, 155), dtype=torch.float)
        img[0, :, :] = torch.as_tensor(t1 / np.percentile(t1.flatten(), 99))
        img[1, :, :] = torch.as_tensor(t1ce / np.percentile(t1ce.flatten(), 99))
        img[2, :, :] = torch.as_tensor(t2 / np.percentile(t2.flatten(), 99))
        img[3, :, :] = torch.as_tensor(flair / np.percentile(flair.flatten(), 99))
        img = self.resize(rearrange(img, "c h w z -> z c h w"))

        try:
            seg = np.asarray(nib.load(glob.glob(folder + "/*-seg.nii.gz")[0]).get_fdata(), dtype=int)
            seg_t = torch.LongTensor(seg)
            mask = self.resize(rearrange(one_hot(seg_t, num_classes=self.num_classes).to(torch.float), 'h w t c -> t c h w'))
        except IndexError:
            mask = None

        return img, mask

    def get_from_ID(self, ID):
        index = self.case_ID.index(ID)
        return self[index]


class VideoDataGenerator(Dataset):
    size = 192

    def __init__(self, root_folder, max_frames=10, in_channels=3, out_channels=126, verbose=False):
        self.root_folder = root_folder
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.frames = []
        self.data = []
        for name in tqdm.tqdm(sorted(os.listdir(self.root_folder)), desc='Building dataset', disable=not verbose, ascii=True):
            folder = os.path.join(self.root_folder, name)
            if not os.path.isdir(folder):
                continue
            frames = sorted(glob.glob(os.path.join(folder, "origin/*.jpg")))
            maskframes = sorted(glob.glob(os.path.join(folder, "mask/*.png")))

            if len(maskframes) != len(frames):
                continue

            nframes = len(frames)

            assert max_frames <= nframes,\
                f"Maximum slice dimension is greater than the number of frames in {folder}"

            nbatches = nframes // max_frames + 1

            self.frames.append(nframes)

            for i in range(nbatches):
                # store the data as the folder and the start frame
                self.data.append([folder, min([i * max_frames, nframes - max_frames])])

        self.transform = transforms.Resize((self.size, self.size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        folder, start = self.data[index]

        img_fnames = sorted(glob.glob(os.path.join(self.root_folder, f"{folder}/origin/*.jpg")))[start:start + self.max_frames]

        imgs = torch.zeros((self.max_frames, self.in_channels, self.size, self.size))
        masks = torch.zeros((self.max_frames, self.out_channels, self.size, self.size))

        for i, fname in enumerate(img_fnames):
            img, mask = self.get_image_mask_pair(fname)
            imgs[i, :] = self.transform(img)
            masks[i, :] = self.transform(mask)

        return imgs, masks

    def get_image_mask_pair(self, fname):
        img = read_image(fname, ImageReadMode.RGB)
        mask = read_image(fname.replace("origin", "mask").replace('.jpg', '.png'), ImageReadMode.GRAY)[0, :]

        mask[mask > 124] = 0
        mask = one_hot(
            mask.to(torch.int64),
            num_classes=self.out_channels
        )
        mask = rearrange(mask, "h w c -> c h w")

        return img, mask


class NpzDataSet(Dataset):
    file_type = 'npz'

    def __init__(self, datafolder, inchannels=3, outchannels=3, norm=1.):
        self.img_datafolder = datafolder

        self.imgfiles = np.asarray(
            sorted(glob.glob(datafolder + f"*.{self.file_type}")))

        self.indices = np.arange(len(self.imgfiles))

        self.ndata = len(self.indices)

        self.inchannels = inchannels
        self.outchannels = outchannels

        self.norm = norm
        self.perc_normalize = False

        # get info about the data

        if self.file_type == 'npz':
            img0 = np.load(self.imgfiles[0])['img']
        else:
            img0 = np.load(self.imgfiles[0])

        if len(img0.shape) == 3:
            self.d, self.h, self.w = img0.shape
        else:
            self.d, self.nch, self.h, self.w = img0.shape

        print(f"Found {self.ndata} images of shape {self.w}x{self.h}x{self.d} with {self.nch} channels")

    def __len__(self):
        return self.ndata

    def __getitem__(self, index):
        imgfile = self.imgfiles[index]

        data = np.load(imgfile)

        imgs = torch.as_tensor(data['img'], dtype=torch.float) / self.norm
        segs = torch.as_tensor(data['mask'], dtype=torch.float)

        return imgs, segs
