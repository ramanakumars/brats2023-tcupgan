import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import Resize, FiveCrop


def query_images_and_masks(list_of_ids, umii_path='../umii-fatchecker-dataset/'):
    '''
        Create an image-mask pair cube from a list of
        Zooniverse subject IDs
    '''
    image_cube = {}
    mask_cube = {}

    # we first resize each 1200x1200xn image cube into a 512x512xn cube
    resize_function = Resize(512)
    # and then do a 5 crop to get it into the 256x256xn size for the
    # TCuPGAN input
    five_crop_function = FiveCrop(size=(256, 256))

    for item_number, each_id in enumerate(list_of_ids):
        # read in the image
        image = plt.imread(f'{umii_path}/images/{each_id}.jpg')
        mask = plt.imread(f'{umii_path}/masks/{each_id}.png')

        # resize the image and mask
        image_resized = resize_function(torch.tensor(np.moveaxis(image, 2, 0)))
        mask_resized = resize_function(torch.tensor(np.expand_dims(mask, 0)))

        # finally, do the five crop
        im_five_crops = five_crop_function(image_resized)
        mask_five_crops = five_crop_function(mask_resized)

        # loop through each crop and add it to a dictionary
        if item_number == 0:
            first_item = each_id
            # each crop will be in a [subject_id]_[crop_number]
            # format e.g., 50494631_0, 50494631_1, ...
            for ncrop in range(5):
                image_cube['%s_%s' % (first_item, ncrop)] = []
                mask_cube['%s_%s' % (first_item, ncrop)] = []
        else:
            pass

        # loop through each image/crop and
        for ncrop, (each_im_crop, each_mask_crop) in enumerate(zip(im_five_crops, mask_five_crops)):
            # switch the channel into the PyTorch formax and also stack the 3ch data together
            # since the images are techincally single channeled
            image_cube['%s_%s' % (first_item, ncrop)].append(each_im_crop.numpy()[:1, :, :])

            # normalize the masks so that they are in the range [0, 1]
            if np.max(each_mask_crop.numpy()) > 0:
                mask_cube['%s_%s' % (first_item, ncrop)].append(np.repeat(each_mask_crop.numpy() / np.max(each_mask_crop.numpy()), 1, axis=0))
            else:
                mask_cube['%s_%s' % (first_item, ncrop)].append(np.repeat(each_mask_crop.numpy(), 1, axis=0))

    return image_cube, mask_cube
