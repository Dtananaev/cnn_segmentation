#
# Author: Denis Tananaev
# Date: 23.02.2020
#

import tensorflow as tf
import os
import numpy as np
import glob
from PIL import Image

dataset_dir="./data/stage1_train"
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
BATCH_SIZE = 4


def create_combined_masks(dataset_dir):
    search_string = os.path.join(dataset_dir, "*")
    folders_list = glob.glob(search_string)
    for folder in folders_list:
        search_string = os.path.join(folder, "masks", "*.png")
        combined_mask_dir = os.path.join(folder, "combined_mask")
        combined_mask_name = os.path.join(combined_mask_dir, folder.split("/")[-1] + ".png")

        os.makedirs(combined_mask_dir, exist_ok=True)

        masks_list =  glob.glob(search_string)
        init_combined_mask = False
        combined_mask = None
        for mask_part in masks_list:
            mask = Image.open(mask_part)
            if combined_mask is None:
               combined_mask = np.zeros_like(mask)
            combined_mask = np.maximum(combined_mask, mask)
        im = Image.fromarray(combined_mask)
        im.save(combined_mask_name)
        print("Save combined mask to {}".format(combined_mask_name))


if __name__ == '__main__':
   create_combined_masks(dataset_dir)
   print("Done!")