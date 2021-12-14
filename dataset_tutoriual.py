#
# Author: Denis Tananaev
# Date: 23.02.2020
#


import tensorflow as tf
import os
import numpy as np
import glob

dataset_dir="./data/stage1_train"
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
BATCH_SIZE = 4
SHUFFLE_BUFFER = 8


def process_image(image_path, resize=True, data_type=tf.float32):
    """
    The function of the loading of the images

    Args:
       image_path: path to the image folder (can be image or segmentation mask)
       (e.g.: ./data/stage1_train/0a7d30b2523/images/0a7d30b252359a10fd298.png)
    Return:
       img: decoded image

    """
    # Read the file
    img = tf.io.read_file(image_path)
    # Decode from png
    img = tf.image.decode_png(img, channels=IMG_CHANNELS) # grayscale
    # Resize width=IMG_WIDTH and height=IMG_HEIGHT
    if resize:
         img = tf.expand_dims(img, axis=0)
         img = tf.compat.v1.image.resize_nearest_neighbor(img, [IMG_WIDTH, IMG_HEIGHT])
         img = tf.squeeze(img, axis=0)
    # Convert to float32 for image or bool for binary mask
    img = tf.cast(img, data_type)
    return img


def load_image_mask(image_path, mask_path):
    image = process_image(image_path, data_type=tf.float32)
    image = image/ 255.0 # normalization between 0 and 1
    mask = process_image(mask_path, data_type=tf.bool)
    mask = tf.cast(mask, tf.float32)
    return image, mask 


def load_data(dataset_dir):
    """
    The function loads images and segmentation binary masks.
    Args:
    dataset_dir: directory with data
    """
    # Make a lists of the data in folders
    all_images_pathes = sorted(glob.glob(dataset_dir + "/*/images/*.png"))
    all_masks_pathes = sorted(glob.glob(dataset_dir + "/*/combined_mask/*.png"))
    dataset_size = int(len(all_images_pathes) / BATCH_SIZE)

    ds = tf.data.Dataset.from_tensor_slices((all_images_pathes, all_masks_pathes))
    ds = ds.map(load_image_mask, num_parallel_calls=2)
    ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER) # randomly shuffle during training (SHUFFLE_BUFFER -number of shuffled images per time)
    ds = ds.batch(BATCH_SIZE) # batch size (number of images which loaded to the CNN)
    ds = ds.prefetch(buffer_size=2) # prefetch in RAM 2 images to make loading to GPU faster
    return ds, dataset_size

if __name__ =="__main__":
    dataset, dataset_size = load_data(dataset_dir)

    for image, mask in dataset:
        print("image {}, mask {}".format(image.shape, mask.shape))




