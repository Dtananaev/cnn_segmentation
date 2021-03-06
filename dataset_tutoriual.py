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
    Функция загрузки черно-белых картинок.

    Входящие переменные:
       image_path: название картинки с расположением в папке
       (может быть входящая картинка или маска сегментации) 
       (например: ./data/stage1_train/0a7d30b2523/images/0a7d30b252359a10fd298.png)
    Выходящие переменные:
       img: декодированная картинка

    """
    # Считывание файла
    img = tf.io.read_file(image_path)
    # Декодируем png формат
    img = tf.image.decode_png(img, channels=IMG_CHANNELS) # grayscale
    # Изменяем размер картинки в ширина=IMG_WIDTH и высота=IMG_HEIGHT
    if resize:
         img = tf.expand_dims(img, axis=0)
         img = tf.compat.v1.image.resize_nearest_neighbor(img, [IMG_WIDTH, IMG_HEIGHT])
         img = tf.squeeze(img, axis=0)
    # Конвертируем картинку из uint8 в float32 формат или bool (для маски)
    img = tf.cast(img, data_type)
    return img


def load_image_mask(image_path, mask_path):
    image = process_image(image_path, data_type=tf.float32)
    image = image/ 255.0 # Нормализуем картинку до значений между 0 и 1
    mask = process_image(mask_path, data_type=tf.bool)
    mask = tf.cast(mask, tf.float32)
    return image, mask 


def load_data(dataset_dir):
    """
    Функция загружает картинки и маски сегментации.
    Входящие переменные:
    dataset_dir: директория с данными
    """
    # Делаем список всех папок внутри папки dataset_dir
    all_images_pathes = sorted(glob.glob(dataset_dir + "/*/images/*.png"))
    all_masks_pathes = sorted(glob.glob(dataset_dir + "/*/combined_mask/*.png"))
    dataset_size = int(len(all_images_pathes) / BATCH_SIZE)

    ds = tf.data.Dataset.from_tensor_slices((all_images_pathes, all_masks_pathes))
    ds = ds.map(load_image_mask, num_parallel_calls=2)
    ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER) # перемешивает картинки в случайном порядке (SHUFFLE_BUFFER -количество перемешиваемых картинок за раз)
    ds = ds.batch(BATCH_SIZE) # размер батча (количество картинок которые загружаются в нейронку за раз)
    ds = ds.prefetch(buffer_size=2) # подгружает в память 2 картинки что бы тренировка шла быстрее
    return ds, dataset_size

if __name__ =="__main__":
    dataset, dataset_size = load_data(dataset_dir)

    for image, mask in dataset:
        print("image {}, mask {}".format(image.shape, mask.shape))




