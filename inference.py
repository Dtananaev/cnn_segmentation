#
# Author: Denis Tananaev
# Date: 28.02.2020
#
import tensorflow as tf 
from dataset_tutoriual import process_image
import argparse
import os
import glob
import numpy as np
import imageio
from tqdm import tqdm
import cv2

input_dir="./data/stage1_train"


def load_image(image_path):
    image = process_image(image_path)
    image /= 255.0
    image = tf.expand_dims(image, axis=0)
    return image


def inference(input_dir, model_dir, output_dir):
    model = tf.keras.models.load_model(model_dir)
    os.makedirs(output_dir, exist_ok=True)
    all_images_pathes = sorted(glob.glob(input_dir + "/*/images/*.png"))
    for image_path in tqdm(all_images_pathes, total=len(all_images_pathes)):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = load_image(image_path)
        predict = model(image)
        predict = predict[0, :, :, 0].numpy()
        predict = predict * 255.0
        predict = predict.astype("uint8")
        prediction_name = os.path.join(output_dir, image_name + ".png")
        cv2.imwrite(prediction_name, predict) 
    print("all images saved in folder: {}".format(output_dir))


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Inference model.')
   parser.add_argument('--input_dir', default='./data/stage1_test')
   parser.add_argument('--model_dir', default='checkpoints/model-0010')
   parser.add_argument('--output_dir', default='inference')
   args = parser.parse_args()
   inference(args.input_dir, args.model_dir, args.output_dir)
