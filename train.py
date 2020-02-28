#
# Author: Denis Tananaev
# Date: 28.02.2020
#
import tensorflow as tf 
from dataset_tutoriual import load_data
from model import SegmentationModel
from tqdm import tqdm
import os
dataset_dir="./data/stage1_train"
checkpoint_path = "checkpoints"


@tf.function
def train_step(images, masks, model, loss_object, optimizer):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(masks, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_outputs = [loss, images, masks, predictions]
  return train_outputs


def train_summaries(train_outputs, optimizer):
    loss, images, masks, prediction = train_outputs
    tf.summary.scalar("loss", loss, step=optimizer.iterations)
    if optimizer.iterations % 50 == 0:
         with tf.name_scope("1-Inputs"):
            tf.summary.image("1. input images", images, step=optimizer.iterations)
            tf.summary.image("2. input masks", masks, step=optimizer.iterations)
         with tf.name_scope("2-Prediction"):
           tf.summary.image("3. Predicted masks", prediction, step=optimizer.iterations)


def train():
    EPOCHS = 100
    dataset, dataset_size = load_data(dataset_dir)
    model = SegmentationModel()
    inputs = tf.ones([1, 128, 128, 1], tf.float32)
    model.predict(inputs)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    model_path = "checkpoints/model-{epoch:04d}"
    writer = tf.summary.create_file_writer("summaries")
   
    with writer.as_default():
        for epoch in range(EPOCHS):
          print("Epoch {}".format(epoch))
          for images, masks in tqdm(dataset, total=dataset_size):
            train_outputs = train_step(images, masks, model, loss_object, optimizer)
            train_summaries(train_outputs, optimizer)
          model.save(model_path.format(epoch=epoch))


if __name__ == "__main__":
     train()

