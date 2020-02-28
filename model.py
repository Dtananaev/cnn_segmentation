#
# Author: Denis Tananaev
# Date: 28.02.2020
#
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras import Model


class SegmentationModel(Model):
  def __init__(self):
    super(SegmentationModel, self).__init__()
    # 128 x 128
    self.conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')
    self.conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')
    self.pool1 = MaxPool2D(pool_size=(2, 2))
    # 64 x 64
    self.conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')
    self.conv4 = Conv2D(16, (3, 3), activation='relu', padding='same')
    self.pool2 = MaxPool2D(pool_size=(2, 2))
    # 32 x 32
    self.conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')
    self.conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')
    self.pool3 = MaxPool2D(pool_size=(2, 2))
    # 16 x 16
    self.conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')
    self.conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')
    self.pool4 = MaxPool2D(pool_size=(2, 2))
    # 8 x 8
    self.conv9 = Conv2D(96, (3, 3), activation='relu', padding='same')
    self.conv10 = Conv2D(96, (3, 3), activation='relu', padding='same')
    self.up1 = UpSampling2D(size=(2, 2))
    # 16 x 16
    self.conv11 =Conv2D(64, (3, 3), activation='relu', padding='same')
    self.up2 = UpSampling2D(size=(2, 2))
    # 32 x 32
    self.conv12 =Conv2D(32, (3, 3), activation='relu', padding='same')
    self.up3 = UpSampling2D(size=(2, 2))
    # 64 x 64
    self.conv13 =Conv2D(16, (3, 3), activation='relu', padding='same')
    self.up4 = UpSampling2D(size=(2, 2))
    # 128 x 128
    self.conv14 = Conv2D(8, (3, 3), activation='relu', padding='same')
    self.prediction = Conv2D(1, (3, 3), activation='sigmoid', padding='same')


  def call(self, x):
    l1 = self.conv1(x)

    l2 = self.conv2(l1)
    p1 = self.pool1(l2)

    l3 = self.conv3(p1)
    l4 = self.conv4(l3)
    p2 = self.pool2(l4)

    l5 = self.conv5(p2)
    l6 = self.conv6(l5)
    p3 = self.pool3(l6)   

    l7 = self.conv7(p3)
    l8 = self.conv8(l7)
    p4 = self.pool4(l8)

    l9 = self.conv9(p4)
    l10 = self.conv10(l9)

    up1 = self.up1(l10)
    concat1 = tf.concat([up1, l8], axis=-1)
    l11 = self.conv11(concat1)

    up2 = self.up2(l11)
    concat2 = tf.concat([up2, l6], axis=-1)
    l12 =  self.conv12(concat2)

    up3 = self.up3(l12)
    concat3 = tf.concat([up3, l4], axis=-1)
    l13 =  self.conv13(concat3)

    up4 = self.up4(l13)
    concat4 = tf.concat([up4, l2], axis=-1)
    l14 = self.conv14(concat4)
    predicition = self.prediction(l14)

    return predicition
