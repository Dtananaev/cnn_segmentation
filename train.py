import tensorflow as tf 
from dataset_tutoriual import load_data


dataset_dir="./data/stage1_train"
# Depth estimation with CNN
dataset = load_data(dataset_dir)
for input, label in dataset:
    #predict = model.fit(input, label)
    #loss = loss(predict, Y)
    print("image {}, mask {}".format(image.shape, mask.shape))

