import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean
import math

RED = 0;
GREEN = 1;
BLUE = 2;

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# reading some metainfo
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

def get_training_data(string):

    datadict = unpickle(string)
    images = datadict["data"] #Images
    classes = datadict["labels"] #Classes
    return images,classes

# images ( quanity , 32 , 32 ,3 )
def cifar_10_color(image_dataset):

    image_quanity = (image_dataset.shape[0]) # How many images we are processing
    image_rgb_mean = np.zeros((image_quanity, 3))

    for i in range(image_quanity):
        # Convert images to mean values of each color channel
        img = image_dataset[i]
        img_8x8 = resize(img, (8, 8))  # this is probably useless
        img_1x1 = resize(img, (1, 1))
        r_vals = img_1x1[:, :, 0].reshape(1 * 1)
        g_vals = img_1x1[:, :, 1].reshape(1 * 1)
        b_vals = img_1x1[:, :, 2].reshape(1 * 1)
        mu_r = r_vals.mean()  # mean value of this color channel for 1 image (decimal 0 - 1
        mu_g = g_vals.mean()
        mu_b = b_vals.mean()

        image_rgb_mean[i, :] = (mu_r, mu_g, mu_b)  # shape = ( imagequanity, 3 )

    return image_rgb_mean

#images_rgb_means shape  = ( quanity, 3 )
def cifar_10_naivebayes_learn(images_rgb_means,classes):

    images_quanity = len(classes);
    class_mean_sums = np.zeros((10,3,1))
    class_variance_sums = np.zeros((10,3,1))

    #array of 10 values, each value is quanity of how many images belong to this class
    class_measurements = np.zeros((10))
    luokkia_2 = 0

    #Calculating mean values for each class (r,g,b)
    for i in range(images_quanity):
        image_class = classes[i]
        image = images_rgb_means[i]

        class_mean_sums[image_class][RED][0] += image[RED]
        class_mean_sums[image_class][GREEN][0] += image[GREEN]
        class_mean_sums[image_class][BLUE][0] += image[BLUE]
        class_measurements[image_class] += 1

    for i in range(10):

        class_mean_sums[i][RED][0] = class_mean_sums[i][RED][0] / class_measurements[i]
        class_mean_sums[i][GREEN][0] = class_mean_sums[i][GREEN][0] / class_measurements[i]
        class_mean_sums[i][BLUE][0] = class_mean_sums[i][BLUE][0] / class_measurements[i]
        #print("for class: ",i," mean is : ", class_mean_sums[i][RED][0],class_mean_sums[i][GREEN][0],class_mean_sums[i][BLUE][0] )



    #starting variance calculations
    for i in range(images_quanity):
        image_class = classes[i]
        image = images_rgb_means[i]

        class_variance_sums[image_class][RED][0] += np.square(image[RED] - class_mean_sums[image_class][RED][0])
        class_variance_sums[image_class][GREEN][0] += np.square(image[GREEN] - class_mean_sums[image_class][GREEN][0])
        class_variance_sums[image_class][BLUE][0] += np.square(image[BLUE] - class_mean_sums[image_class][BLUE][0])

    print(class_variance_sums[image_class][RED][0])
    print(class_variance_sums)

    for i in range(10):

        class_variance_sums[i][RED][0] = class_variance_sums[i][RED][0] / class_measurements[i]
        class_variance_sums[i][GREEN][0] = class_variance_sums[i][GREEN][0] / class_measurements[i]
        class_variance_sums[i][BLUE][0] = class_variance_sums[i][BLUE][0] / class_measurements[i]

    print("aaaaaaaaa")
    print(class_variance_sums)

    return

def test(images):
    image = images[1]
    print("test")
    print(images.shape)
    print(image.shape)

    return


#Getting the training data from batch 1
images_1, classes_1 = get_training_data('cifar-10-batches-py/data_batch_1')
images_1 = images_1.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
classes_1 = np.array(classes_1)

# How many pictures we take from the batch ( 1 - 10,000 )
DATA_SET_QUANITY = 250;
images_1 = images_1[0:DATA_SET_QUANITY]
classes_1 = classes_1[0:DATA_SET_QUANITY]

rgb_means = cifar_10_color(images_1)
print(rgb_means.shape)

print()
cifar_10_naivebayes_learn(rgb_means,classes_1)
