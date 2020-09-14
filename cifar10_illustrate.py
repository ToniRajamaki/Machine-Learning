import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from PIL import Image



#gt is the ground truth
#takes in two array parameters and compares the values
#returns information how the comparison went
def class_acc(pred,gt):

    truePositive = 0
    pictureQuantity = len(gt)
    for i in range(pictureQuantity):
       # print(pred[i], gt[i])
        if pred[i] == gt[i]:
            truePositive = truePositive + 1

    print('Predictions: ',truePositive,' / ', pictureQuantity)
    print('Percentage: ', 100 * truePositive / pictureQuantity,'%')
    return

# chooses random class from array of classes
def cifar10_classifier_random(dataSet):

    randomClassFromTestData = random.choice(dataSet)
    return randomClassFromTestData

#Takes in array of images and converts it to array with dimensions of [1x3072]
def convert_images_to_1d_arrays(images_array):

    converted_images = []
    length = len(images_array)
    rows, cols, colors = images_array[0].shape  # 32, 32, 3

    for i in range(length):
        img = images_array[i]
        oneD_image_array = img.reshape(rows * cols * colors) # from 3 matrixes to 1-D array
        converted_images.append(oneD_image_array)

    return converted_images


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


#Setting up TRAINING datasets
def setting_up_training_datasets():

    t_1 = unpickle('cifar-10-batches-py/data_batch_1')
    t_2 = unpickle('cifar-10-batches-py/data_batch_2')
    t_3 = unpickle('cifar-10-batches-py/data_batch_3')
    t_4 = unpickle('cifar-10-batches-py/data_batch_4')
    t_5 = unpickle('cifar-10-batches-py/data_batch_5')
    all_classes = []
    all_images = []

    #T1
    X_training = t_1["data"]
    Y_training = t_1["labels"]
    all_classes += Y_training
    X_training = X_training.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")  # Array of images
    Y_training = np.array(Y_training)  # Array of classes of each image 1D
    t1_images_1d = convert_images_to_1d_arrays(X_training)

    #T2
    X_training = t_2["data"]
    Y_training = t_2["labels"]
    all_classes += Y_training
    X_training = X_training.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")  # Array of images
    Y_training = np.array(Y_training)  # Array of classes of each image
    t2_images_1d = convert_images_to_1d_arrays(X_training)

    # T3
    X_training = t_3["data"]
    Y_training = t_3["labels"]
    all_classes += Y_training
    X_training = X_training.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")  # Array of images
    Y_training = np.array(Y_training)  # Array of classes of each image
    t3_images_1d = convert_images_to_1d_arrays(X_training)

    # T4
    X_training = t_4["data"]
    Y_training = t_4["labels"]
    all_classes += Y_training
    X_training = X_training.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")  # Array of images
    Y_training = np.array(Y_training)  # Array of classes of each image
    t4_images_1d = convert_images_to_1d_arrays(X_training)

    # T5
    X_training = t_5["data"]
    Y_training = t_5["labels"]
    all_classes += Y_training
    X_training = X_training.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")  # Array of images
    Y_training = np.array(Y_training)  # Array of classes of each image
    t5_images_1d = convert_images_to_1d_arrays(X_training)

    all_images = t1_images_1d + t2_images_1d + t3_images_1d + t4_images_1d + t5_images_1d

    np_all_images = np.array(all_images)
    np_all_classes = np.array(all_classes)



    return np_all_images, np_all_classes

setting_up_training_datasets()





#refactor this somewhere
training_image_classes = Y_training # Array of classes of each image

#Setting up TEST date

test_datadict = unpickle('cifar-10-batches-py/test_batch')
X_test = test_datadict["data"]
Y_test = test_datadict["labels"]
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8") # Array of images
Y_test = np.array(Y_test) # Array of classes of each image



#turha varmaa
#groundTruths = Y_training[0:picturesIterated]

#Random test
def test_class_acc_with_random_array():
    #creating prediction array from randomclass
    randomClass = cifar10_classifier_random(Y_training)
    temp = [randomClass]
    randomPredictionArray = len(Y_training) * temp
    #comparing randomArray with test_data classes
    class_acc(randomPredictionArray,Y_test)

#Renders pictures, not used
def for_each_picture():

    #original range was X.shape[0]
    for i in range(picturesIterated):
        if True:
            plt.figure(1);
            plt.clf()
            plt.imshow(X_training[i])
            plt.title(f"Image {i} label={label_names[Y_training[i]]} (num {Y_training[i]})")
            plt.pause(0.1)



#unclear TO DO
image_quantity = 10000
split_Y_test = Y_test[0:image_quantity]

#Splitting training image batch
training_images = X_training[0:image_quantity]
training_images_1D = convert_images_to_1d_arrays(training_images)

#Choosing how many images to classify
quanity_to_classify = 100
test_images = X_test[0:quanity_to_classify]
test_images_1D = convert_images_to_1d_arrays(test_images)


#runs closest distance for each image in test dataset, and calls class_acc to compare predictions with actual data
def cifar_10_classifier_1nn():

    prediction_array = []
    count = 0

    for i in range(image_quantity):
        count = count + 1
        print(count,' / ',image_quantity)
        predicted_class = closest_distance(test_images[i],training_images,training_image_classes,image_quantity)
        prediction_array.append(predicted_class)

    #Comparing predicted array with actual classes
    class_acc(prediction_array,split_Y_test)
    return

#Compares one image with every training image, finds the datapoint that is closest and returns it's class
#
# test_image is 1-D array
# training_images is 1-D array of every training image
# training_image_classes is array 1-D array of classes of each image from training set
# image_quantity is passed in for loop, so no need to calc it each time
def closest_distance(test_image, training_images,training_image_classes, image_quantity):

    closest_distance_so_far = math.inf
    prediction_class = math.inf

    for i in range(image_quantity):
        distance = np.sum(np.square(test_image - training_images[i])) #distance between images
        if distance < closest_distance_so_far:
            closest_distance_so_far = distance
            prediction_class = training_image_classes[i]

    return prediction_class


#t = time.time()
#test_class_acc_with_random_array()
#cifar10_classifier_1nn()
#print(time.time() - t)
