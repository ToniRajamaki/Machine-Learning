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

#Divides given data set to two vectors: classes  and images, type = list
def create_vector(path):

    t_set = unpickle(path)
    X_training = t_set["data"]
    Y_training = t_set["labels"]
    X_training = X_training.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")  # Array of images
    images_1d = convert_images_to_1d_arrays(X_training)
    return images_1d, Y_training

# Creates numpy arrays of test data
def setting_up_test_dataset():
    images, classes = create_vector('cifar-10-batches-py/test_batch')
    test_images = np.array(images)
    test_classes = np.array(classes)
    return test_images, test_classes

#Setting up TRAINING datasets
def setting_up_training_datasets():

    images_t1, classes_t1 = create_vector('cifar-10-batches-py/data_batch_1')
    images_t2, classes_t2 = create_vector('cifar-10-batches-py/data_batch_2')
    images_t3, classes_t3 = create_vector('cifar-10-batches-py/data_batch_3')
    images_t4, classes_t4 = create_vector('cifar-10-batches-py/data_batch_4')
    images_t5, classes_t5 = create_vector('cifar-10-batches-py/data_batch_5')

    # Combining lists to numpy array
    all_images = np.array(images_t1 + images_t2 + images_t3 + images_t4 + images_t5)
    all_classes = np.array(classes_t1 + classes_t2 + classes_t3 + classes_t4 + classes_t5)

    return all_images, all_classes

#Random test
def test_class_acc_with_random_array(test_classes):
    #creating prediction array from randomclass
    randomClass = cifar10_classifier_random(training_classes)
    temp = [randomClass]
    randomPredictionArray = np.array(len(training_classes) * temp)
    #comparing randomArray with test_data classes
    class_acc(randomPredictionArray,test_classes)

#Renders pictures, not used
def for_each_picture(picturesIterated,training_images,training_classes,label_names):


    for i in range(picturesIterated):
        if True:
            plt.figure(1);
            plt.clf()
            plt.imshow(training_images[i])
            plt.title(f"Image {i} label={label_names[training_classes[i]]} (num {training_classes[i]})")
            plt.pause(0.1)
    return

#runs closest distance for each image in test dataset, and calls class_acc to compare predictions with actual data
def cifar_10_classifier_1nn(training_images, training_classes,test_classes):

    prediction_array = []
    count = 0
    image_quantity = len(test_classes)
    print(len(training_images), len(training_classes), len(test_classes))

    for i in range(image_quantity):
        count = count + 1
        print(count,' / ',image_quantity)
        predicted_class = closest_distance(test_images[i],training_images,training_classes,image_quantity)
        prediction_array.append(predicted_class)

    #Comparing predicted array with actual classes
    class_acc(prediction_array,test_classes)
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

def set_training_data_quanity(num):
    tr_images = training_images[0:num]
    tr_classes = training_classes[0:num]
    return tr_images,tr_classes

def set_test_date_size(num):
    te_images = test_images[0:num]
    te_classes = test_classes[0:num]
    return te_images, te_classes

training_images, training_classes = setting_up_training_datasets()
test_images, test_classes = setting_up_test_dataset()

TEST_DATA_SIZE = 100        # 1 - 10,000
TRAINING_DATA_SIZE = 30000     # 1 - 50,0000
training_images, training_classes = set_training_data_quanity(TRAINING_DATA_SIZE)
test_images, test_classes = set_test_date_size(TEST_DATA_SIZE)


t = time.time()
cifar_10_classifier_1nn(training_images,training_classes,test_classes)
#test_class_acc_with_random_array(test_classes)
print(time.time() - t)
