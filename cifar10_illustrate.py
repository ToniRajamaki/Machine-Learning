import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from PIL import Image



#gt is the ground truth
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

# chooses random class from temp data, then calls class_acc with prediction array
# of randomclass
def cifar10_classifier_random(dataSet):

    randomClassFromTestData = random.choice(dataSet)
    return randomClassFromTestData


def convert_images_to_1d_arrays(images_array):

    converted_images = []
    length = len(images_array)
    rows, cols, colors = images_array[0].shape # 32, 32, 3

    for i in range(length):
        img = images_array[i]
        oneD_image_array = img.reshape(rows * cols * colors) # from 3 matrixes to 1-D array
        converted_images.append(oneD_image_array)

    return converted_images


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


picturesIterated = 1

datadict = unpickle('cifar-10-batches-py/data_batch_1')
X_training = datadict["data"]
Y_training = datadict["labels"]
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
X_training = X_training.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y_training = np.array(Y_training)
training_image_classes = Y_training

test_datadict = unpickle('cifar-10-batches-py/test_batch')
X_test = test_datadict["data"]
Y_test = test_datadict["labels"]
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y_test = np.array(Y_test)





#turha varmaa
#groundTruths = Y_training[0:picturesIterated]

def test_class_acc_with_random_array():
    #creating prediction array from randomclass
    randomClass = cifar10_classifier_random(Y_training)
    temp = [randomClass]
    randomPredictionArray = len(Y_training) * temp
    class_acc(randomPredictionArray,Y_training)


def for_each_picture():

    #original range was X.shape[0]
    for i in range(picturesIterated):
        if True:
            plt.figure(1);
            plt.clf()
            #plt.imshow(X_training[i])
            plt.title(f"Image {i} label={label_names[Y_training[i]]} (num {Y_training[i]})")
            #plt.pause(0.1)



for_each_picture()

image_quantity = 1500
split_Y_test = Y_test[0:image_quantity]

training_images = X_training[0:image_quantity]
converted_training_images = convert_images_to_1d_arrays(training_images)

test_images = X_test[0:image_quantity]
converted_test_images = convert_images_to_1d_arrays(test_images)


def cifar_10():
    prediction_array = []
    count = 0

    for i in range(image_quantity):
        count = count + 1
        print(count,' / ',image_quantity)
        distance, predicted_class = closest_distance(test_images[i],training_images,training_image_classes,image_quantity)

        prediction_array.append(predicted_class)

    class_acc(prediction_array,split_Y_test)
    return

def closest_distance(test_image, training_images,training_image_classes, image_quantity):
    closest_distance_so_far = math.inf
    prediction_class = 10

    for i in range(image_quantity):

        distance = np.sum(np.square(test_image - training_images[i]))
        if distance < closest_distance_so_far:
            closest_distance_so_far = distance
            prediction_class = training_image_classes[i]
    return closest_distance_so_far, prediction_class


cifar_10()
