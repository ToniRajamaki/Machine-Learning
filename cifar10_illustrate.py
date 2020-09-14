import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image



#gt is the ground truth
def class_acc(pred,gt):

    truePositive = 0
    pictureQuantity = len(gt)
    for i in range(pictureQuantity):
        print(pred[i], gt[i])
        if pred[i] == gt[i]:
            truePositive = truePositive + 1

    print('Predictions: ',truePositive,' / ', pictureQuantity)
    print('Percentage: ', 100 * truePositive / pictureQuantity,'%')

    return

# chooses random class from temp data, then calls class_acc with prediction array
# of randomclass
def cifar10_classifier_random(x):

    randomClassFromTestData = random.choice(x)
    return randomClassFromTestData


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


picturesIterated = 1

#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/data_batch_1')
datadict = unpickle('cifar-10-batches-py/test_batch')


X = datadict["data"]
Y = datadict["labels"]
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

groundTruths = Y[0:picturesIterated]

def test_class_acc_with_random_array():
    #creating prediction array from randomclass
    randomClass = cifar10_classifier_random(Y)
    temp = [randomClass]
    randomPredictionArray = len(Y) * temp

    class_acc(randomPredictionArray,Y)


def for_each_picture():

    #original range was X.shape[0]
    for i in range(picturesIterated):
        if True:
            plt.figure(1);
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")

            #print(X[i])
            img = X[i]
            rows, cols, colors = img.shape  # gives dimensions for RGB array
            oneD_image_array = img.reshape(rows * cols * colors)



            plt.pause(1000)


for_each_picture()
