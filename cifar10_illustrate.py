import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random



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



def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


picturesIterated = 2000
#init predictions
temp = [1]
predictions = picturesIterated*temp






#datadict = unpickle('/home/kamarain/Data/cifar-10-batches-py/data_batch_1')
datadict = unpickle('cifar-10-batches-py/test_batch')


X = datadict["data"]
Y = datadict["labels"]
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

groundTruths = Y[0:picturesIterated]
class_acc(predictions,groundTruths)



#original range was X.shape[0]
for i in range(picturesIterated):
    # Show some images randomly
    #if random() > 0.999:
    if True:
        plt.figure(1);
        plt.clf()
       # plt.imshow(X[i])
        plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        plt.pause(0.01)
