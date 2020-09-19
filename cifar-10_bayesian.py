import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from skimage.transform import rescale, resize, downscale_local_mean


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

def get_training_data(string):

    datadict = unpickle(string)
    images = datadict["data"] #Images
    classes = datadict["labels"] #Classes
    return images,classes


images_1, classes_1 = get_training_data('cifar-10-batches-py/data_batch_1')



images_1 = images_1.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

classes_1 = np.array(classes_1)

X_mean = np.zeros((10000, 3))
for i in range(images_1.shape[0]):
    # Convert images to mean values of each color channel
    img = images_1[i]
    img_8x8 = resize(img, (8, 8))
    img_1x1 = resize(img, (1, 1))
    r_vals = img_1x1[:, :, 0].reshape(1 * 1)
    g_vals = img_1x1[:, :, 1].reshape(1 * 1)
    b_vals = img_1x1[:, :, 2].reshape(1 * 1)
    mu_r = r_vals.mean()
    mu_g = g_vals.mean()
    mu_b = b_vals.mean()
    X_mean[i, :] = (mu_r, mu_g, mu_b)

    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(img_8x8)
        plt.title(f"Image {i} label={label_names[classes_1[i]]} (num {classes_1[i]})")
        plt.pause(1)
