import pickle
import numpy as np


from scipy.stats import multivariate_normal




import cifar10_naiveBayes_noCovMatrix as nCm
from skimage.transform import rescale, resize, downscale_local_mean


RED = 0;
GREEN = 1;
BLUE = 2;

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


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


def get_training_data(string):

    datadict = unpickle(string)
    images = datadict["data"] #Images
    classes = datadict["labels"] #Classes
    return images,classes





def exercise_1():

    #Exercise 1, no COV matrix
    stds, means, prior_p = nCm.cifar_10_naivebayes_learn(rgb_means_images_1,classes_1)
    predictionArray = nCm.naive_bayes_classification(rgb_means_images_t, means, stds, prior_p)
    class_acc(predictionArray,t_classes)
    return



def excercise_2(training_images, training_classes):

    means, prior_p = cifar_10_bayes_learn(training_images, training_classes)
    images_sorted_by_class = divide_images_to_classes(training_images,training_classes)
    cov_matrixes = calculate_cov_matrixes(images_sorted_by_class)

    predictionArray = bayes_classification(rgb_means_images_t, means, cov_matrixes, prior_p)
    class_acc(predictionArray, t_classes)
    return


def divide_images_to_classes(images, classes):

    image_quanity = len(classes)
    images_by_classes = [ [], [], [], [], [], [], [], [], [], [],]


    for i in range(image_quanity):
        current_class = classes_1[i]
        current_image = images[i]
        images_by_classes[current_class].append(current_image)

    class_array = np.array(images_by_classes,dtype = object)
    for i in range(10):
        temp = np.array(class_array[i])
        class_array[i] = temp

    return class_array






#1
def bayes_classification(t_images, means, cov_matrixes,prior_p):

    predictionArray = np.zeros((len(t_images)))

    count = len(t_images)
    for i in range(len(t_images)):
        count = count -1
        print(count)
        test_image = t_images[i]
        best_class = get_best_matching_class(test_image,means,cov_matrixes,prior_p)
        predictionArray[i] = best_class

    return predictionArray


#2, probability is calculated here
def get_best_matching_class(x,mu, cov_matrixes,prior_p):

    probabilities = np.ones(10)

    for i in range(10): #classes

        # y = multivariate_normal(mu[i], cov_matrixes[i]).pdf(x)


        y = multivariate_normal.logpdf(x,mean = mu[i], cov=cov_matrixes[i])
        probabilities[i] = (y*prior_p[i])


    return np.argmax(probabilities)


def calculate_cov_matrixes(images_by_classes,N):

    sigma = np.empty(shape=(10, N*N*3, N*N*3))

    for i in range(10):
        sigma[i] = np.cov(images_by_classes[i],rowvar=False)

    return sigma


# images ( quanity , 32 , 32 ,3 )
def cifar_10_color(image_dataset, N = 1):

    image_quanity = (image_dataset.shape[0]) # How many images we are processing
    image_rgb_mean = np.zeros(shape =(image_quanity, N*N*3))


    for image in range(image_quanity):
        # Convert images to mean values of each color channel
        img = image_dataset[image]
        img_NxN = resize(img, (N,N))


        #Taking values for each channel
        r_vals = img_NxN[:, :, 0].reshape(N * N)
        g_vals = img_NxN[:, :, 1].reshape(N * N)
        b_vals = img_NxN[:, :, 2].reshape(N * N)


        if(N == 1):
            #calculating means for each channel
            mu_r = r_vals.mean()
            mu_g = g_vals.mean()
            mu_b = b_vals.mean()
            image_rgb_mean[image, :] = (mu_r, mu_g, mu_b)  # shape = ( imagequanity, 3 )
        else:

            all_vals = np.concatenate((r_vals, g_vals, b_vals))
            image_rgb_mean[image] = all_vals




    return image_rgb_mean



def means_for_each_class(class_array,N):

    means = []
    for i in range(10):
        means.append(np.mean(class_array[i], axis=0))

    return means


def cifar_10_bayes_learn(images_rgb_means, classes):

    prior_p = np.zeros((10))
    class_array = divide_images_to_classes(rgb_means_images_1, classes_1)
    class_measurements = np.zeros((10))
    means = means_for_each_class(class_array, N)
    cov_matrixes = calculate_cov_matrixes(class_array, N)
    for i in range(10):
        prior_p[i] = len(class_array[i]) / len(classes)


    return means, cov_matrixes,prior_p




#Getting the training data from batch 1
images_1, classes_1 = get_training_data('cifar-10-batches-py/data_batch_1')
t_images, t_classes = get_training_data('cifar-10-batches-py/test_batch')
images_1 = images_1.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
t_images = t_images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
classes_1 = np.array(classes_1)
t_classes = np.array(t_classes)

# How many pictures we take from the batch ( 1 - 10,000 )
DATA_SET_QUANITY = 10000;
TEST_DATA = 1000
N = 1
images_1 = images_1[0:DATA_SET_QUANITY]
classes_1 = classes_1[0:DATA_SET_QUANITY]
t_images = t_images[0:TEST_DATA]
t_classes = t_classes[0:TEST_DATA]
rgb_means_images_t = cifar_10_color(t_images,N)
rgb_means_images_1 = cifar_10_color(images_1,N)


p_means, p_cov_matrixes, p_prior_probability = cifar_10_bayes_learn(rgb_means_images_1,classes_1)
predictionArray = bayes_classification(rgb_means_images_t, p_means, p_cov_matrixes, p_prior_probability)
class_acc(predictionArray, t_classes)


