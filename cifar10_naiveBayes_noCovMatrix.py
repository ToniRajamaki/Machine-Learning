
import numpy as np

from scipy.stats import norm

RED = 0;
GREEN = 1;
BLUE = 2;

def cifar_10_naivebayes_learn(images_rgb_means,classes):

    images_quanity = len(classes);
    class_mean_sums = np.zeros((10,3,1))
    class_std_sums = np.zeros((10,3,1))
    prior_p = np.zeros((10))

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


    #starting std calculations
    for i in range(images_quanity):
        image_class = classes[i]
        image = images_rgb_means[i]

        class_std_sums[image_class][RED][0] += np.square(image[RED] - class_mean_sums[image_class][RED][0])
        class_std_sums[image_class][GREEN][0] += np.square(image[GREEN] - class_mean_sums[image_class][GREEN][0])
        class_std_sums[image_class][BLUE][0] += np.square(image[BLUE] - class_mean_sums[image_class][BLUE][0])

    for i in range(10):

        class_std_sums[i][RED][0] = np.sqrt(class_std_sums[i][RED][0] / class_measurements[i])
        class_std_sums[i][GREEN][0] = np.sqrt(class_std_sums[i][GREEN][0] / class_measurements[i])
        class_std_sums[i][BLUE][0] = np.sqrt(class_std_sums[i][BLUE][0] / class_measurements[i])


    for i in range(10):
        prior_p[i] = class_measurements[i] / len(classes)

    return class_std_sums, class_mean_sums, prior_p






def cifar10_classifier_naivebayes(x,mu,sigma,prior_p):

    probabilities = np.ones(10)
    normpdf = norm.pdf
    for i in range(10):
        for j in range(3):
            probabilities[i] *= normpdf(x[j], mu[i][j], sigma[i][j])
            probabilities[i] *= prior_p[i]

    return np.argmax(probabilities)

def naive_bayes_classification(t_images, means, variances,prior_p):

    predictionArray = np.zeros((len(t_images)))

    count = len(t_images)
    for i in range(len(t_images)):
        count = count -1
        print(count)
        test_image = t_images[i]
        best_class = cifar10_classifier_naivebayes(test_image,means,variances,prior_p)
        predictionArray[i] = best_class

    return predictionArray