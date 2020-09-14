# Let's estimate the probability densities using Gaussian kernels
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
x_1 = np.random.normal(165,5,20) # Measurements from the class 1
x_2 = np.random.normal(180,6,20) # Measurements from the class 2
x = np.arange(100,200,1)

# Kernel width is actually the variance of gaussians
kernel_width = 2.4

# Output value is Gaussian kernel multiplied by all positive samples
yval1 = np.zeros(len(x))
for xind, xval in enumerate(x):
    yval1[xind] = sum(stats.norm.pdf(x_1, xval, kernel_width))
yval2 = np.zeros(len(x))
for xind, xval in enumerate(x):
    yval2[xind] = sum(stats.norm.pdf(x_2, xval, kernel_width))

# We normalize values to sum one (this is ad hoc)
plt.plot(x, yval1/sum(yval1),'r-')
plt.plot(x, yval2/sum(yval2),'g-')

# For comparison let's also print Gaussians
mu1 = np.mean(x_1)
mu2 = np.mean(x_2)
sigma1 = np.std(x_1)
sigma2 = np.std(x_2)
x = np.arange(100,200,1)
plt.plot(x, stats.norm.pdf(x, mu1, sigma1),'r--')
plt.plot(x, stats.norm.pdf(x, mu2, sigma2),'g--')
plt.show()