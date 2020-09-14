#!/usr/bin/env python
# coding: utf-8

# # Probability distributions
# 
# At the core of statistical methods inc. Bayesian inference is modelling the underlying phenomenon. By modeling we start to understand our measurements and can develop techniques to maximise information, e.g., by reducing noise in digital camera sensor or microphone.
# 
# ![image.png](attachment:image.png)

# In[1]:


# Let's have a bayesien look on the male/female classification problem
import matplotlib.pyplot as plt
import numpy as np
plt.xlabel('x: height [cm]')
plt.axis([140,200,-2,2])
x_1 = np.random.normal(165,5,20) # Measurements from the class 1
x_2 = np.random.normal(180,6,20) # Measurements from the class 2
plt.plot(x_1,np.zeros(len(x_1)),'rx')
plt.plot(x_2,np.zeros(len(x_2)),'ko')
plt.show()


# In[2]:


# Let's estimate the probability densities using Gaussian kernels
import scipy.stats as stats
x = np.arange(100,200,1)

# Kernel width is actually the variance of gaussians
kernel_width = 0.5

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


# In[3]:


# Some random experiments with 2D Gaussians
mu1 = [165,60]
cov1 = [[10,0],[0,5]]
mu2 = [180,80]
cov2 = [[6,0],[0,10]]
x1 = np.random.multivariate_normal(mu1, cov1, 100)
x2 = np.random.multivariate_normal(mu2, cov2, 100)
plt.plot(x1[:,0],x1[:,1],'rx')
plt.plot(x2[:,0],x2[:,1],'gx')


# ## References
# 
# C.M. Bishop (2006): Pattern Recognition and Machine Learning, Chapter 1-2.
