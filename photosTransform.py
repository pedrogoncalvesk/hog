import random
import sys
import os
import numpy as np
from math import cos, sin
import cv2
import matplotlib.pyplot as plt
from cv2.cv2 import HOGDescriptor
from skimage.exposure import  exposure
from skimage.io import imread, imshow
from skimage.feature import hog

#help(cv2.HOGDescriptor())

A = cv2.imread("dataset1/testes/train_5a_01000.png")
# To open
# imshow ( "dataset1/testes/train_5a_01000.png")
# plt.show("dataset1/testes/train_5a_01000.png")


A.shape

# Test if all images planes
a1 = A[:, :, 0]
a2 = A[:, :, 1]
a3 = A[:, :, 2]

#print("All image plans are equal:")
#print(np.array_equal(a1, a2) and np.array_equal(a1, a3))

# HOG CODE
winSize = (128, 128) #tamanho da Imagem 128x128
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (16, 16)
nbins = 9 # números de angulos do esquadro
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = True

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                        L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)
descriptor = hog.compute(A)
hog.save("hog.xml")
x = repr(descriptor)

print(descriptor)
print(len(descriptor))


imshow(descriptor)
plt.show()

