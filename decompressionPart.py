from itertools import groupby
from heapq import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from itertools import product
from skimage import io,color
import pickle
import math

# Module necesare pentru modelul de retea neurala
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import *
from keras.layers import *
from keras.models import *
from keras.callbacks import *


img=mpimg.imread('lena512.bmp')

windowSize = 11
borderSize = int((windowSize-1)/2)
(n,m)=img.shape

hL = img[:borderSize, :]
vL = img[:, :borderSize]

imgWithPadding = np.zeros([n+borderSize, m+borderSize])
(new_n,new_m)=imgWithPadding.shape
imgWithPadding[:borderSize, :m] = hL
imgWithPadding[:n, :borderSize] = vL

plt.imshow(img, cmap='gray')
plt.figure()
plt.imshow(imgWithPadding, cmap='gray')

with open("errorImageFile", "rb") as file:
    errorImage = pickle.load(file)
    k = 0
    model = load_model('mlpModel.h5')
    for i in range(borderSize, n):
        for j in range(borderSize, m):
            window = imgWithPadding[i-borderSize:i+borderSize+1, j-borderSize:j+borderSize+1]
            firstPartOfPredVector = (window[:borderSize, :]).reshape([-1])
            secondPartOfPredVector = (window[borderSize:borderSize+1, :borderSize]).reshape([-1])
            predVector = np.concatenate((firstPartOfPredVector, secondPartOfPredVector))
            predVector = predVector/255
            predict = int((255*model.predict(np.array([predVector])))[0])
            error = errorImage[k]
            finalPredictedValue = predict+error
            imgWithPadding[i,j] = finalPredictedValue
            k += 1

            
plt.figure()
plt.imshow(imgWithPadding[:n,:m], cmap='gray')