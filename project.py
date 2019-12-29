from itertools import groupby
from heapq import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from itertools import product
from skimage import io,color
import pickle

# Module necesare pentru construirea modelului de retea neurala
import tensorflow as tf
from tensorflow import keras
from keras.utils import *
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import Adam
from keras.optimizers import RMSprop

isTrainingStage = False

# Functia ce implementeaza algoritmul de codare Huffman
# Si clasa Node necesara constructiei unui arbore binar
# Implementarea celor doua provine din sursa:
# [http://www.techrepublic.com/article/huffman-coding-in-python/]

class Node(object):
	left = None
	right = None
	item = None
	weight = 0

	def __init__(self, i, w):
		self.item = i
		self.weight = w

	def setChildren(self, ln, rn):
		self.left = ln
		self.right = rn

	def __repr__(self):
		return "%s - %s — %s _ %s" % (self.item, self.weight, self.left, self.right)

	def __lt__(self, a):
		return (self.weight < a.weight)

# Functia de codare Huffman
# Returneaza un tuple ce contine fiecare element codat intr-o secventa de biti (sub forma de dictionar)
# si secventa finala de biti
def huffman(input):
    itemqueue =  [Node(a,len(list(b))) for a,b in groupby(sorted(input))]
    heapify(itemqueue)
    while len(itemqueue) > 1:
        l = heappop(itemqueue)
        r = heappop(itemqueue)
        n = Node(None, r.weight+l.weight)
        n.setChildren(l,r)
        heappush(itemqueue, n) 
        
    codes = {}
    def codeIt(s, node):
        if node.item:
            if not s:
                codes[node.item] = "0"
            else:
                codes[node.item] = s
        else:
            codeIt(s+"0", node.left)
            codeIt(s+"1", node.right)
    codeIt("",itemqueue[0])
    return codes, "".join([codes[a] for a in input])

# Pt o variabila aleatoare X ce ia k stari discrete, entropia se calculeaza:
# H(X) = -sum(fiecare k in K p(k)*log(p(k)))
def entropy(X):
    n = len(X)
    if n <= 1:
        return 0
    # Probabilitatea simbolurilor se afla cel mai usor de pe histograma
    # Raportand numarul de aparitii al unui simbol la numarul total de simboluri
    [counts, bins] = np.histogram(X,np.append(np.unique(X),np.inf))
    probs = counts.astype(float)/float(n)
    # Formula entropiei standard.
    ent = -np.sum(np.multiply(probs,np.log2(probs)))
    return ent


# Main() code    
img=mpimg.imread('lena512.bmp')

img_input=img.reshape([-1]).astype(str)
huffman_img = huffman(img_input)

# bpp = Scomp / NPixels - compressed data size (Scomp) / number of pixels (NPixels)
print('Bitrate of the original image')
print('Bits per pixel is ' + str(float(len(huffman_img[1])/float(len(img_input)))) + ' bpp')


windowSize = 11
borderSize = int((windowSize-1)/2)
(n,m)=img.shape

imgWithPadding = np.zeros([n+borderSize, m+borderSize])
(new_n,new_m)=imgWithPadding.shape
imgWithPadding[:n, :m] = img

print(img.shape)
print(imgWithPadding.shape)

# Construirea vectorilor de predictie
# vectori de dimensiune 60 ca intrare in reteaua neurala
# asociati valorii reale a pixelului central (valoarea ce se vrea prezisa)
count = 0
trainingData = []
trainingLabels = []
for i in range(borderSize, n):
    for j in range(borderSize, m):
        count += 1
        window = imgWithPadding[i-borderSize:i+borderSize+1, j-borderSize:j+borderSize+1]
        firstPartOfPredVector = (window[:borderSize, :]).reshape([-1])
        secondPartOfPredVector = (window[borderSize:borderSize+1, :borderSize]).reshape([-1])
        predVector = np.concatenate((firstPartOfPredVector, secondPartOfPredVector))
        trainingData.append(predVector)
        pixelValue = window[borderSize, borderSize] 
        trainingLabels.append(pixelValue)

print(window.shape)
print(n*m)
print(count)

# Normalizarea setului de date
trainingDataNotNormalized = trainingData
trainingLabelsNotNormalized = trainingLabels
trainingData = np.array(trainingData)/255
trainingLabels = np.array(trainingLabels)/255

if isTrainingStage:
    # For training
    # Construirea modelului de retea
    input = Input((60,), name='input')
    
    hidden = Dense(32, activation='relu', name='hidden_1')(input)
    dropout = Dropout(0.2, name='dropout_1')(hidden)
    hidden = Dense(16, activation='relu', name='hidden_2')(dropout)
    dropout = Dropout(0.2, name='dropout_2')(hidden)
    hidden = Dense(8, activation='relu', name='hidden_3')(dropout)
    dropout = Dropout(0.2, name='dropout_3')(hidden)
    hidden = Dense(4, activation='relu', name='hidden_4')(dropout)
    dropout = Dropout(0.2, name='dropout_4')(hidden)
    hidden = Dense(2, activation='relu', name='hidden_5')(dropout)
    dropout = Dropout(0.2, name='dropout_5')(hidden)

    output = Dense(1, activation='relu', name='output')(dropout)

    model = Model(input=[input], output=[output])
    opt = Adam(lr=5e-5, beta_1=0.3, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="mean_absolute_error", optimizer=opt)
    history = model.fit(trainingData, trainingLabels, epochs=100, batch_size=128)
    #model.save('mlpModel.h5') # pentru salvarea retelei in format de fisiere h5
else:
    model = load_model('mlpModel.h5')

predicted = [int(x[0]*255) for x in model.predict(trainingData)]
errorImage = [int((a_i - b_i)) for a_i, b_i in zip(list(trainingLabelsNotNormalized), predicted)]
huffman_error_img = huffman((np.array(errorImage)).astype(str))
print('Bitrate of the error image')
print('Bits per pixel is ' + str(float(len(huffman_error_img[1])/float(len(img_input)))) + ' bpp')

with open("errorImageFile", "wb") as file:
    pickle.dump(errorImage, file)