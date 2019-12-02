from itertools import groupby
from heapq import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from itertools import product
from skimage import io,color


#Binary tree data structure & Huffman Encoder
#http://www.techrepublic.com/article/huffman-coding-in-python/

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


#Huffman encoder  
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

def entropy(labels,degree):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    [counts, bins]  = np.histogram(labels,np.append(np.unique(labels),np.inf))
    probs = counts.astype(float) / float(n_labels)
    

    # Compute standard entropy.
    if(degree==1):
        ent = -np.sum(np.multiply(probs,np.log2(probs)))
    else:
        ent = np.log2(np.sum(np.power(probs,degree)))/(1-degree)

    return ent

    



# Main()
    
img=mpimg.imread('lena512.bmp')

img_input=img.reshape([-1]).astype(str)
huffman_img = huffman(img_input)

# bpp = Scomp / NPixels - compressed data size (Scomp) & number of pixels (NPixels)
print('Bitrate of the original image')
print('Bits per pixel is ' + str(float(len(huffman_img[1])/float(len(img_input)))) + ' bpp')


windowSize = 11
borderSize = int((windowSize-1)/2)
(n,m)=img.shape

imgWithPadding = np.zeros([n+2*borderSize, m+2*borderSize])
(new_n,new_m)=imgWithPadding.shape
imgWithPadding[borderSize:new_n-borderSize, borderSize:new_m-borderSize] = img

print(img.shape)
print(imgWithPadding.shape)

# Construct prediction vectors
# 60 dim vectors as input with one label as central pixel
count = 0
trainingData = []
trainingLabels = []
for i in range(borderSize, new_n-borderSize):
    for j in range(borderSize, new_m-borderSize):
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
