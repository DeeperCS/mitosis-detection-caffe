# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:40:21 2016

@author: joe
"""
import sys
sys.path.insert(0, '~/github/caffe/python')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import caffe

def classify(imPath):  
    net = caffe.Net('net.prototxt', '_iter_20000.caffemodel', caffe.TEST)
    
    #im_i = Image.open('test1.TIF')    # array([[  9.99602258e-01,   3.97737342e-04]], dtype=float32)
    im_i = Image.open(imPath)    # array([[  8.97469406e-04,   9.99102592e-01]], dtype=float32)
    im_r = im_i.resize([101, 101])
    im = np.asarray(im_r)
    if im.shape[2]>1:
        im = im[:,:,0]
    #    im = im[np.newaxis, :, :]
    
    im_input = im[np.newaxis, np.newaxis, :, :]
    net.blobs['data'].reshape(*im_input.shape)
    net.blobs['data'].data[...] = im_input
    
    out = net.forward()
#    print out
    return out['prob'][0][0]
    
if __name__=='__main__':
    picNames = ['mito1.TIF', 'mito2.TIF', 'mito3.TIF', 'mito4.TIF']
    for idx, picName in enumerate(picNames):
        #-----------------------------#
        # mitosis probability predict
        prob = classify(picName)
        #-----------------------------#
        print('{}\'s probability is {}'.format(picName, prob))
        plt.subplot(1,4,idx+1)
        plt.imshow(np.asarray(Image.open(picName)))
        plt.title(prob)