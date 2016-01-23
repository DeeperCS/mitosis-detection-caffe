# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:40:21 2016

@author: joe
"""
"""
Detect single position using maxima border
"""
import sys
sys.path.insert(0, '/home/joe/github/caffe/python')

import os,time
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from PIL import Image
import glob
import matplotlib.pyplot as plt

def mirror_edges(X, nPixels):
    assert(nPixels>0)
    
    [s, height, width] = X.shape
    Xm = np.zeros([s, height+2*nPixels, width+2*nPixels], dtype=X.dtype)
    
    Xm[:, nPixels:height+nPixels, nPixels:width+nPixels] = X
    
    for i in range(s):
        # top left corner
        Xm[i, 0:nPixels, 0:nPixels] = np.fliplr(np.flipud(X[i, 0:nPixels, 0:nPixels]))
        # top right corner
        Xm[i, 0:nPixels, width+nPixels:width+2*nPixels] = np.fliplr(np.flipud(X[i, 0:nPixels, width-nPixels:width]))
        # bottom left corner
        Xm[i, height+nPixels:height+2*nPixels, 0:nPixels] = np.fliplr(np.flipud(X[i, height-nPixels:height, 0:nPixels]))
        # bottom right corner        
        Xm[i, height+nPixels:height+2*nPixels, width+nPixels:width+2*nPixels] = np.fliplr(np.flipud(X[i, height-nPixels:height, width-nPixels:width]))
        # top
        Xm[i, 0:nPixels, nPixels:width+nPixels] = np.flipud(X[i, 0:nPixels, 0:width])
        # bottom
        Xm[i, height+nPixels:height+2*nPixels, nPixels:width+nPixels] = np.flipud(X[i, height-nPixels:height, 0:width])
        # left
        Xm[i, nPixels:height+nPixels, 0:nPixels] = np.fliplr(X[i, 0:height, 0:nPixels])
        # right
        Xm[i, nPixels:height+nPixels, width+nPixels:width+2*nPixels] = np.fliplr(X[i, 0:height, width-nPixels:width])
    return Xm

def load_tiff_data(filePath, dtype='float32'):
    """ Loads data from a multilayer .tif file.  
    Returns result as a numpy tensor with dimensions (layers, width, height).
    """
    X = [];
    for dataFile in filePath:
        if not os.path.isfile(dataFile):
            raise RuntimeError('could not find file "%s"' % dataFile)
        
        # load the data from TIF files
        dataImg = Image.open(dataFile)
        
        Xi = np.array(dataImg, dtype=dtype)
        if len(Xi.shape)==3:
            Xi = Xi[...,0]
        Xi = np.reshape(Xi, (1, Xi.shape[0], Xi.shape[1]))  # add a slice dimension
        X.append(Xi)
            
    X = np.concatenate(X, axis=0)  # list -> tensor
        
    return X

def pixel_generator_test(X, tileRadius, batchSize):
    [s, height, width] = X.shape
    
    bitMask = np.ones(X.shape, dtype=bool)
    
    bitMask[:, 0:tileRadius, :] = 0
    bitMask[:, (height-tileRadius):height, :] = 0
    bitMask[:, :, 0:tileRadius] = 0
    bitMask[:, :, (width-tileRadius):width] = 0
    
    Idx = np.column_stack(np.nonzero(bitMask))
    
    for i in range(0, Idx.shape[0], batchSize):
        nRet = min(batchSize, Idx.shape[0]-i)
        yield Idx[i:(i+nRet), :]
    
def loadNet(fileName):
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(fileName).read(), netParam)        
    return netParam
    
def eval_cube(netFile, modelFile, X):
    netParam = loadNet(netFile)
    
    # input dim
    inputDim = None
    if netParam.layer[0].type == 'MemoryData':
        batch_size = netParam.layer[0].memory_data_param.batch_size
        channels = netParam.layer[0].memory_data_param.channels
        height = netParam.layer[0].memory_data_param.height
        width = netParam.layer[0].memory_data_param.width
        inputDim = [batch_size, channels, height, width]
    
    # inputDim[batch_size, channels, height, width]
    assert(inputDim!=None)
    tileEdge = inputDim[2]  
    tileRadius = tileEdge//2
    
    Xm = mirror_edges(X, tileRadius)  
    
    X_batch = np.zeros(inputDim, dtype=np.float32)
    Y_batch = np.zeros((inputDim[0],), dtype=np.float32)
    
    Yhat = None
    
    #######################
    # Create net
    #######################
    net = caffe.Net(netFile, modelFile, caffe.TEST)
    
    for name, blobs in net.params.iteritems():
        print('{} : {}'.format(name, blobs[0].data.shape))
    
    cnnTime = 0
    epochTime = 0
    tic = time.time()
    lastChatter = None
    
    iterator = pixel_generator_test(Xm, tileRadius, inputDim[0])
    for Idx in iterator:
        for j in range(Idx.shape[0]):        
            left = Idx[j,1] - tileRadius
            right = Idx[j,1] + tileRadius + 1
            top = Idx[j,2] - tileRadius
            bottom = Idx[j,2] + tileRadius + 1
            X_batch[j, 0, :, :] = Xm[Idx[j,0], left:right, top:bottom]
        
        _tmp = time.time()
        X_batch /= 255
        net.set_input_arrays(X_batch, Y_batch)
        net.forward()
        yiHat = net.blobs['prob'].data
        cnnTime += time.time() - _tmp
        
        nClasses = yiHat.shape[1]
        if Yhat is None: # on first iteration, create Yhat
            Yhat = -1*np.zeros((nClasses, Xm.shape[0], Xm.shape[1], Xm.shape[2]))
            
        # the size of yiHat may not match the remaining space in Yhat( not a full batch size)
        for j in range(nClasses):
            yijHat = np.squeeze(yiHat[:, j])
            assert(len(yijHat.shape)==1)
            Yhat[j, Idx[:,0], Idx[:,1], Idx[:,2]] = yijHat[:Idx.shape[0]]
        
        
        elapsed = (time.time() - tic) #/ 60.
        if (lastChatter is None) or ((elapsed - lastChatter) > 10):
            lastChatter = elapsed
            print('[deploy]: processed pixel at index {0} ({1:.2f} seconds elapsed; {2:.2f} CNN seconds)'.format(str(Idx[-1,:]), elapsed, cnnTime))
    
    epochTime += (time.time() - tic)
    Y = np.zeros((Yhat.shape[1], X.shape[1], X.shape[2]))
    # cut the edges
    Y[...] = np.squeeze(Yhat[0, :, tileRadius:tileRadius+X.shape[1], tileRadius:tileRadius+X.shape[2]])
    
    epochTime += time.time() - _tmp
    
    print('[deploy]: Finished! Epoch time: {0:0.2f} seconds!'.format(epochTime));
    return Y

def nms(Y, threshold=0.5, radius=30):
    assert(len(Y.shape)==2)
    maxValueAxises = np.nonzero(Y==np.max(Y))
    maxValueTop = np.min(maxValueAxises[0])
    maxValueDown = np.max(maxValueAxises[0])
    maxValueLeft = np.min(maxValueAxises[1])
    maxValueRight = np.max(maxValueAxises[1])
    
    maxPosH = (maxValueDown - maxValueTop)//2
    maxPosW = (maxValueRight - maxValueLeft)//2
    
    return maxValueTop+maxPosH,maxValueLeft+maxPosW
    
def drawCircle(X, ptH, ptW, radius=15):
    Xrgb = np.zeros([X.shape[0], X.shape[1], 3])
    Xrgb[:,:,0] = X[...]
    Xrgb[:,:,1] = X[...]
    Xrgb[:,:,2] = X[...]
    # draw center green point
    Xrgb[ptH, ptW, :] = [0, 255, 0]
    # draw circle
    for i in range(ptH-radius, ptH+radius+1, 1):
        for j in range(ptW-radius, ptW+radius+1, 1):
            if int(np.sqrt((i-ptH)**2+(j-ptW)**2)) == radius:
                Xrgb[i, j, :] = [255, 0, 0]              
    return Xrgb

def detect(filename, netFile, modelFile, outputFilename=''):
    caffe.set_mode_gpu()
    caffe.set_device(0)
    # 1.load TIF to X 
    fileListData = []
    fileListData.append(filename)
    
    if outputFilename=='':
        outputFilename = filename[:-4] + 'detected.tif'

    X = load_tiff_data(fileListData, np.float32)

    validSlice = range(0, 1)
    Xtest = X[validSlice, ...]

    #--------------
    # Create Caffe network and evaluate the slice
    #--------------
    Y = eval_cube(netFile, modelFile, Xtest)
    
    # if one slice, squeeze them
    Y = np.squeeze(Y)    
    Xtest = np.squeeze(Xtest)

    # NMS to Y
    pos = nms(Y)
    # Draw red circle and green dot on X slice
    Xrgb = drawCircle(Xtest, pos[0], pos[1])
#    plt.imshow(Xrgb)
    
    # turn to Image 
    """ Be sure using Image to save a image instead matplotlib(Make Image more dark , weird)
    """
    im = Image.fromarray(Xrgb.astype(np.uint8))
    
    im.save(outputFilename)
    
    
if __name__=='__main__':
    
    fileName = './data/osk-8000-B-new_t122c1.TIF'
    
    netFile = 'net-101-deploy.prototxt'
    
    modelFile = './models/iter_10000.caffemodel'
    
    detect(fileName, netFile, modelFile)
    
#    detect(fileName, netFile, modelFile, 'output.tif')

