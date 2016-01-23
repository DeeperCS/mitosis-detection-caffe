# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:40:21 2016

@author: joe
"""
import sys
sys.path.insert(0, '~/github/caffe/python')

import os,time
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import glob
from PIL import Image

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
    if len(X)==0:
        raise RuntimeError('No file was found in {}!'.format(filePath))
    X = np.concatenate(X, axis=0)  # list -> tensor
        
    return X
    
def pixel_generator(Y, tileRadius, batchSize):
    [s, height, width] = Y.shape
    yAll = np.unique(Y)
#    print('[yAll original:{}]'.format(yAll))
    # turn the content of Y to idx
    for yIdx, yOriginal in enumerate(yAll):
        Y[Y==yOriginal] = yIdx
        
    yAll = np.unique(Y)
#    print('[yAll:{}]'.format(yAll))
    
    assert(len(yAll)) > 0
    
    # using a bitMask to substract the edges
    bitMask = np.ones(Y.shape, dtype=np.bool)
    bitMask[:, 0:tileRadius, :] = 0
    bitMask[:, height-tileRadius:height,:] = 0
    bitMask[:, :, 0:tileRadius] = 0
    bitMask[:, :, width-tileRadius:width] = 0
    
    # determine how many instances of each class to report
    count = {}
    for y in yAll:
        count[y] = np.sum((Y==y) & bitMask)
    print('[train:]pixels per class label is: {} => {}'.format(count.keys(), count.values()))
    
    sampleCount = np.min(count.values())
    
    Idx = np.zeros([0, 3], dtype=np.int32)
    for y in yAll:
        tup = np.nonzero((Y==y) & bitMask)    # Iterate all slices
        Yi = np.column_stack(tup)    # Idx[1]:height Idx[2]:width
        # minor class will be all sampled
        # major class will be random sampled by the shuffle() below
        np.random.shuffle(Yi)
        Idx = np.vstack((Idx, Yi[:sampleCount,:]))
        
    # one last shuffle to mix all the classes together
    np.random.shuffle(Idx)
    
    #return
    for i in range(0, Idx.shape[0], batchSize):
        nRet = min(batchSize, Idx.shape[0]-i)
        yield Idx[i:i+nRet,:]   
        
def loadSolver(fileName):
    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(solverFile).read(), solverParam)
    # net parameter
    netFile = solverParam.net
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFile).read(), netParam)
   
    # model storage
    outDir = solverParam.snapshot_prefix
    if not os.path.isdir(outDir):
        os.mkdir(outDir)
        
    return solverParam, netParam
     
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
        
def train_loop(solverFile, X, Y, XTest, YTest, doTest=True): 
    solverParam, netParam = loadSolver(solverFile)     
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
    Ym = mirror_edges(Y, tileRadius)
   
    #######################
    # 3.yield idx
    #######################
    print('yAll is {}'.format(np.unique(Y)))
    
    X_batch = np.zeros(inputDim, dtype=np.float32)
    Y_batch = np.zeros(inputDim[0], dtype=np.float32)
    
    losses = np.zeros((solverParam.max_iter,), dtype=np.float32)
    accuries = np.zeros((solverParam.max_iter,), dtype=np.float32)

    #######################
    # Create solver
    #######################
    solver = caffe.get_solver(solverFile)   
    
    for name, blobs in solver.net.params.iteritems():
        for bIdx, b in enumerate(blobs):
            print('{}[{}] : {}'.format(name, bIdx, b.data.shape))
    
    # Iteration for a max_iter and record epoch
    currIter = 0
    currEpoch = 0  
    while currIter < solverParam.max_iter:
        #-------------------------------------------------------------------
        # iterate for a single epoch (All training data for one time)
        # The inner loop below is for a single epoch, which we may terminate
        # early if the max of iterations is reached.
        #-------------------------------------------------------------------
        currEpoch += 1    # one iteration of the iterator for a epoch
        loss_total_sigle_epoch = []
        print('[train]: epoch {} begin'.format(currEpoch))
        epochTime = 0
        _tmp = time.time()
        iterator = pixel_generator(Ym, tileRadius, inputDim[0])
        """Epoch Start
        """
        for Idx in iterator:
            if currIter >= solverParam.max_iter:
                break     
            # Extract tiles and labels from axis information from Idx(prepare for a batch)
            for j in range(Idx.shape[0]):
                left = Idx[j, 1] - tileRadius
                right = Idx[j, 1] + tileRadius + 1
                top = Idx[j, 2] - tileRadius
                bottom = Idx[j, 2] + tileRadius + 1
                X_batch[j, 0, :, :] = Xm[Idx[j, 0], left:right, top:bottom]    # Idx[j, 0]: slice id
                Y_batch[j] = Ym[Idx[j, 0], Idx[j, 1], Idx[j, 2]]
            
            # when a batch data prepared, put it to caffe
            X_batch /= 255
            solver.net.set_input_arrays(X_batch, Y_batch)
            
            # launch one step for gradient decent
            solver.step(1)
            
            # get loss and accuracy
            loss = float(solver.net.blobs['loss'].data)
            accuracy = float(solver.net.blobs['accuracy'].data)
            losses[currIter] = loss
            accuries[currIter] = accuracy
            
            currIter += 1  # one batch for a iteration
        
        """Epoch End
        """
        
        modelFileName = os.path.join(solverParam.snapshot_prefix, 'iter_{}.caffemodel'.format(currIter))
        solver.net.save(str(modelFileName))
        epochTime += time.time() - _tmp
        print('[train]: epoch {0} finished in {1:.2f} seconds, {2:.2f} min. Current Iteration:{3}'.format(currEpoch, epochTime, epochTime/60, currIter))
        print('[train]: loss:{}, saved {}'.format(np.mean(loss_total_sigle_epoch), modelFileName))       
                
        """Test Begin
        """
        
        if doTest:
            XTestm = mirror_edges(XTest, tileRadius)  
            YTestm = mirror_edges(YTest, tileRadius)
            accArray = []
            lossArray = []
            iteratorTest = pixel_generator(YTestm, tileRadius, inputDim[0])
            for Idx in iteratorTest:   
                # Extract tiles and labels from axis information from Idx(prepare for a batch)
                for j in range(Idx.shape[0]):
                    left = Idx[j, 1] - tileRadius
                    right = Idx[j, 1] + tileRadius + 1
                    top = Idx[j, 2] - tileRadius
                    bottom = Idx[j, 2] + tileRadius + 1
                    X_batch[j, 0, :, :] = XTestm[Idx[j, 0], left:right, top:bottom]    # Idx[j, 0]: slice id
                    Y_batch[j] = YTestm[Idx[j, 0], Idx[j, 1], Idx[j, 2]]
                
                # when a batch data prepared, put it to caffe
                X_batch /= 255
                solver.net.set_input_arrays(X_batch, Y_batch)
                
                # launch one step for gradient decent
                solver.net.forward()
                
                # get loss and accuracy
                loss = float(solver.net.blobs['loss'].data)
                accuracy = float(solver.net.blobs['accuracy'].data)
                accArray.append(accuracy)
                lossArray.append(loss)
            
            print('[test]: loss:{}, acc:{}'.format(np.mean(lossArray), np.mean(accArray)))
            
            """Test End
            """
            
    print('training completed')
    # return training informations
    return losses, accuries
    
if __name__=='__main__':
    caffe.set_mode_gpu()
    caffe.set_device(0)
    plotResult = True 
#    fileDataDir = 'dataset/data'
#    fileLabelDir = 'dataset/label'
#    print('[deploy:]: loading EM data file: %s' % fileDataDir)
#    print('[deploy:]: loading EM data file: %s' % fileLabelDir)
#    fileListData = sorted(glob.glob(fileDataDir+"/*."+'TIF'))
#    fileListLabel = sorted(glob.glob(fileLabelDir+"/*."+'TIF'))
#    
#    X = load_tiff_data(fileListData, np.float32)
#    Y = load_tiff_data(fileListLabel, np.float32)  
#    np.save('dataX.npy', X)
#    np.save('dataY.npy', Y)
    
    #######################
    # 1.load TIF to X
    #######################
    X = np.load('dataX.npy')
    Y = np.load('dataY.npy')
    
#    mid = 40
#    trainSlice = range(0,mid)
#    validSlice = range(mid,X.shape[0])
    mid = 10
    trainSlice = range(0,mid)
    validSlice = range(mid, 15)
    Xtrn = X[trainSlice, ...]
    Ytrn = Y[trainSlice, ...]
    Xval = X[validSlice, ...]
    Yval = Y[validSlice, ...]
    #######################
    # 2. Train
    #######################
    solverFile = 'solver-101.prototxt'
    
    losses, accuries = train_loop(solverFile, Xtrn, Ytrn, Xval, Yval)
    if plotResult:
        plt.plot(losses)
        plt.plot(accuries)
    
