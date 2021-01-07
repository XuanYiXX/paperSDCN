# -*- coding: utf-8 -*-
from __future__ import print_function, division

import keras as K
import keras.layers as L
import numpy as np
import os
import time
import h5py
import argparse
import matplotlib.pyplot as plt
from data_util import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import regularizers
from config import variables
from keras.layers import Lambda
from keras import initializers
# ===================cascade net=============
class HSI_2Branchnet(object):
    def __init__(self,strategy):
        NUM_CLASS=variables.NUM_CLASS
        ksize=variables.ksize
        hchn=variables.hchn
        filters = [32,64, 128, 256, 512]
        dilations = [1, 3, 5, 7]
        self.input_hsi2D = L.Input((ksize, ksize, hchn))
        getindicelayer = Lambda(lambda x: x[:,variables.r,variables.r,:,np.newaxis])
        self.input_hsi1D = getindicelayer(self.input_hsi2D)
        # self.first=L.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=None)
        self.conv0_0 = L.Conv2D(64, (3, 3), padding='same')(self.input_hsi2D)
        self.conv0_1 = L.BatchNormalization(axis=-1)(self.conv0_0)
        self.conv0_2 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.conv0_1)
        self.conv0_3 = L.Conv2D(128, (1,1), padding='same')(self.conv0_2)
        self.conv0_4 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.conv0_3)
        self.conv0_5 = L.MaxPool2D(pool_size=(2, 2),padding='same')(self.conv0_4)
        self.conv0_6 = L.Flatten()(self.conv0_5)
       
        self.conv1_0 = L.Conv1D(64, 3, padding='valid')(self.input_hsi1D) 
        self.conv1_1 = L.BatchNormalization(axis=-1)(self.conv1_0)
        self.conv1_2 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.conv1_1)
        self.conv1_3 = L.Conv1D(128,3, padding='valid')(self.conv1_2)  
        self.conv1_4 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.conv1_3)
        self.conv1_5 = L.MaxPool1D(pool_size=2, padding='same')(self.conv1_4)
        self.conv1_6 = L.Flatten()(self.conv1_5)

        self.conv7=L.concatenate([self.conv0_6,self.conv1_6])
        # self.conv7=self.conv0_6
        self.conv8=L.Dropout(0.5)(self.conv7)
        self.conv9=L.Dense(NUM_CLASS, 
                   activation='linear',kernel_regularizer=regularizers.l2(0.5))(self.conv8)
                #    activation='softmax')(self.conv8)
        self.model = K.models.Model([self.input_hsi2D],self.conv9)
        if strategy=='Adam':
            opti = K.optimizers.Adam(lr=0.001)
        if strategy=='SGD_005_1e-6_0':
            opti = K.optimizers.SGD(lr=0.005, momentum=1e-6)
        if strategy=='SGD_001_95_1e-5':
            opti=K.optimizers.SGD(lr=0.001, momentum=0.95,decay=1e-5)
        if strategy=='SGD_001_99_1e-3':
            opti=K.optimizers.SGD(lr=0.001, momentum=0.99,decay=1e-3)
        kwargs = K.backend.moving_averages
        self.model.compile(optimizer=opti, 
                           loss='categorical_squared_hinge',#'mean_squared_error',
                        #    loss='categorical_crossentropy',
                           metrics=['acc'])
        # self.model.summary()
def cascade_block(input, nb_filter, kernel_size=3):
    conv1_1 = L.Conv2D(nb_filter * 2, (kernel_size, kernel_size), padding='same')(input)  # nb_filters*2
    conv1_2 = L.Conv2D(nb_filter, (1, 1),padding='same')(conv1_1)  # nb_filters
    relu1 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv1_2)

    conv2_1 = L.Conv2D(nb_filter * 2, (kernel_size, kernel_size),padding='same')(relu1)  # nb_filters*2
    conv2_1 = L.add([conv1_1, conv2_1])
    conv2_1 = L.BatchNormalization(axis=-1)(conv2_1)

    conv2_2 = L.Conv2D(nb_filter, (1, 1), padding='same')(conv2_1)  # nb_filters
    relu2 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv2_2)
    relu2 = L.add([relu1, relu2])
    
    conv3_1 = L.Conv2D(nb_filter , (1, 1),padding='same')(relu2)  # nb_filters*2
    relu3 = L.advanced_activations.LeakyReLU(alpha=0.2)(conv3_1)
    return relu3
class HSI_2Branchnet_LIDAR(object):
    def __init__(self,strategy='adam',TWOsingle_weight=None):
        """
        input:
            input_shape: input shape of HSI or LIDAR
        """
        NUM_CLASS=variables.NUM_CLASS
        ksize=variables.ksize
        lchn=variables.lchn

        filters = [32,64, 128, 256, 512]
        dilations = [1, 3, 5, 7]

        TWOsingle =HSI_2Branchnet(strategy)
        TWOsingle.model.load_weights(TWOsingle_weight)
        TWOsingle.model.trainable = False
        TWOsingle_in = TWOsingle.model.input
        self.lidar_in=L.Input((ksize, ksize, lchn))
        if lchn==1:
            getindicelayer = Lambda(lambda x: x[:,variables.r,variables.r,np.newaxis])
        else:
            getindicelayer = Lambda(lambda x: x[:,variables.r,variables.r,:,np.newaxis])
        self.lidar_in1D = getindicelayer(self.lidar_in)
        # self.first=L.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=None)
        self.conv2_0 = L.Conv2D(128, (3, 3), padding='same')(self.lidar_in)
        self.conv3_0 = L.Conv1D(32, 3, padding='same')(self.lidar_in1D)

        self.merge0_0 = L.concatenate([TWOsingle.conv0_0,self.conv2_0], axis=-1)
        self.merge0_1 = L.BatchNormalization(axis=-1)(self.merge0_0)
        self.merge0_2 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge0_1)
        self.merge0_3 = L.concatenate([TWOsingle.conv0_2,self.merge0_2], axis=-1)
        self.merge0_4 = L.Conv2D(256, (1,1), padding='same')(self.merge0_3)
        self.merge0_5 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge0_4)
        self.merge0_6 = L.MaxPool2D(pool_size=(2, 2),padding='same')(self.merge0_5)
        self.merge0_7 = L.concatenate([TWOsingle.conv0_5,self.merge0_6], axis=-1)
        self.merge0_8 = L.Dropout(0.5)(self.merge0_7)
        # self.merge0_8 = L.Conv2D(256, (3,3), padding='same')(self.merge0_8)
        self.merge0_8 = L.BatchNormalization(axis=-1)(self.merge0_8)
        self.merge0_8 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge0_8)
        self.merge0_8 = L.Flatten()(self.merge0_8)
        
 
        # merge1_0 = L.concatenate([TWOsingle.conv1_0,conv3_0], axis=-1)
        self.merge1_1 = L.BatchNormalization(axis=-1)(self.conv3_0)
        self.merge1_2 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge1_1)
        # merge1_3 = L.concatenate([TWOsingle.conv1_2,merge1_2], axis=-1)
        self.merge1_4 = L.Conv1D(256, 3, padding='same')(self.merge1_2)
        self.merge1_5 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge1_4)
        self.merge1_6 =L.MaxPool1D(pool_size=2,padding='same')(self.merge1_5)
        self.merge1_6 = L.Flatten()(self.merge1_6)
        self.merge1_7 = L.concatenate([TWOsingle.conv1_6,self.merge1_6], axis=-1)
        self.merge1_8 = L.Dropout(0.5)(self.merge1_7)
        # self.merge1_8 = L.Conv1D(256, 3, padding='same')(self.merge1_8)
        self.merge1_8 = L.BatchNormalization(axis=-1)(self.merge1_8)
        self.merge1_8 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge1_8)
        # merge1_8 = L.Flatten()(merge1_8)
        self.merge2_0 = L.concatenate([self.merge1_8,self.merge0_8], axis=-1)
        # self.merge2_0 = self.merge0_8
        # merge2_1 = L.Conv1D(128, 3, padding='same')(merge2_0)
        # merge2_2 = L.advanced_activations.LeakyReLU(alpha=0.2)(merge2_0)
        # merge2_3 =L.MaxPool2D(pool_size=(2,2),padding='same')(merge2_2)
        # merge2_3 = L.Flatten()(merge2_0)


        # def l2_svm(ll=1,rr=2):
        #     def categorical(y_true, y_pred):
        #         """
        #         hinge with 0.5*W^2 ,SVM
        #         """
        #         y_true = 2. * y_true - 1 # trans [0,1] to [-1,1]，注意这个，svm类别标签是-1和1
        #         vvvv = K.maximum(1. - y_true * y_pred, 0.) # hinge loss，参考keras自带的hinge loss
        #         # vvv = K.square(vvvv) # 文章《Deep Learning using Linear Support Vector Machines》有进行平方
        #         vv = K.sum(vvvv, 1, keepdims=False)  #axis=len(y_true.get_shape()) - 1
        #         v = K.mean(vv, axis=-1)
        #         return v
        #     return  categorical

        self.merge2_4=L.Dense(NUM_CLASS, 
                           activation='linear',kernel_regularizer=regularizers.l2(0.5))(self.merge2_0)
                    # activation='softmax')(self.merge2_0)

        self.model = K.models.Model([TWOsingle_in,self.lidar_in],self.merge2_4)
        if strategy=='Adam':
            opti = K.optimizers.Adam(lr=0.0001)
        if strategy=='SGD_005_1e-6_0':
            opti = K.optimizers.SGD(lr=0.005, momentum=1e-6)
        if strategy=='SGD_001_95_1e-5':
            opti=K.optimizers.SGD(lr=0.001, momentum=0.95,decay=1e-5)
        if strategy=='SGD_001_99_1e-3':
            opti=K.optimizers.SGD(lr=0.001, momentum=0.99,decay=1e-3)
        # optm=K.optimizers.SGD(lr=0.001, momentum=0.99,decay=1e-3)
        self.model.compile(optimizer=opti,
                        #    loss='categorical_crossentropy', metrics=['acc'])
                             loss='categorical_squared_hinge', metrics=['acc'])
                            # loss=l2_svm(), metrics=['acc'])
        # self.model.summary()
# class HSI_2Branchnet_LIDAR(object):
#     def __init__(self,strategy='adam',TWOsingle_weight=None):
#         """
#         input:
#             input_shape: input shape of HSI or LIDAR
#         """
#         NUM_CLASS=variables.NUM_CLASS
#         ksize=variables.ksize
#         lchn=variables.lchn

#         filters = [32,64, 128, 256, 512]
#         dilations = [1, 3, 5, 7]

#         TWOsingle =HSI_2Branchnet(strategy)
#         TWOsingle.model.load_weights(TWOsingle_weight)
#         TWOsingle.model.trainable = False
#         TWOsingle_in = TWOsingle.model.input
#         self.lidar_in=L.Input((ksize, ksize, lchn))
#         if lchn==1:
#             getindicelayer = Lambda(lambda x: x[:,variables.r,variables.r,np.newaxis])
#         else:
#             getindicelayer = Lambda(lambda x: x[:,variables.r,variables.r,:,np.newaxis])
#         self.lidar_in1D = getindicelayer(self.lidar_in)
#         # self.first=L.initializers.RandomNormal(mean=0.0,stddev=0.05,seed=None)
#         self.conv2_0 = L.Conv2D(128, (3, 3), padding='same')(self.lidar_in)
#         self.conv3_0 = L.Conv1D(32, 3, padding='same')(self.lidar_in1D)

#         self.merge0_0 = L.concatenate([TWOsingle.conv0_0,self.conv2_0], axis=-1)
#         self.merge0_1 = L.BatchNormalization(axis=-1)(self.merge0_0)
#         self.merge0_2 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge0_1)
#         self.merge0_3 = L.concatenate([TWOsingle.conv0_2,self.merge0_2], axis=-1)
#         self.merge0_4 = L.Conv2D(256, (1,1), padding='same')(self.merge0_3)
#         self.merge0_5 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge0_4)
#         self.merge0_6 = L.MaxPool2D(pool_size=(2, 2),padding='same')(self.merge0_5)
#         self.merge0_7 = L.concatenate([TWOsingle.conv0_5,self.merge0_6], axis=-1)
#         self.merge0_8 = L.Dropout(0.5)(self.merge0_7)
#         # self.merge0_8 = L.Conv2D(256, (3,3), padding='same')(self.merge0_8)
#         self.merge0_8 = L.BatchNormalization(axis=-1)(self.merge0_8)
#         self.merge0_8 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge0_8)
#         self.merge0_8 = L.Flatten()(self.merge0_8)
        
 
#         # merge1_0 = L.concatenate([TWOsingle.conv1_0,conv3_0], axis=-1)
#         self.merge1_1 = L.BatchNormalization(axis=-1)(self.conv3_0)
#         self.merge1_2 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge1_1)
#         # merge1_3 = L.concatenate([TWOsingle.conv1_2,merge1_2], axis=-1)
#         self.merge1_4 = L.Conv1D(256, 3, padding='same')(self.merge1_2)
#         self.merge1_5 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge1_4)
#         self.merge1_6 =L.MaxPool1D(pool_size=2,padding='same')(self.merge1_5)
#         self.merge1_6 = L.Flatten()(self.merge1_6)
#         self.merge1_7 = L.concatenate([TWOsingle.conv1_6,self.merge1_6], axis=-1)
#         self.merge1_8 = L.Dropout(0.5)(self.merge1_7)
#         # self.merge1_8 = L.Conv1D(256, 3, padding='same')(self.merge1_8)
#         self.merge1_8 = L.BatchNormalization(axis=-1)(self.merge1_8)
#         self.merge1_8 = L.advanced_activations.LeakyReLU(alpha=0.2)(self.merge1_8)
#         # merge1_8 = L.Flatten()(merge1_8)
#         self.merge2_0 = L.concatenate([self.merge1_8,self.merge0_8], axis=-1)
#         # merge2_1 = L.Conv1D(128, 3, padding='same')(merge2_0)
#         # merge2_2 = L.advanced_activations.LeakyReLU(alpha=0.2)(merge2_0)
#         # merge2_3 =L.MaxPool2D(pool_size=(2,2),padding='same')(merge2_2)
#         # merge2_3 = L.Flatten()(merge2_0)


#         # def l2_svm(ll=1,rr=2):
#         #     def categorical(y_true, y_pred):
#         #         """
#         #         hinge with 0.5*W^2 ,SVM
#         #         """
#         #         y_true = 2. * y_true - 1 # trans [0,1] to [-1,1]，注意这个，svm类别标签是-1和1
#         #         vvvv = K.maximum(1. - y_true * y_pred, 0.) # hinge loss，参考keras自带的hinge loss
#         #         # vvv = K.square(vvvv) # 文章《Deep Learning using Linear Support Vector Machines》有进行平方
#         #         vv = K.sum(vvvv, 1, keepdims=False)  #axis=len(y_true.get_shape()) - 1
#         #         v = K.mean(vv, axis=-1)
#         #         return v
#         #     return  categorical

#         self.merge2_4=L.Dense(NUM_CLASS, 
#                            activation='linear',kernel_regularizer=regularizers.l2(0.5))(self.merge2_0)
#                     # activation='softmax')(self.merge2_0)

#         self.model = K.models.Model([TWOsingle_in,self.lidar_in],self.merge2_4)
#         if strategy=='Adam':
#             opti = K.optimizers.Adam(lr=0.0001)
#         if strategy=='SGD_005_1e-6_0':
#             opti = K.optimizers.SGD(lr=0.005, momentum=1e-6)
#         if strategy=='SGD_001_95_1e-5':
#             opti=K.optimizers.SGD(lr=0.001, momentum=0.95,decay=1e-5)
#         if strategy=='SGD_001_99_1e-3':
#             opti=K.optimizers.SGD(lr=0.001, momentum=0.99,decay=1e-3)
#         # optm=K.optimizers.SGD(lr=0.001, momentum=0.99,decay=1e-3)
#         self.model.compile(optimizer=opti,
#                         #    loss='categorical_crossentropy', metrics=['acc'])
#                              loss='categorical_squared_hinge', metrics=['acc'])
#                             # loss=l2_svm(), metrics=['acc'])