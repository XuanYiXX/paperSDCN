# -*- coding: utf-8 -*-
import keras as K
import tensorflow as tf
import keras.layers as L
import numpy as np
import os
import time
import h5py
import argparse 
from config import variables
from data_util import *
from models import *
from ops import *
from train import*
# from config import variables
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import scipy.io as scio
from keras.utils import plot_model
from keras import backend as KB
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KB.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument('--train',
                    type=str,
                    default= None,
                    help='hsi,hsi_lidar,hsi_lidar_hsi')
parser.add_argument('--test',
                    type=str,
                    default='images',
                    help='hsi,hsi_lidar')
parser.add_argument('--modelname', type=str,
                    default='logs/weights/models.h5', help='final model save name')
parser.add_argument('--epochs',type=int,
                    default=500,help='number of epochs')
parser.add_argument('--ksize',
                    type=int,
                    default=7,
                    help='window size')
parser.add_argument('--PATH',
                    type=str,
                    default='data/forest',
                    help='load path')  
parser.add_argument('--strategy',
                    type=str,
                    default='Adam',
                    help='learning strategy')                                    
args = parser.parse_args()
variables.train=args.train

variables.PATH=args.PATH
variables.data_type='mat'

variables.ksize=args.ksize
variables.r = args.ksize //2 #数据增强图像块半径
variables.BATCH_SIZE=100 #FOREST100 MULL 128 houston 128
variables.epoch=args.epochs
# variables.label_name ='groundreferenc.mat'

# variables.dataset='houston'
# variables.NUM_CLASS =15#VNIR_LIDAR 7 italy 6 MULL 11 houston 15
# variables.mdata_name ='houston_hsi.mat'#'italy6.mat'#'houston_hsi.mat'# 'houston15.mat'#'hsi_1no23.mat' #'Hyperspectral.mat'#MUUFL 'hsi_data.mat'
# variables.ldata_name ='houston_lidar.mat'#'italy6_Lidar'#'houston_lidar.mat'#'houston_Lidar15.mat'#'lidar2_1.mat' #'FullWave_LiDAR.mat''DSM_LiDAR.mat''lidar_data1.mat'
# variables.train_label_name ='houston15_mask_train.mat'# 'italy6_mask_train.mat'#'houston_train.mat'#FOREST 'mask_train.mat' MULL'mask_train_150.mat' houston houston15_mask_train.mat
# variables.test_label_name ='houston15_mask_test.mat'#'italy6_mask_test'#'houston_test.mat'# FOREST 'mask_test.mat' MULL  'mask_test_150.mat' HOUSTON houston15_mask_test.mat
# variables.hchn=144# FOREST 286 MULL 64 houston 144 italy
# variables.lchn=1#FOREST 11 MULL 2 houston 1 italy 2

variables.dataset='forest'  
variables.NUM_CLASS =7#7 
variables.mdata_name ='hsi_1no23.mat' 
variables.ldata_name ='lidar2_1.mat'#'FullWave_LiDAR.mat'#'lidar2_1.mat'
variables.train_label_name ='mask_train.mat'
variables.test_label_name = 'mask_test.mat'
variables.hchn=286
variables.lchn=11

# variables.dataset='mull'
# variables.NUM_CLASS =11 
# variables.mdata_name ='hsi_data.mat' 
# variables.ldata_name ='lidar_data2.mat'
# variables.train_label_name ='mask_train_150.mat'
# variables.test_label_name = 'mask_test_150.mat'
# variables.hchn=64
# variables.lchn=2

# variables.dataset='italy'
# variables.NUM_CLASS =6
# variables.mdata_name ='italy6.mat'
# variables.ldata_name ='italy6_Lidar.mat'
# variables.train_label_name ='italy6_mask_train.mat'
# variables.test_label_name ='italy6_mask_test.mat'
# variables.hchn=63
# variables.lchn=2

# save weights
_weights_h = "logs/weights/forest/_hsi_weights.h5"
_weights_h_l = "logs/weights/forest_hsi_lidar_weights.h5"
variables.weights_path = os.path.join('logs/weights/' + variables.dataset+'/') 
variables.HSI_2Branchnet = os.path.join(variables.weights_path + args.strategy+'forest_H.h5')
variables.HSI_2Branchnet_LIDAR=os.path.join(variables.weights_path + args.strategy+'forest_HL.h5')
_TFBooard = 'logs/events/'

if not os.path.exists('logs/weights/'):
    os.makedirs('logs/weights/')
if not os.path.exists(_TFBooard):
    # shutil.rmtree(_TFBooard)
    os.mkdir(_TFBooard)

def main():
    r=variables.r
    mdata = read_data(variables.PATH, variables.mdata_name,'data',variables.data_type)
    ldata = read_data(variables.PATH, variables.ldata_name,'data',variables.data_type)
    label_train = read_data(variables.PATH, variables.train_label_name,'mask_train', variables.data_type)
    label_train = np.pad(label_train, ((r,r), (r, r)), 'constant', constant_values=(0, 0))
    label_test = read_data(variables.PATH,variables.test_label_name, 'mask_test', variables.data_type)
    label_test = np.pad(label_test, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    mdata = np.asarray(mdata, dtype=np.float32)
    ldata = np.asarray(ldata, dtype=np.float32)
    hsi = np.pad(mdata, ((r, r), (r, r), (0, 0)), 'symmetric')
    if len(ldata.shape) == 2:
        lidar = np.pad(ldata, ((r, r), (r, r)), 'symmetric')
    if len(ldata.shape) == 3:
        lidar = np.pad(ldata, ((r, r), (r, r), (0, 0)), 'symmetric')
    creat_test(hsi,lidar,label_test,variables.r,validation=False)
    Xh_val,Xl_val,Y_val=creat_trainval(hsi,lidar,label_train,variables.r,validation=False)
    Xh_train,Xl_train,Y_train=creat_train(hsi,lidar,label_train,variables.r,validation=False)
    #train phase
    if args.train == 'hsi':
        model =HSI_2Branchnet(args.strategy).model
        start_time = time.time()
        # plot_model(model, to_file='HSI_2Branchnet.png', show_shapes=True)
        train_hsi(model,Xh_train,Xh_val)
        duration = time.time() - start_time
        print('train time:{:.2f}s'.format(time.time() - start_time))
        # print(get_flops(model))
    if args.train == 'hsi_lidar':
        model =HSI_2Branchnet_LIDAR(args.strategy,TWOsingle_weight=variables.HSI_2Branchnet).model
        start_time = time.time()
        # plot_model(model, to_file='HSI_2Branchnet_LIDAR.png', show_shapes=True)
        train_hsi_lidar(model,Xh_train,Xl_train,Xh_val,Xl_val)
        duration = time.time() - start_time
        print('train time:{:.2f}s'.format(time.time() - start_time))
        # print(get_flops(model))
    #test phase
    if args.test == 'hsi':
        start = time.time()
        test('hsi',args.strategy)
        print('test elapsed time:{:.2f}s'.format(time.time() - start))
    if args.test == 'hsi_lidar':
        start = time.time()
        test('hsi_lidar',args.strategy)
        print('test elapsed time:{:.2f}s'.format(time.time() - start))
    # if args.test == 'images':
    #     start = time.time()
    #     images('hsi_lidar',args.strategy)
    #     print('test elapsed time:{:.2f}s'.format(time.time() - start))
if __name__ == '__main__':
    main()
    
