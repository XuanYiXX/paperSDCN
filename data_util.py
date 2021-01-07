# -*- coding: utf-8 -*-
import numpy as np
import tifffile as tiff
import os
import cv2
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from scipy.cluster.vq import whiten
from config import variables


def read_data(path, file_name, data_name, data_type):
    mdata = []
    if data_type == 'tif':
        mdata = tiff.imread(os.path.join(path, file_name))
        return mdata
    if data_type == 'mat':
        mdata = sio.loadmat(os.path.join(path, file_name))
        mdata = np.array(mdata[data_name])
        return mdata
    if data_type == 'npy':
        mdata=np.load(os.path.join(path + file_name+'.npy'))
        return mdata

def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)
    
def samele_wise_normalization(data):
    """
    normalize each sample to 0-1
    Input:
        sample
    Output:
        Normalized sample
    """
    if np.max(data) == np.min(data):
        return np.ones_like(data, dtype=np.float32) * 1e-6
    else:
        return 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))

def creat_train(hsi,lidar,gth,r,validation=False):    
    per=1.0
    Xh = []
    Xl = []
    Y = []
    num_class = np.max(gth)
    for c in range(1, num_class + 1):
        idx, idy = np.where(gth == c)
        if not validation:
            idx = idx[:int(per * len(idx))]
            idy = idy[:int(per * len(idy))]
        else:
            idx = idx[int(per * len(idx)):]
            idy = idy[int(per * len(idy)):]
        np.random.seed(820)
        ID = np.random.permutation(len(idx))
        idx = idx[ID]
        idy = idy[ID]
        for i in range(len(idx)):
            tmph = hsi[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
            if len(lidar.shape)==2:
                tmpl = lidar[idx[i] - r:idx[i] + r +1, idy[i] - r:idy[i] + r + 1]
            elif len(lidar.shape)==3:
                tmpl = lidar[idx[i] - r:idx[i] + r +1, idy[i] - r:idy[i] + r + 1,:]
            tmpy = gth[idx[i], idy[i]] - 1
            Xh.append(tmph)  
            Xh.append(np.flip(tmph, axis=0))
            noise = np.random.normal(0.0, 0.01, size=tmph.shape)
            Xh.append(np.flip(tmph + noise, axis=1))
            k = np.random.randint(4)
            Xh.append(np.rot90(tmph, k=k))

            Xl.append(tmpl)
            Xl.append(np.flip(tmpl, axis=0))
            noise = np.random.normal(0.0, 0.02, size=tmpl.shape)
            Xl.append(np.flip(tmpl + noise, axis=1))
            Xl.append(np.rot90(tmpl, k=k))
            
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
    index = np.random.permutation(len(Xh))
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    Xh = Xh[index, ...]
    if len(Xl.shape)==3:
        Xl = Xl[index, ..., np.newaxis]
    elif len(Xl.shape)==4:
        Xl = Xl[index, ...]
    Y = Y[index]
    print('train hsi data shape:{},train lidar data shape:{}'.format(Xh.shape,Xl.shape))
    print('label train data shape:{}'.format(Y.shape)) 
    path_train = os.path.join('file/' + 'train_data_H.npy')
    np.save(path_train, Xh)
    path_train = os.path.join('file/' + 'train_data_L.npy')
    np.save(path_train, Xl)
    path_train = os.path.join('file/' + 'train_label_H.npy')
    np.save(path_train, Y)
    return Xh,Xl,Y
# def creat_test(hsi,lidar,gth,r,validation=False):
#     per=1.0
#     Xh = []
#     Xl = []
#     Y = []
#     num_class = np.max(gth)
#     idx, idy = np.where(gth != 0)
#     ID = np.random.permutation(len(idx))
#     for i in range(len(idx)):
#         tmph = hsi[idx[ID[i]] - r:idx[ID[i]] + r +
#                     1, idy[ID[i]] - r:idy[ID[i]] + r + 1, :]
#         if len(lidar.shape)==2:
#             tmpl = lidar[idx[ID[i]] - r:idx[ID[i]]+r + 1, idy[ID[i]] - r:idy[ID[i]] + r + 1]
#         elif len(lidar.shape)==3:
#             tmpl = lidar[idx[ID[i]] - r:idx[ID[i]]+r + 1, idy[ID[i]] - r:idy[ID[i]] + r + 1,:]
#         tmpy = gth[idx[ID[i]], idy[ID[i]]] - 1  
#     # for c in range(1, num_class + 1):
#     #     idx, idy = np.where(gth == c)
#     #     if not validation:
#     #         idx = idx[:int(per * len(idx))]
#     #         idy = idy[:int(per * len(idy))]
#     #     else:
#     #         idx = idx[int(per * len(idx)):]
#     #         idy = idy[int(per * len(idy)):]
#     #     np.random.seed(820)
#     #     ID = np.random.permutation(len(idx))
#     #     # idx = idx[ID]
#     #     # idy = idy[ID]
#     #     for i in range(len(idx)):
#     #         tmph = hsi[idx[ID[i]] - r:idx[ID[i]] + r + 1, idy[ID[i]] - r:idy[ID[i]] + r + 1, :]
#     #         if len(lidar.shape)==2:
#     #             tmpl = lidar[idx[ID[i]] - r:idx[ID[i]] + r +1, idy[ID[i]] - r:idy[ID[i]] + r + 1]
#     #         elif len(lidar.shape)==3:
#     #             tmpl = lidar[idx[ID[i]] - r:idx[ID[i]] + r +1, idy[ID[i]] - r:idy[ID[i]] + r + 1,:]
#     #         tmpy = gth[idx[ID[i]], idy[ID[i]]] - 1
#         Xh.append(tmph)
#         Xl.append(tmpl)
#         Y.append(tmpy)
#     index = np.concatenate(
#         (idx[..., np.newaxis], idy[..., np.newaxis]), axis=1)
#     np.save(os.path.join('file/','index.npy'), [idx[ID] - r, idy[ID] - r])
#     # index = np.random.permutation(len(Xh))
#     Xh = np.asarray(Xh, dtype=np.float32)
#     Xl = np.asarray(Xl, dtype=np.float32)
#     Y = np.asarray(Y, dtype=np.int8)
#     # Xh = Xh[index,...]
#     if len(Xl.shape)==3:
#         Xl = Xl[..., np.newaxis]
#     # Y = Y[index]
#     print('test hsi data shape:{},test lidar data shape:{}'.format(Xh.shape,Xl.shape))
#     print('label test data shape:{}'.format(Y.shape)) 
#     path_train = os.path.join('file/' + 'test_data_H.npy')
#     np.save(path_train, Xh)
#     path_train = os.path.join('file/' + 'test_data_L.npy')
#     np.save(path_train, Xl)
#     path_train = os.path.join('file/' + 'test_label_H.npy')
#     np.save(path_train, Y)
#     return Xh, Xl,Y
def creat_test(hsi,lidar,gth,r,validation=False):
    per=1.0
    Xh = []
    Xl = []
    Y = []
    num_class = np.max(gth)
    # for i in range(len(idx)):
    #     tmph = hsi[idx[ID[i]] - r:idx[ID[i]] + r +
    #                 1, idy[ID[i]] - r:idy[ID[i]] + r + 1, :]
    #     if len(lidar.shape)==2:
    #         tmpl = lidar[idx[ID[i]] - r:idx[ID[i]]+r + 1, idy[ID[i]] - r:idy[ID[i]] + r + 1]
    #     elif len(lidar.shape)==3:
    #         tmpl = lidar[idx[ID[i]] - r:idx[ID[i]]+r + 1, idy[ID[i]] - r:idy[ID[i]] + r + 1,:]
    #     tmpy = gth[idx[ID[i]], idy[ID[i]]] - 1  
    for c in range(1, num_class + 1):
        idx, idy = np.where(gth == c)
        if not validation:
            idx = idx[:int(per * len(idx))]
            idy = idy[:int(per * len(idy))]
        else:
            idx = idx[int(per * len(idx)):]
            idy = idy[int(per * len(idy)):]
        np.random.seed(820)
        ID = np.random.permutation(len(idx))
        idx = idx[ID]
        idy = idy[ID]
        for i in range(len(idx)):
            tmph = hsi[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
            if len(lidar.shape)==2:
                tmpl = lidar[idx[i] - r:idx[i] + r +1, idy[i] - r:idy[i] + r + 1]
            elif len(lidar.shape)==3:
                tmpl = lidar[idx[i] - r:idx[i] + r +1, idy[i] - r:idy[i] + r + 1,:]
            tmpy = gth[idx[i], idy[i]] - 1
            Xh.append(tmph)
            Xl.append(tmpl)
            Y.append(tmpy)
    index = np.random.permutation(len(Xh))
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    Xh = Xh[index,...]
    if len(Xl.shape)==3:
        Xl = Xl[index,..., np.newaxis]
    elif len(Xl.shape)==4:
        Xl = Xl[index,...]
    Y = Y[index]
    print('test hsi data shape:{},test lidar data shape:{}'.format(Xh.shape,Xl.shape))
    print('label test data shape:{}'.format(Y.shape)) 
    path_train = os.path.join('file/' + 'test_data_H.npy')
    np.save(path_train, Xh)
    path_train = os.path.join('file/' + 'test_data_L.npy')
    np.save(path_train, Xl)
    path_train = os.path.join('file/' + 'test_label_H.npy')
    np.save(path_train, Y)
    return Xh, Xl,Y
def creat_trainval(hsi,lidar,gth,r,validation=False):    
    per=1.0
    Xh = []
    Xl = []
    Y = []
    num_class = np.max(gth)
    for c in range(1, num_class + 1):
        idx, idy = np.where(gth == c)
        if not validation:
            idx = idx[:int(per * len(idx))]
            idy = idy[:int(per * len(idy))]
            # idx = idx[int(per * len(idx)):]
            # idy = idy[int(per * len(idy)):]
        else:
            idx = idx[int(per * len(idx)):]
            idy = idy[int(per * len(idy)):]
        np.random.seed(820)
        ID = np.random.permutation(len(idx))
        idx = idx[ID]
        idy = idy[ID]
        for i in range(len(idx)):
            tmph = hsi[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
            if len(lidar.shape)==2:
                tmpl = lidar[idx[i] - r:idx[i] + r +1, idy[i] - r:idy[i] + r + 1]
            elif len(lidar.shape)==3:
                tmpl = lidar[idx[i] - r:idx[i] + r +1, idy[i] - r:idy[i] + r + 1,:]
            tmpy = gth[idx[i], idy[i]] - 1
            Xh.append(tmph)
            Xh.append(np.flip(tmph, axis=0))
            noise = np.random.normal(0.0, 0.01, size=tmph.shape)
            Xh.append(np.flip(tmph + noise, axis=1))
            k = np.random.randint(4)
            Xh.append(np.rot90(tmph, k=k))

            Xl.append(tmpl)
            Xl.append(np.flip(tmpl, axis=0))
            noise = np.random.normal(0.0, 0.02, size=tmpl.shape)
            Xl.append(np.flip(tmpl + noise, axis=1))
            Xl.append(np.rot90(tmpl, k=k))
            
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
            Y.append(tmpy)
    index = np.random.permutation(len(Xh))
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    Xh = Xh[index, ...]
    if len(Xl.shape)==3:
        Xl = Xl[index, ..., np.newaxis]
    elif len(Xl.shape)==4:
        Xl = Xl[index, ...]
    Y = Y[index]
    print('val hsi data shape:{},val lidar data shape:{}'.format(Xh.shape,Xl.shape))
    print('label train data shape:{}'.format(Y.shape)) 
    path_train = os.path.join('file/' + 'data_valH.npy')
    np.save(path_train, Xh)
    path_train = os.path.join('file/' + 'data_valL.npy')
    np.save(path_train, Xl)
    path_train = os.path.join('file/' + 'label_valH.npy')
    np.save(path_train, Y)
    return Xh,Xl,Y
# def val_train(hsi,lidar,gth,r,validation=False):
#     per = 0.69
#     Xh = []
#     Xl = []
#     Y = []
#     NUM_CLASS = np.max(gth)
#     for c in range(1, NUM_CLASS + 1):
#         idx, idy = np.where(gth == c)
#         print(c,len(idx))
#         if not validation:
#             idx = idx[:int(per * len(idx))]
#             idy = idy[:int(per * len(idy))]
#         else:
#             idx = idx[int(per * len(idx)):]
#             idy = idy[int(per * len(idy)):]
#         np.random.seed(820)
#         ID = np.random.permutation(len(idx))
#         idx = idx[ID]
#         idy = idy[ID]
#         for i in range(len(idx)):
#             tmph = hsi[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
#             tmpl = lidar[idx[i] - r:idx[i] + r + 1, idy[i] - r:idy[i] + r + 1, :]
#             tmpy = gth[idx[i], idy[i]] - 1
#             Xh.append(tmph)
#             Xh.append(np.flip(tmph, axis=0))
#             noise = np.random.normal(0.0, 0.01, size=tmph.shape)
#             Xh.append(np.flip(tmph + noise, axis=1))
#             k = np.random.randint(4)
#             Xh.append(np.rot90(tmph, k=k))

#             Xl.append(tmpl)
#             Xl.append(np.flip(tmpl, axis=0))
#             noise = np.random.normal(0.0, 0.01, size=tmpl.shape)
#             Xl.append(np.flip(tmpl + noise, axis=1))
#             k = np.random.randint(4)
#             Xl.append(np.rot90(tmpl, k=k))

#             Y.append(tmpy)
#             Y.append(tmpy)
#             Y.append(tmpy)
#             Y.append(tmpy)
#     index = np.random.permutation(len(Xh))
#     Xh = np.asarray(Xh, dtype=np.float32)
#     index = np.random.permutation(len(Xl))
#     Xl = np.asarray(Xl, dtype=np.float32)
#     Y = np.asarray(Y, dtype=np.int8)
#     Xh = Xh[index, ...]
#     if len(Xl.shape) == 3:
#         Xl = Xl[index, ..., np.newaxis]
#     elif len(Xl.shape) == 4:
#         Xl = Xl[index, ...]
#     Y = Y[index]
#     if not validation:
#         print('train hsi data shape:{},train lidar data shape:{}'.format(Xh.shape, Xl.shape))
#         np.save(os.path.join('file/', 'train_Xh.npy'), Xh)
#         np.save(os.path.join('file/', 'train_Xl.npy'), Xl)
#         np.save(os.path.join('file/', 'train_Y.npy'), Y)
#         return  Xh,Xl
#     else:
#         print('validate hsi data shape:{},validate lidar data shape:{}'.format(Xh.shape, Xl.shape))
#         np.save(os.path.join('file/', 'val_Xh.npy'), Xh)
#         np.save(os.path.join('file/', 'val_Xl.npy'), Xl)
#         np.save(os.path.join('file/', 'val_Y.npy'), Y)
#         return Xh,Xl

# def creat_images(hsi,lidar,gth,r):
#     idx, idy = np.where(gth != 0)
#     ID = np.random.permutation(len(idx))
#     Xh = []
#     Xl = []
#     Y = []
#     for i in range(len(idx)):
#         tmph = hsi[idx[ID[i]] - r:idx[ID[i]] + r +
#                     1, idy[ID[i]] - r:idy[ID[i]] + r + 1, :]
#         if len(lidar.shape)==2:
#             tmpl = lidar[idx[i] - r:idx[i] + r +1, idy[i] - r:idy[i] + r + 1]
#         elif len(lidar.shape)==3:
#             tmpl = lidar[idx[i] - r:idx[i] + r +1, idy[i] - r:idy[i] + r + 1,:]
#         tmpy = gth[idx[ID[i]], idy[ID[i]]] - 1
#         Xh.append(tmph)
#         Xl.append(tmpl)
#         Y.append(tmpy)
#     index = np.concatenate(
#         (idx[..., np.newaxis], idy[..., np.newaxis]), axis=1)
#     Xh = np.asarray(Xh, dtype=np.float32)
#     Xl = np.asarray(Xl, dtype=np.float32)
#     Y = np.asarray(Y, dtype=np.int8)
#     if len(Xl.shape) == 3:
#         Xl = Xl[..., np.newaxis]
#     # Xh = Xh[index,...]
#     # if len(Xl.shape)==3:
#     #     Xl = Xl[index,..., np.newaxis]
#     # elif len(Xl.shape)==4:
#     #     Xl = Xl[index,...]
#     # Y = Y[index]
#     np.save(os.path.join('file/','index.npy'), [idx[ID] - r, idy[ID] - r])
#     return Xh,Xl
def compute_Kappa(confusion_matrix):
    """
    TODO =_= 
    """
    N = np.sum(confusion_matrix)
    N_observed = np.trace(confusion_matrix)
    Po = 1.0 * N_observed / N
    h_sum = np.sum(confusion_matrix, axis=0)
    v_sum = np.sum(confusion_matrix, axis=1)
    Pe = np.sum(np.multiply(1.0 * h_sum / N, 1.0 * v_sum / N))
    kappa = (Po - Pe) / (1.0 - Pe)
    return kappa

# def cvt_map(pred, show=False):
#     """
#     convert prediction percent to map
#     """
#     # gth = tiff.imread(os.path.join('../data/univ/', 'mask_test_all.mat'))
#     mdata = sio.loadmat(os.path.join('../data/houston/', 'houston15_mask_test1.mat'))
#     gth = np.array(mdata['mask_test'])
#     pred = np.argmax(pred, axis=1)
#     pred = np.asarray(pred, dtype=np.int8) + 1
#     print pred
#     index = np.load(os.path.join('../file/houston/', 'index.npy'))
#     pred_map = np.zeros_like(gth)
#     cls = []
#     for i in xrange(index.shape[1]):
#         pred_map[index[0, i], index[1, i]] = pred[i]
#         cls.append(gth[index[0, i], index[1, i]])
#     cls = np.asarray(cls, dtype=np.int8)
#     if show:
#         plt.axis('off')
#         plt.imshow(pred_map)
#         plt.figure()
#         plt.imshow(gth)
#         plt.show()
#     # tiff.imsave('results/Houston_lidar.tif',pred_map)
#     count = np.sum(pred == cls)
#     mx = confusion(pred - 1, cls - 1)
#     print mx
#     acc = 100.0 * count / np.sum(gth != 0)
#     kappa = compute_Kappa(mx)
#     return acc, kappa