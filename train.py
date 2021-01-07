# -*- coding: utf-8 -*-
import tensorflow as tf
import keras as K
import numpy as np
from data_util import *
from models import *
from ops import *
from config import variables
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.backend.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
 
    return flops.total_float_ops  # Prints the "flops" of the model.


def train_hsi(model,Xh_train,Xh_val):
    _TFBooard='./logs/events/'
    Y_train = K.utils.np_utils.to_categorical(np.load('./file/train_label_H.npy'))
    Y_val = K.utils.np_utils.to_categorical(np.load('./file/label_valH.npy'))
    model_ckt = ModelCheckpoint(filepath=variables.HSI_2Branchnet,verbose=1, save_best_only=True)
    TFBoard = TensorBoard(_TFBooard+str(variables.dataset)+'/'+str(variables.BATCH_SIZE)+str(variables.ksize)+'/', write_graph=True, write_images=False)
    model.fit(Xh_train, Y_train, batch_size=variables.BATCH_SIZE, epochs=variables.epoch,callbacks=[model_ckt,TFBoard], validation_data=(Xh_val, Y_val))
    scores = model.evaluate(Xh_val,Y_val, batch_size=100)
    model.save(os.path.join(variables.weights_path + 'thistime_forest_H.h5'))

def train_hsi_lidar(model,Xh_train,Xl_train,Xh_val,Xl_val):
    _TFBooard='./logs/events/'

    Y_train = K.utils.np_utils.to_categorical(np.load('./file/train_label_H.npy'))

    Y_val = K.utils.np_utils.to_categorical(np.load('./file/label_valH.npy'))

    # model_ckt = ModelCheckpoint(filepath=variables.HSI_2Branchnet_LIDAR,monitor='val_acc',verbose=1, save_best_only=True,mode='max')
    model_ckt = ModelCheckpoint(filepath=variables.HSI_2Branchnet_LIDAR,verbose=1, save_best_only=True)
    # if you need tensorboard while training phase just change train fit like 
    TFBoard = TensorBoard(
        log_dir=_TFBooard+str(variables.dataset)+'/'+str(variables.BATCH_SIZE)+str(variables.ksize)+'/', write_graph=True, write_images=False)
    # model.fit([Xh_train, Xh_train[:, r, r, :, np.newaxis]], Y_train, batch_size=BATCH_SIZE, class_weight=cls_weights,
    #           epochs=args.epochs, callbacks=[model_ckt, TFBoard], validation_data=([Xh_val, Xh_val[:, r, r, :, np.newaxis]], Y_val))
    model.fit([Xh_train,Xl_train], Y_train, batch_size=variables.BATCH_SIZE, epochs=variables.epoch,
              callbacks=[model_ckt,TFBoard], validation_data=([Xh_val,Xl_val], Y_val))
    scores = model.evaluate(
        [Xh_val,Xl_val], Y_val, batch_size=100)
    print('Test score:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(os.path.join(variables.weights_path + 'thistime_forest_HL.h5'))

def test(network,strategy,mode=None):
    if network =='hsi':
        model = HSI_2Branchnet(strategy).model
        model.load_weights(variables.HSI_2Branchnet)
        # model.load_weights(os.path.join(variables.weights_path + 'thistime_forest_H.h5'))
        Xh = np.load('./file/test_data_H.npy')
        pred = model.predict([Xh])
    if network == 'hsi_lidar':
        model = HSI_2Branchnet_LIDAR(strategy,TWOsingle_weight=variables.HSI_2Branchnet).model
        model.load_weights(variables.HSI_2Branchnet_LIDAR)
        # model.load_weights(os.path.join(variables.weights_path + 'thistime_forest_HL.h5'))
        Xh = np.load('./file/test_data_H.npy')
        Xl = np.load('./file/test_data_L.npy')
        pred = model.predict([Xh,Xl])
    np.save('pred.npy',pred)
    # scio.savemat('result.mat',{'pred':data['pred']})
    Y = np.load('./file/test_label_H.npy')
    Y=np.expand_dims(Y,0)
    Y = np.reshape(Y.T, (Y.shape[1]))
    # acc,kappa = cvt_map(pred,show=False)
    # print('acc: {:.2f}%  Kappa: {:.4f}'.format(acc,kappa))
    #修改
    print('pred_Y.shape',pred.shape, Y.shape)
    print('ksize',variables.ksize)
    print('batchsize',variables.BATCH_SIZE)
    print('hsi_name',variables.mdata_name)
    print('lidar_name',variables.ldata_name)
    print('val_OA: {}%'.format(eval(pred,Y)))
    pred = np.asarray(pred)
    prediction = np.argmax(pred, axis=1)
    prediction = np.asarray(prediction, dtype=np.int8)
    print (confusion_matrix(Y, prediction))
    kappa = compute_Kappa(confusion_matrix(Y, prediction))
    print('Kappa: {:.4f}'.format(kappa))
    print (classification_report(Y, prediction,digits=4))
    # generate accuracy
    f = open(os.path.join(str(variables.dataset)+str(variables.NUM_CLASS)+str(variables.ksize)+ 'prediction.txt'), 'w+')
    n = pred.shape[0]
    for i in range(n):
        pre_label = np.argmax(pred[i], 0)
        f.write(str(pre_label) + '\n')
    f.close()
    f2 = open(os.path.join(str(variables.dataset)+str(variables.NUM_CLASS)+str(variables.ksize)+ 'confusion.txt'), 'w+')
    n = confusion_matrix(Y, prediction).shape[0]
    m = confusion_matrix(Y, prediction).shape[1]
    for a in range(n):
        for b in range(m):
            pre_label = confusion_matrix(Y, prediction)[a,b]
            f2.write(str(pre_label)+' ')
        f2.write('\n')
    f2.close()
    # f3 = open(os.path.join(str(variables.dataset)+str(variables.NUM_CLASS)+str(variables.ksize)+ 'classfication.txt'), 'w+')
    # n = classification_report(Y, prediction,digits=4).shape[0]
    # m = classification_report(Y, prediction,digits=4).shape[1]
    # for c in range(n):
    #     for d in range(m):
    #         pre_label = classification_report(Y, prediction,digits=4)[a,b]
    #         f2.write(str(pre_label)+' ')
    #     f2.write('\n')
    # f2.close()
    # print('over')

def images(network,strategy,mode=None):
    model = HSI_2Branchnet_LIDAR(strategy,TWOsingle_weight=variables.HSI_2Branchnet).model
    model.load_weights(variables.HSI_2Branchnet_LIDAR)
    r=variables.r
    mdata = read_data(variables.PATH, variables.mdata_name,'data',variables.data_type)
    ldata = read_data(variables.PATH, variables.ldata_name,'data',variables.data_type)
    label = read_data(variables.PATH,variables.label_name, 'data', variables.data_type)
    label = np.pad(label, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    mdata = np.asarray(mdata, dtype=np.float32)
    ldata = np.asarray(ldata, dtype=np.float32)
    hsi = np.pad(mdata, ((r, r), (r, r), (0, 0)), 'symmetric')
    if len(ldata.shape) == 2:
        lidar = np.pad(ldata, ((r, r), (r, r)), 'symmetric')
    if len(ldata.shape) == 3:
        lidar = np.pad(ldata, ((r, r), (r, r), (0, 0)), 'symmetric')
    lidar = samele_wise_normalization(lidar)
    hsi = samele_wise_normalization(hsi)  
    Xh,Xl=creat_images(hsi,lidar,label,variables.r)
    pred = model.predict([Xh,Xl])
    np.save('./file/pred.npy',pred)
    label = read_data(variables.PATH,variables.label_name,'data', variables.data_type)
    pred = np.argmax(pred, axis=1)
    pred = np.asarray(pred, dtype=np.int8) + 1
    print(pred)
    index = np.load(('./file/index.npy'))
    pred_map = np.zeros_like(label)
    cls = []
    for i in range(index.shape[1]):
        pred_map[index[0, i], index[1, i]] = pred[i]
        cls.append(label[index[0, i], index[1, i]])
    cls = np.asarray(cls, dtype=np.int8)
    sio.savemat('./file/propose(H+L)'+variables.dataset+'.mat', {'map': pred_map})