import tensorflow as tf
from tensorflow import keras

print(tf.test.gpu_device_name())
# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], enable=True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
              
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

print('\ntf version: ', tf.__version__)
print('keras version: ', keras.__version__, '\n')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
print('tf.test.is_built_with_cuda(): ', tf.test.is_built_with_cuda())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 20:46:01 2022

@author: kasikritdamkliang
"""
import segmentation_models as sm

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical ,Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adadelta, Nadam ,Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, platform, sys
from pathlib import Path
from tqdm import tqdm
from random import sample, choice
from PIL import Image

import matplotlib
from datetime import datetime 
import random
import glob

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

import utility

#%
default_n_classes = 5
default_classes = [0, 1, 2, 3, 4]
n_classes = 4
classes = [0, 1, 2, 3]
labels = ['BG+FG', 'Normal','GP3', 'GP4']
seed = 1337
epochs = 100
batch_size = 16
img_size = patch_size = 256
target_size = patch_size
# fold = 'fold-4'
target_size = patch_size
dropout_rate = 0.25
split_train_val = 0.35
lr = 1e-4
data_note = f'eva-AA1-4c-SL-val-unet-model-{epochs}ep'
LOGSCREEN = False
# LOGSCREEN = True
CW = False
# PROB = True
# PROB = False
# BACKBONE = 'resnet34'
# preprocess_input = sm.get_preprocessing(BACKBONE)

if platform.system() == 'Darwin':
    pass
else:
    # model_best_path = Path(os.path.abspath('G:/My Drive/PCa-Kasikrit/segmentation/'))  
    model_best_path = Path(os.path.abspath('E:/PCa-Kasikrit/42slides-data-models/'))
    dataset_directory = Path(os.path.abspath('C:/67slides-data/train55slides'))
    dataset_directory_test = Path(os.path.abspath('C:/67slides-data/test12slides'))

#%%
### Pipe standard outout to file
timestr = datetime.now().strftime("%Y%m%d-%H%M")
print("Running date: ", timestr)
log_datetime = datetime.now()
log_file_name = data_note + "-" + timestr + ".txt"
model_name = data_note + "-" + timestr


if LOGSCREEN:
    log_file = open(log_file_name, "w")

    old_stdout = sys.stdout
    sys.stdout = log_file

#%
all_exe_time1 = datetime.now()

print("\nRunning date: ", timestr)
print('Start all exe time: ', all_exe_time1)
print('tf version: ', tf.__version__)
print('keras version: ', keras.__version__)

print("\ndata_note: ", data_note)
print("n_classes: ", n_classes)
print("labels: ", labels)
print('LOGSCREEN: ', LOGSCREEN)
print('dataset_path: ', dataset_directory)
print('model_best_path: ', model_best_path)
print('model_name: ', model_name)
# print('test_dataset_path: ', test_dataset_path)

print('dropout: ', dropout_rate)
# print('split_train_val: ', split_train_val)
print('lr: ', lr)


# #%%
# train_imgs = list(  (dataset_path / "train_input" / "image").glob("*.png")) 
# train_labels = list(  (dataset_path / "train_input" / "mask").glob("*.png"))
# print(len(train_imgs), len(train_labels))
 
# assert len(train_imgs) == len(train_labels), "No of Train images and label mismatch"
# sorted(train_imgs), sorted(train_labels)

# #%%
# val_imgs = list( (dataset_path / "val_input" / "image").glob("*.png")) 
# val_labels = list( (dataset_path / "val_input"/ "mask").glob("*.png"))
# print(len(val_imgs), len(val_labels))

# assert len(val_imgs) == len(val_labels), "No of val images and label mismatch"
# sorted(val_imgs), sorted(val_labels)

# #%%
# train_pair = make_pair(train_imgs, "train_input/mask", dataset_path)
# val_pair = make_pair(val_imgs, "val_input/mask", dataset_path)
    
# print(train_pair[9])
# print(val_pair[9])

#%%
slide_list_train = [ 
            #train 32 slides
            '1329', 
            '1332', 
            '1338', 
            '1343', 
            '1347', 
            '1355', 
            '1359', 
            '1673', 
            '1677', 
            '1678', 
            '1679', 
            '1682', 
            '1683', 
            '1684', 
            '1788', 
            '1827', 
            '1830', 
            '1832', 
            '1837', 
            '1845', 
            '1848', 
            '1849', 
            '1852', 
            '1876', 
            '1879', 
            '1890', 
            '1896', 
            '1898', 
            '1901', 
            '1910', 
            '1913', 
            '1916',
            
             # Val 10 slides
            # '1323',
            # '1346',
            # '1360',
            # '1828',
            # '1840',
            # '1844',
            # '1894',
            # '1895',
            # '1924',
            # '1929',
                    
            # #test 10 slides
            # '1919',
            # '1921',
            # '1923',
            # '1924',
            # '1929',
            # '1930',
            # '1935',
            # '1941',
            # '1942',
            # '1945',
    ]
sorted(slide_list_train)
print("\nSlides:\n", slide_list_train)
print("slide len: ", len(slide_list_train))

#%%
image_patches_train = utility.read_patches(slide_list_train, 'image20x', dataset_directory)
mask_patches_train = utility.read_patches(slide_list_train, 'mask20x', dataset_directory)
print('Before RBF:', len(image_patches_train), len(image_patches_train))

print(image_patches_train[199])
print(mask_patches_train[199])


#%%
train_pairs = utility.make_pair(image_patches_train, mask_patches_train)
print("\nSanity check for train_pairs")
for i in range(0,10):
    x = random.randint(0, len(train_pairs)-1)
    print(x)
    print(train_pairs[x][0])
    print(train_pairs[x][1])
    print()

#%%
slide_list_val = [ 
            # Val 10 slides
            '1323',
            '1346',
            '1360',
            '1828',
            '1840',
            '1844',
            '1894',
            '1895',
            '1924',
            '1929',

    ]
sorted(slide_list_val)
print("\nSlides:\n", slide_list_val)
print("slide len: ", len(slide_list_val))

#%%
image_patches = utility.read_patches(slide_list_val, 'image20x', dataset_directory)
mask_patches = utility.read_patches(slide_list_val, 'mask20x', dataset_directory)
print('Before RBF:', len(image_patches), len(mask_patches))

print(image_patches[199])
print(mask_patches[199])

#%%
val_pairs = utility.make_pair(image_patches, mask_patches)
print("\nSanity check for val_pairs")
for i in range(0,10):
    x = random.randint(0, len(val_pairs)-1)
    print(x)
    print(val_pairs[x][0])
    print(val_pairs[x][1])
    print()


#%% 
# print('Sanity check for in train_pair')
# for i in range(0,20):
#     temp = choice(train_pair)
#     img = img_to_array(load_img(temp[0], target_size=(img_size,img_size)),
#                        dtype='uint8')
#     mask = cv2.imread(temp[1].as_posix(), 0)
#     plt.figure(figsize=(10,10))
#     plt.subplot(121)
#     plt.imshow(img)
#     plt.subplot(122)
#     plt.imshow(mask, cmap='gray')
#     (unique, counts) = np.unique(mask, return_counts=True)
#     xlabel = str(unique) + "\n" + str(counts)
#     plt.xlabel(xlabel)
#     plt.title('Sanity check for in train_pair')
#     plt.show()
    
#%%
# print("\nTrain: Remove all just only background patches(0) and forground patches (1)")
# train_pair_RBF = []
# for i, pair in enumerate(train_pairs):
#     # print(i, pair)
#     #mask = cv2.imread(pair[1].as_posix(), 0)
#     mask = cv2.imread(pair[1], 0)
#     count, unique = np.unique(mask, return_counts=True)
    
#     if(count[0] == 0 and len(unique)==1): # BG
#         # print("BG:", i, count, unique)
#         pass
#     elif(count[0] == 1 and len(unique)==1): #FG
#         # print("FG:", i, count, unique)
#         pass
#     elif( (count[0] == 0 and count[1] == 1) and len(unique)==2 ): #BG and FG
#         # print("BG and FG:", i, count, unique)
#         pass
#     else:
#         if(i%2000==0):
#             print("Keep: ", i, count, unique)
        
#         train_pair_RBF.append(pair)
#         # pass

# #%
# # pair = val_pair[14000][1]
# # mask = cv2.imread(pair.as_posix(), 0)
# # count, unique = np.unique(mask, return_counts=True)
# print('\ntrain_pair_RBF: ', len(train_pair_RBF)) 

#%%
print("\nVal: Remove all just only backgroud patches(0) and forground patches (1)")
val_pair_RBF = []
for i, pair in enumerate(val_pairs):
    # print(i, pair)
    # mask = cv2.imread(pair[1].as_posix(), 0) 
    mask = cv2.imread(pair[1], 0)
    count, unique = np.unique(mask, return_counts=True)
    
    if(count[0] == 0 and len(unique)==1): # BG
        # print("BG:", i, count, unique)
        pass
    elif(count[0] == 1 and len(unique)==1): #FG
        # print("FG:", i, count, unique)
        pass
    elif( (count[0] == 0 and count[1] == 1) and len(unique)==2 ): #BG and FG
        # print("BG and FG:", i, count, unique)
        pass
    else:
        if(i%1000==0):
            print("Keep: ", i, count, unique)
        val_pair_RBF.append(pair)
        # pass   
    
#%

print('\nval_pair_RBF: ', len(val_pair_RBF))

#%   
print("\nTest section\n")  
slide_list_test = [
            #test 10 slides
            '1919',
            '1921',
            '1923',
            '1924',
            '1929',
            '1930',
            '1935',
            '1941',
            '1942',
            '1945',
            ]
sorted(slide_list_test)
print("\nSlides:\n", slide_list_test)
print("slide len: ", len(slide_list_test))

#%
image_patches_test = utility.read_patches(slide_list_test, 'image20x', dataset_directory_test)
mask_patches_test = utility.read_patches(slide_list_test, 'mask20x', dataset_directory_test)
print('Before RBF:', len(image_patches_test), len(mask_patches_test))

print(image_patches_test[1999])
print(mask_patches_test[1999])

#%
test_pairs = utility.make_pair(image_patches_test, mask_patches_test)
print("\nSanity check for test_pairs")
for i in range(0,10):
    x = random.randint(0, len(test_pairs)-1)
    print(x)
    print(test_pairs[x][0])
    print(test_pairs[x][1])
    print()

#%
# test_imgs = list(  (test_dataset_path / "image").glob("*.png")) 
# test_labels = list(  (test_dataset_path / "mask").glob("*.png"))
 
# assert len(test_imgs) == len(test_labels), "No of Test images and label mismatch"
# sorted(test_imgs), sorted(test_labels) #12388
    
# test_pair = make_pair(test_imgs, "mask", test_dataset_path)
# print('len(test_pair): ', len(test_pair))


#%
print("\nTest: Remove all just only backgroud patches(0) and forground patches (1)")
test_pair_RBF = []
for i, pair in enumerate(test_pairs):
    # print(i, pair)
    # mask = cv2.imread(pair[1].as_posix(), 0)
    mask = cv2.imread(pair[1], 0)
    count, unique = np.unique(mask, return_counts=True)
    
    if(count[0] == 0 and len(unique)==1): # BG
        # print("BG:", idx,count, unique)
        pass
    elif(count[0] == 1 and len(unique)==1): #FG
        # print("FG:", idx,count, unique)
        pass
    elif( (count[0] == 0 and count[1] == 1) and len(unique)==2 ): #BG and FG
        # print("BG and FG:", i, count, unique)
        pass
    else:
        if(i%1000==0):
            print("Keep: ", i, count, unique)
        test_pair_RBF.append(pair)

print('test_pair_RBF: ', len(test_pair_RBF)) # 


        
#%%
# print('\nCreate train_generator_RBF')
# train_generator_RBF = DataGenerator(train_pair_RBF,
#                                 batch_size=batch_size, 
#                                 dim=(patch_size, patch_size, 3),
#                                 # dim=(target_size, target_size, 3),
#                                 shuffle=True,
#                                 augmentation=get_training_augmentation(),
#                                 # preprocessing=get_preprocessing(preprocess_input),
#                                 )
# train_steps_RBF = train_generator_RBF.__len__()
# print('train_steps_RBF: ', train_steps_RBF)

# #
# image_number = random.randint(0, train_steps_RBF)
# print('random image number: ', image_number)
# X2, y2 = train_generator_RBF.__getitem__(image_number)
# print(X2.shape, y2.shape)
# y_argmax2 = np.argmax(y2, axis=3).astype('uint8')

# #
# sanity_check(X2, y_argmax2, note='Train RBF ', batch_size=batch_size//2)
    

#%%
# print('\nCreate val_generator_RBF')
# val_generator_RBF_train = DataGenerator(val_pair_RBF,
#                                 batch_size=batch_size, 
#                                 dim=(patch_size, patch_size, 3),
#                                 # dim=(target_size,target_size,3),
#                                 shuffle=True,
#                                 augmentation=get_training_augmentation(),
#                                 # preprocessing=get_preprocessing(preprocess_input),
#                                 )
# val_steps_RBF = val_generator_RBF.__len__()
# print('val_steps_RBF: ', val_steps_RBF)

# #%
# image_number = random.randint(0, val_steps_RBF)
# print('random image number: ', image_number)
# X1, y1 = val_generator_RBF.__getitem__(image_number)
# print(X1.shape, y1.shape)
# y_argmax1 = np.argmax(y1, axis=3).astype('uint8')

# sanity_check(X1, y_argmax1, note='Val RBF ', batch_size=batch_size//2)



#%%
if CW == True:
    print("\nGet all labels preparing for computing class weights")
    train_labels_RBF = get_all_labels(train_generator_RBF)
    print('train_labels_RBF.shape: ', train_labels_RBF.shape)
    train_labels_RBF_flatten = train_labels_RBF.flatten()
    print('train_labels_RBF_flatten.shape: ', train_labels_RBF_flatten.shape)
    print('Calculating class weights...')
    class_weights = cal_class_weight(train_labels_RBF_flatten, classes)
    print('class_weights: ', class_weights)
else:  
#%
#RBF-both
    class_weights = [0.500981666256899, 1.0815061468805043, 
                     2.0159053357520116, 1.7145968330578576]

#binary class
# class_weights = [0.6847553952019375, 1.8531404575587638]
    
print('\nclass_weights = ', class_weights)


#%%
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime 
import cv2
from PIL import Image
from tensorflow.keras import backend, optimizers

#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.
# from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss
# from models import (Attention_ResUNet, UNet, Attention_UNet, 
#                     dice_coef, dice_coef_loss, jacard_coef,
#                     jacard_coef_loss, iou_coef1, dice_coef1)


# dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
# focal_loss = sm.losses.CategoricalFocalLoss()
# # focal_loss = sm.losses.BinaryFocalLoss()
# total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, 
# above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss

# sm_total_loss = sm.losses.categorical_focal_dice_loss  

# sm_metrics = [
#             sm.metrics.IOUScore(
#             threshold=0.5, 
#             class_weights=class_weights
#             ), 
#             sm.metrics.FScore(
#                  threshold=0.5, 
#               class_weights=class_weights
#              )
#            ]


# metrics = ['accuracy', sm_metrics, jacard_coef, dice_coef, iou_coef, dice_coef1] 

# print('\nDefine and config model')
# model = UNet(
#     input_shape=(patch_size, patch_size, 3),
#     # input_shape=(None, None, 3), #TypeError: unsupported operand type(s) for //: 'NoneType' and 'NoneType'
#     NUM_CLASSES=n_classes,
#     #dropout_rate=0.2,
#     dropout_rate=dropout_rate,
#     activation='softmax',
#     )

# print('Model: ', model.name)

#%%
# print("\nTransfer trained weights from: ", backup_model_best_tr)
# unet_model = load_model(backup_model_best_tr, compile=False)
# print('\nTransfer completed.')

# #%%
# model.compile(optimizer=Adam(lr = lr), 
#         # loss=BinaryFocalLoss(gamma=2), # run properly
#         # loss=SparseCategoricalFocalLoss(gamma=4), 
#         loss=total_loss, # run properly
#         # loss = sm_total_loss,
#         # metrics=['accuracy', jacard_coef],
#         metrics = metrics
#         )
# print('\nloss = (total_loss =  dice_loss + (1 * focal_loss)')
# print('\nmetrics = ', metrics)

# #%%
# print(model.summary())

                                           
# #%%
# backup_model_best = os.path.join(model_best_path,
#                                    f'backup_model_{model_name}.hdf5')

# print('\nbackup_model_best: ', backup_model_best)
# mcp2 = ModelCheckpoint(backup_model_best, save_best_only=True) 

# reLR = ReduceLROnPlateau(monitor='val_jacard_coef',
#                            factor=0.8,
#                            patience=5,
#                            verbose=1,
#                            mode='auto',
#                            #min_lr = 0.00001,#1e-5
#                            min_lr = lr/100,
#                          )

# early_stop = EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
#                           min_delta=lr/100)

# #%%
# import tensorflow_addons as tfa
# tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)

# from livelossplot import PlotLossesKeras

# # print('steps_per_epoch: ', train_steps_RBF)
# # print('val_steps_per_epoch: ', val_steps_RBF)

# #%
# if LOGSCREEN:
#     verbose=2
# else:
#     verbose=1

# print('\nverbose: ', verbose)


# #%%
# print("\n\nPerform training...");
# print(data_note)
# t1 = datetime.now()
# with tf.device('/device:GPU:0'):
#     model_history = model.fit(
#             train_generator_RBF, 
#             # steps_per_epoch=train_steps_RBF,
#             validation_data=val_generator_RBF,   
#             # validation_steps=val_steps_RBF,
#             epochs=epochs,
#             verbose=verbose,
#             callbacks=[
#                         reLR,
#                         mcp2,
#                         # early_stop, 
#                         tqdm_callback, 
#                         PlotLossesKeras(),
#                         ],
#             )
# t2 = datetime.now() - t1
# print("\nTraining time: ", t2)

# #%%
# # convert the history.history dict to a pandas DataFrame and save as csv for
# # future plotting
# import pandas as pd    
# # unet_history_df = pd.DataFrame(unet_history.history) 
# # att_unet_history_df = pd.DataFrame(att_unet_history.history) 
# model_history_df = pd.DataFrame(model_history.history) 

# # with open('unet_history_df.csv', mode='w') as f:
# #     unet_history_df.to_csv(f)
    
# # with open('att_unet_history_df.csv', mode='w') as f:
# #     att_unet_history_df.to_csv(f)
# history_file = f'{model_name}_history_df.csv'
# with open(history_file, mode='w') as f:
#     model_history_df.to_csv(f)  
# print("\nSaved: ", history_file)
    
# #%%
# #Check history plots, one model at a time
# # history1 = unet_history
# # history1 = att_unet_history
# history1 = model_history

# ##########################################################
# #plot the training and validation accuracy and loss at each epoch
# loss = history1.history['loss']
# val_loss = history1.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure(figsize=(8, 6), dpi=600)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss ' + data_note)
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# #%%
# acc = history1.history['jacard_coef']
# val_acc = history1.history['val_jacard_coef']
# plt.figure(figsize=(8, 6), dpi=600)
# plt.plot(epochs, acc, 'y', label='Training Jacard')
# plt.plot(epochs, val_acc, 'r', label='Validation Jacard')
# plt.title('Training and validation Jacard ' + data_note)
# plt.xlabel('Epochs')
# plt.ylabel('Jacard')
# plt.legend()
# plt.show()

# #%%
# acc = history1.history['f1-score']
# val_acc = history1.history['val_f1-score']
# plt.figure(figsize=(8, 6), dpi=600)
# plt.plot(epochs, acc, 'y', label='Training F1-Score')
# plt.plot(epochs, val_acc, 'r', label='Validation F1-Score')
# plt.title('Training and validation F1-Score ' + data_note)
# plt.xlabel('Epochs')
# plt.ylabel('F1-Score')
# plt.legend()
# plt.show()

# #%%
# acc = history1.history['iou_score']
# val_acc = history1.history['val_iou_score']
# plt.figure(figsize=(8, 6), dpi=600)
# plt.plot(epochs, acc, 'y', label='Training IOUScore')
# plt.plot(epochs, val_acc, 'r', label='Validation IOUScore')
# plt.title('Training and validation F1-Score ' + data_note)
# plt.xlabel('Epochs')
# plt.ylabel('IOUScore')
# plt.legend()
# plt.show()



#%%#####################################################
from keras.models import load_model
# model1 = sm.Unet('resnet34', input_shape=(None, None, 3))

#Set compile=False as we are not loading it for training, only for prediction.
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-unet-model-100ep-20221024-1114.hdf5'
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-att-unet-model-100ep-20221022-2137.hdf5'
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-att-res-unet-model-100ep-20221014-1727.hdf5'

# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-tr-weights-2c-att-res-unet-100ep-20221020-1049.hdf5'
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-tr-weights-2c-att-unet-100ep-20221025-2244.hdf5'
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-unet-model-100ep-20221009-2042.hdf5'

backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-att-res-unet-model-100ep-20220930-1707.hdf5'
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-att-unet-model-100ep-20221012-0959.hdf5'
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-unet-model-100ep-20221009-2042.hdf5'

print('\nbackup_model_best: ', backup_model_best)
model1 = load_model(backup_model_best, compile=False)
print(model1.summary())
print('\nLoaded: ' , backup_model_best)

#%%
from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss
from models import (Attention_ResUNet, UNet, Attention_UNet, 
                    dice_coef, dice_coef_loss, jacard_coef,
                    jacard_coef_loss, iou_coef1, dice_coef1)

dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
focal_loss = sm.losses.CategoricalFocalLoss()
# focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

sm_metrics = [
            sm.metrics.IOUScore(
            # threshold=0.5, 
            class_weights=class_weights,
            class_indexes = classes,
            ), 
            sm.metrics.FScore(
            # threshold=0.5, 
              class_weights=class_weights,
              class_indexes = classes,
             )
           ]


metrics = ['accuracy', sm_metrics, 
           #jacard_coef, dice_coef, 
           iou_coef1, dice_coef1]  

model1.compile(Adam(lr = lr), 
               total_loss, 
               metrics= metrics)

#%%
print('\nCreate val_generator_RBF')
val_generator_RBF_eva = DataGenerator(val_pair_RBF,
                                num_classes=n_classes,
                                batch_size=batch_size, 
                                dim=(patch_size, patch_size, 3),                       
                                shuffle=False,
                                )
val_steps_RBF = val_generator_RBF_eva.__len__()
print('val_steps_RBF: ', val_steps_RBF)

#%%
image_number = random.randint(0, val_steps_RBF)
print('random image number: ', image_number)
X1, y1 = val_generator_RBF_eva.__getitem__(image_number)
print(X1.shape, y1.shape)
y_argmax1 = np.argmax(y1, axis=3).astype('uint8')

for i in range(0, len(y_argmax1)):
    (unique, counts) = np.unique(y_argmax1[i], return_counts=True)
    print(unique, counts)
    
sanity_check(X1, y_argmax1, note='Val RBF ', batch_size=batch_size//2)

#%%
start_exe1 = datetime.now() 
print("\nEvaluate model for the val set")
with tf.device('/device:GPU:0'):
    verbose=1
    scores = model1.evaluate(val_generator_RBF_eva, 
                              batch_size=batch_size, 
                              verbose=verbose)
    
for metric, value in zip(model1.metrics_names, scores):
    print("mean {}: {:.4}".format(metric, value))


start_exe2 = datetime.now() - start_exe1
print('Execution times: ', start_exe2, '\n')    


#%%   
# print("\nPredict for the val set")
# y_pred1_val = model1.predict(val_generator_RBF, 
#                          batch_size=batch_size, 
#                          verbose=verbose)

# y_pred1_argmax_val = np.argmax(y_pred1_val, axis=3).astype('uint8')

# # print(unique_counts(y_pred1_argmax))
# print(y_pred1_argmax_val.shape)  
    
# #%
# print("\nGet all batches of val fold")
# X_test_list = []
# y_test_list = []

# # for i in range(10):
# for i in range(val_generator_RBF.__len__()):
#     X_test, y_test = val_generator_RBF.__getitem__(i)
#     print(i, X_test.shape, y_test.shape)
#     y_test_list.append(y_test)
#     X_test_list.append(X_test)
     
# X_test1_val = np.concatenate(X_test_list, axis=0)
# y_test1_val = np.concatenate(y_test_list, axis=0).astype('uint8')
# y_test1_argmax_val = np.argmax(y_test1_val, axis=3).astype('uint8')

# print('\nSanity check')
# sanity_check(X_test1_val[:32], y_test1_argmax_val[:32], note='Val ', batch_size=2)
# sanity_check(X_test1_val[50:], y_test1_argmax_val[50:], note='Val ', batch_size=2)
# sanity_check(X_tX_test1_valest1[500:], y_test1_argmax_val[500:], note='Val ', batch_size=2)


#%%
import utility
print('\nCreate val_generator_RBF_inf')
val_generator_RBF_inf = DataGenerator(val_pair_RBF,
                                num_classes=n_classes,
                                batch_size=batch_size, 
                                dim=(patch_size, patch_size, 3),
                                shuffle=False,
                                inference=True,
                                )
val_steps_RBF = val_generator_RBF_inf.__len__()
print('val_steps_RBF: ', val_steps_RBF)

image_number = random.randint(0, val_steps_RBF)
print('random image number: ', image_number)
X_val, y_val, pair_idx_val = val_generator_RBF_inf.__getitem__(image_number)
print(X_val.shape, y_val.shape)
y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')

for i in range(0, len(y_val_argmax)):
    (unique, counts) = np.unique(y_val_argmax[i], return_counts=True)
    print(unique, counts)

#%
# sanity_check_v3(X_val, y_val_argmax, 
#                 note='Val RBF ',
#                 pair_idx=pair_idx_val,
#                 pairs=val_pair_RBF)

#%%
# start_exe1 = datetime.now() 
# print("\nPredict for the val set for only one batch")
# y_pred = model1.predict(
#                     # test_generator_RBF.__getitem__(image_number),
#                     X_val,   
#                     batch_size=batch_size, 
#                     verbose=verbose)

# y_pred_argmax = np.argmax(y_pred, axis=3).astype('uint8')
# print('y_pred_argmax.shape: ', y_pred_argmax.shape)
# print('y_val_argmax.shape: ', y_val_argmax.shape)

# (unique, counts) = np.unique(y_pred_argmax, return_counts=True)
# print(unique, counts)

# start_exe2 = datetime.now() - start_exe1
# print('Execution times: ', data_note, start_exe2, '\n')

# print("\nInference Plot\n")
# inference_plot_v2(X_val, y_val_argmax, y_pred_argmax, 
#                note='Val ',
#                pair_idx=pair_idx_val,
#                pairs=val_pair_RBF,
#                )
#%
# y_pred_val = model1.predict(val_generator_RBF_eva, verbose=1)

#%%
print("\nEvaluate the whole val set")
start_exe1 = datetime.now()
# sample_size = 10
# X_val_list= []
y_val_list = []
y_val_pred_list = []
pair_idx_val_list = []
   
# for batch in range(1, val_steps_RBF+1):
print('\nPredicting...')
for batch in tqdm(range(val_generator_RBF_inf.__len__())): 
    X_val, y_val, pair_idx_val = val_generator_RBF_inf.__getitem__(batch)
    # print('\n', X_val.shape, y_val.shape, len(pair_idx_val))
    
    y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')
    # print('y_val_argmax.shape: ', y_val_argmax.shape)
    verbose=2   
    y_val_pred = model1.predict(
                    X_val,   
                    batch_size=batch_size, 
                    verbose=verbose)

    # y_pred_argmax = np.argmax(y_pred, axis=3).astype('uint8')
    # print('\ny_pred_argmax.shape: ', y_pred_argmax.shape)
    

    # X_val_list.append(X_val)
    y_val_list.append(y_val_argmax)
    y_val_pred_list.append(y_val_pred)
    pair_idx_val_list.append(pair_idx_val)

#%
# X_val = np.concatenate(X_val_list, axis=0)
y_val_true = np.concatenate(y_val_list, axis=0)
y_val_pred = np.concatenate(y_val_pred_list, axis=0)
pair_idx_val = np.concatenate(pair_idx_val_list, axis=0)
print('y_val_true.shape, y_val_pred.shape, pair_idx_val.shape')
print(y_val_true.shape, y_val_pred.shape, pair_idx_val.shape)
del y_val_list
del y_val_pred_list

y_val_pred_argmax = np.argmax(y_val_pred, axis=3) #int64
print(y_val_pred_argmax.shape)

# np.save("y_val_true.npy", y_val_true)
#np.save("y_val_pred.npy", y_val_pred)
del y_val_pred

#%%
# print("\nVal: Calculate Jacard and Dice manually")
# with tf.device('/device:GPU:0'):
#     jacard_coef_val = jacard_coef(np.array(y_val_true, dtype='float32'), 
#                                   np.array(y_val_pred_argmax, dtype='float32')
#                                 )
                                  
#     dice_coef_val  = dice_coef(np.array(y_val_true, dtype='float32'), 
#                                 np.array(y_val_pred_argmax, dtype='float32')
#                                 )

# print('jacard_coef:', jacard_coef_val)
# print('dice_coef:', dice_coef_val)

# start_exe2 = datetime.now() - start_exe1
# print('Execution times: ', start_exe2, '\n')

#%%
print("\nVal: Calculate iou_coef1 and dice_coef1 manually")
with tf.device('/device:GPU:0'):
    val_iou_coef1 = iou_coef1(np.array(y_val_true, dtype='float32'), 
                                  np.array(y_val_pred_argmax, dtype='float32')
                                )
                                  
    val_dice_coef1  = dice_coef1(np.array(y_val_true, dtype='float32'), 
                                np.array(y_val_pred_argmax, dtype='float32')
                                )

print('val_iou_coef1:', val_iou_coef1)
print('val_dice_coef1:', val_dice_coef1)


print(np.mean(val_iou_coef1))
print(np.mean(val_dice_coef1))

# # dice_coef2_val = dice_coef2(np.array(y_val_true, dtype='float32'),
# #                        np.array(y_val_pred_argmax, dtype='float32')
# #                        )
# # print(dice_coef2_val)


    
#%%
# y_val_true = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-tr-weights-2c-att-res-unet-100ep-20221020-1049/y_val_true.npy')
# print(y_val_true.shape)

# y_val_pred = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-att-unet-model-100ep-20221012-0959/y_val_pred.npy')
# print(y_val_pred.shape)

# y_val_pred = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-unet-model-100ep-20221009-2042/y_val_pred.npy')
# print(y_val_pred.shape)

# y_val_pred_argmax = np.argmax(y_val_pred, axis=3).astype('uint8')
# print(y_val_pred_argmax.shape)

# del y_val_pred

#%%
# for i in range(0,16):
#     image_number = random.randint(0, len(val_pair_RBF))
#     # image_number = 984
#     print(i, 'random image number: ', image_number)
#     utility.plot_crosscheck_4c(val_pair_RBF, 
#                                y_val_true, 
#                                y_val_pred_argmax, 
#                                image_number,
#                                'Validation')
    
#%%
# y_val_true = np.array(y_val_true, dtype='uint8')
# y_val_pred = np.array(y_val_pred, dtype='uint8')

#%%
# print("\nCompute IoU for Val set using sklearn.metrics")
# print('sample_size: ', sample_size)
# with tf.device('/device:GPU:0'):
#     iou_score = compute_iou(y_val_pred1, y_val1, classes)
#     print('Mean IoU: ', iou_score)
# t4 = datetime.now() - t3
# print("Exe time: ", t4)


 
#%%,
print("\nCompute MeanIoU for Val ratio 0.5")
for i in [1,2,3]:
    random_ratio = 0.5
    random_size = int(len(y_val_true) * random_ratio)
    batch_idx = random.sample(range(1, len(y_val_true)), random_size)
    print('len(batch_idx): ', len(batch_idx))
    values4c = utility.compute_MeanIoU_4c(y_val_true[batch_idx], y_val_pred_argmax[batch_idx])
    print('len(batch_idx): ', len(batch_idx))
    
#     To calculate IoU for each class...AA2 att-res-unet
# [[1.27466704e+08 9.59797400e+06 1.61193400e+06 2.84281600e+06]
#  [8.44485900e+06 4.49259660e+07 1.76130300e+06 1.72506400e+06]
#  [8.07588800e+06 1.40348900e+06 1.76473130e+07 3.95893600e+06]
#  [8.16818700e+06 5.65840300e+06 4.82193600e+06 2.34704120e+07]]
 
# To calculate IoU for each class... AA2 unet
# [[1.25273378e+08 1.02185430e+07 3.08389700e+06 1.35714400e+06]
#  [1.10908100e+07 4.07199110e+07 3.41362900e+06 1.69537000e+06]
#  [8.11374200e+06 2.09965500e+06 2.01342470e+07 1.61158600e+06]
#  [1.08107090e+07 5.69175100e+06 7.83296500e+06 1.84338470e+07]]
   
#%%
values4c = utility.compute_MeanIoU_4c(y_val_true, y_val_pred_argmax)

#%%
# print('\nRelease all the gpu memory')
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()

#%%
start_exe1 = datetime.now() 
print("\nEvaluate model for the test set by model.evaluate()")
test_generator_RBF_eva = DataGenerator(test_pair_RBF,
                                num_classes=n_classes,
                                batch_size=batch_size, 
                                dim=(img_size,img_size,3),
                                shuffle=False,
                                # inference=True,
                                )
for i in tqdm(range(test_generator_RBF_eva.__len__())):
    print(i, end=' ')

#%%
image_number = random.randint(0, test_steps_RBF)
# image_number = 500
print('random image number: ', image_number)
X_test, y_test, = test_generator_RBF_eva.__getitem__(image_number)
print(X_test.shape, y_test.shape)
y_test_argmax = np.argmax(y_test, axis=3).astype('uint8')

for i in range(0, len(y_argmax1)):
    (unique, counts) = np.unique(y_test_argmax[i], return_counts=True)
    print(unique, counts)
    
#%%
sanity_check(X_test, y_val_argmax, 
             note='Test ', 
             batch_size=batch_size)
    
#%%
with tf.device('/device:GPU:0'):
    verbose=1
    scores = model1.evaluate(test_generator_RBF_eva, 
                             batch_size=batch_size, 
                             verbose=verbose)
    
for metric, value in zip(model1.metrics_names, scores):
    print("mean {}: {:.4}".format(metric, value))

    
start_exe2 = datetime.now() - start_exe1
print('Execution times: ', start_exe2, '\n')

     
#%%
print("\nCreate test generator for inference")
test_generator_RBF_inf = DataGenerator(test_pair_RBF,
                                num_classes=n_classes,
                                batch_size=batch_size, 
                                dim=(img_size,img_size,3),
                                shuffle=False,
                                inference=True,
                                )

test_steps_RBF = test_generator_RBF_inf.__len__()
print('test_steps_RBF: ', test_steps_RBF)

#%%
image_number = random.randint(0, test_steps_RBF)
print('random image number: ', image_number)
X_inf, y_inf, pair_idx_inf = test_generator_RBF_inf.__getitem__(image_number)
print(X_inf.shape, y_inf.shape)
y_inf_argmax = np.argmax(y_inf, axis=3).astype('uint8')

for i in range(0, len(y_inf_argmax)):
    (unique, counts) = np.unique(y_inf_argmax[i], return_counts=True)
    print(unique, counts)

#%%
utility.sanity_check_with_patch_id(X_inf, y_inf_argmax, note='Test ', 
                pair_idx=pair_idx_inf, 
                pairs=test_pair_RBF)


#%%
print("\nEvaluate the whole test set")
t3 = datetime.now()
# sample_size = 10
# X_test_list= []
# y_test_list = []
y_test_pred_list = []
pair_idx_test_list = []
   
for batch in tqdm(range(test_generator_RBF_inf.__len__())):
    # print('\nPredicting batch: ', batch)
    X_test, y_test, pair_idx_test = test_generator_RBF_inf.__getitem__(batch)
      
    # print(X_test.shape, y_test.shape, len(pair_idx_test))
    # y_test_argmax = np.argmax(y_test, axis=3).astype('uint8')
    verbose=2
    with tf.device('/device:GPU:0'):
        y_test_pred = model1.predict(
                        # test_generator_RBF.__getitem__(image_number),
                        X_test,   
                        batch_size=batch_size, 
                        verbose=verbose)

    # y_pred_argmax = np.argmax(y_pred, axis=3).astype('uint8')
    # print('\ny_pred_argmax.shape: ', y_pred_argmax.shape)
    # print('y_test_argmax.shape: ', y_test_argmax.shape)
   
    # X_test_list.append(X_test)
    # y_test_list.append(y_test_argmax)
    y_test_pred_list.append(y_test_pred)
    pair_idx_test_list.append(pair_idx_test)
      
# X_val1 = np.concatenate(X_val_list, axis=0)
# y_test_true = np.concatenate(y_test_list, axis=0)
y_test_pred = np.concatenate(y_test_pred_list, axis=0)
pair_idx_test = np.concatenate(pair_idx_test_list, axis=0)
# print('y_test_true.shape, y_test_pred.shape, pair_idx_test.shape')
# print(y_test_true.shape, y_test_pred.shape, pair_idx_test.shape)
del y_test_pred_list

y_test_pred_argmax = np.argmax(y_test_pred, axis=3)
print(y_test_pred_argmax.shape)

t4 = datetime.now() - t3
print('Execution times: ', t4, '\n')

# np.save("y_test_true.npy", y_test_true)
np.save("y_test_pred.npy", y_test_pred)
del y_test_pred


#%%
y_test_true = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-tr-weights-2c-att-res-unet-100ep-20221020-1049/y_test_true.npy')
print(y_test_true.shape)
y_test_pred = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-unet-model-100ep-20221009-2042/y_test_pred.npy')
y_test_pred_argmax = np.argmax(y_test_pred, axis=3).astype('uint8')
print(y_test_pred_argmax.shape)
del y_test_pred

#%%
for i in range(0,16):
    image_number = random.randint(0, len(test_pair_RBF))
    # image_number = 984
    print(i, 'random image number: ', image_number)
    utility.plot_crosscheck_4c(test_pair_RBF, 
                       y_test_true, 
                       y_test_pred_argmax,
                       image_number,
                       'Test')

#%%
for i in [1,2,3]:
    print(i)   
    random_ratio = 0.5 # 0.6 =>  Resource exhausted: OOM when allocating tensor with shape[325844992] 
    random_size = int(len(y_test_true) * random_ratio)
    batch_idx = random.sample( range(1, len(y_test_true)), random_size)
    print('len(batch_idx): ', len(batch_idx))
    values_test_4c = compute_MeanIoU_4c(y_test_true[batch_idx], y_test_pred_argmax[batch_idx])
    print('\n')
    

#%%
start_exe1 = datetime.now() 
print("\nPredict for the test set")
test_generator_RBF_inf1 = DataGenerator(test_pair_RBF,
                                batch_size=batch_size, 
                                dim=(img_size,img_size,3),
                                shuffle=True,
                                inference=True,
                                )

test_steps_RBF = test_generator_RBF_inf1.__len__()
print('test_steps_RBF: ', test_steps_RBF)

#%%
image_number = random.randint(0, test_steps_RBF)
print('random image number: ', image_number)
X1, y1, pair_idx = test_generator_RBF_inf1.__getitem__(image_number)
print(X1.shape, y1.shape)
y_argmax1 = np.argmax(y1, axis=3).astype('uint8')

for i in range(0, len(y_argmax1)):
    (unique, counts) = np.unique(y_argmax1[i], return_counts=True)
    print(unique, counts)

y_pred1 = model1.predict(
                    # test_generator_RBF.__getitem__(image_number),
                    X1,   
                    batch_size=batch_size, 
                    verbose=verbose)

y_pred1_argmax = np.argmax(y_pred1, axis=3).astype('uint8')
print('y_pred1_argmax.shape: ', y_pred1_argmax.shape)

(unique, counts) = np.unique(y_pred1_argmax, return_counts=True)
print(unique, counts)

start_exe2 = datetime.now() - start_exe1
print('Execution times: ', data_note, start_exe2, '\n')

print("\nInference Plot\n")
inference_plot_v2(X1, y_argmax1, y_pred1_argmax, 
               note='Test ',
               pair_idx=pair_idx,
               pairs=test_pair_RBF,
               )

#%%
all_exe_time2 = datetime.now() - all_exe_time1
print('All execution times: ', all_exe_time2, '\n')

#%%
if LOGSCREEN:
    sys.stdout = old_stdout
    log_file.close()   



#%%
