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
n_classes = 2
classes = [0, 1]
labels = ['BG+FG and Normal','GP3 and GP4']
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

#%
image_patches = utility.read_patches(slide_list_val, 'image20x', dataset_directory)
mask_patches = utility.read_patches(slide_list_val, 'mask20x', dataset_directory)
print('Before RBF:', len(image_patches), len(mask_patches))

print(image_patches[199])
print(mask_patches[199])

#%
val_pairs = utility.make_pair(image_patches, mask_patches)
print("\nSanity check for val_pairs")
for i in range(0,10):
    x = random.randint(0, len(val_pairs)-1)
    print(x)
    print(val_pairs[x][0])
    print(val_pairs[x][1])
    print()

#%
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

#%%
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


#%%#####################################################
from keras.models import load_model
# model1 = sm.Unet('resnet34', input_shape=(None, None, 3))

#Set compile=False as we are not loading it for training, only for prediction.
backup_model_best1 = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-unet-model-100ep-20221024-1114.hdf5'
backup_model_best2 = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-att-unet-model-100ep-20221022-2137.hdf5'
backup_model_best3 = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-att-res-unet-model-100ep-20221014-1727.hdf5'

# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-tr-weights-2c-att-res-unet-100ep-20221020-1049.hdf5'
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-tr-weights-2c-att-unet-100ep-20221025-2244.hdf5'
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-unet-model-100ep-20221009-2042.hdf5'

# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-att-res-unet-model-100ep-20220930-1707.hdf5'
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-att-unet-model-100ep-20221012-0959.hdf5'
# backup_model_best = 'E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-unet-model-100ep-20221009-2042.hdf5'

print('\nbackup_model_best')
model1 = load_model(backup_model_best1, compile=False)
model2 = load_model(backup_model_best2, compile=False)
model3 = load_model(backup_model_best3, compile=False)
# print(model1.summary())
# print('\nLoaded: ' , backup_model_best)
print('Models loaded')

model_list = [model1, model2, model3]

#%%
stop

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

#%%
image_number = random.randint(0, val_steps_RBF)
print('random image number: ', image_number)
X_val, y_val, pair_idx_val = val_generator_RBF_inf.__getitem__(image_number)
print(X_val.shape, y_val.shape)
y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')

#%
for i in range(0, len(y_val_argmax)):
    (unique, counts) = np.unique(y_val_argmax[i], return_counts=True)
    print(pair_idx_val[i], unique, counts)

#%%
sanity_check_v3(X_val, y_val_argmax, 
                note='Val RBF ',
                pair_idx=pair_idx_val,
                pairs=val_pair_RBF)

#%%
start_exe1 = datetime.now() 
print("\nPredict for the val set for only one batch")
y_pred = model1.predict(
                    # test_generator_RBF.__getitem__(image_number),
                    X_val,   
                    batch_size=batch_size, 
                    verbose=verbose)
print(y_pred.shape)
y_pred_argmax = np.argmax(y_pred, axis=3).astype('uint8')
print('y_pred_argmax.shape: ', y_pred_argmax.shape)
print('y_val_argmax.shape: ', y_val_argmax.shape)

(unique, counts) = np.unique(y_pred_argmax, return_counts=True)
print(unique, counts)

start_exe2 = datetime.now() - start_exe1
print('Execution times: ', start_exe2, '\n')

#%%
print("\nInference Plot\n")
inference_plot_v2(X_val, y_val_argmax, y_pred_argmax, 
                note='Val ',
                pair_idx=pair_idx_val,
                pairs=val_pair_RBF,
                )


#%%
import utility
utility.inference_plot_prob(X_val, y_val_argmax, y_pred, 
                note='Val ',
                pair_idx=pair_idx_val,
                pairs=val_pair_RBF,
                )

#%%
from matplotlib.colors import Normalize
from matplotlib import cm

y_pred_patch = y_pred[0]
y_pred_patch_argmax = np.argmax(y_pred_patch, axis=2)
plt.imshow(y_pred_patch_argmax, 
           # cmap='gray',
           cmap='RdYlBu_r')

print(y_pred_patch_argmax[0][0])
print(y_pred_patch[0][0])

y_pred_patch_prob = np.zeros((patch_size,patch_size))
print(y_pred_patch_prob.shape)
# y_pred_patch_prob = np.where(y_pred_patch_argmax==0,
#                               y_pred_patch[:,:,1], 
#                               y_pred_patch_prob)

y_pred_patch_prob = np.where(y_pred_patch_argmax==1,
                             y_pred_patch[:,:,1], 
                             y_pred_patch_prob)

print(np.min(y_pred_patch_prob), np.max(y_pred_patch_prob))


cmap1=matplotlib.colors.ListedColormap([
                        'white',
                        'royalblue',
                        'blue',
                        'green',
                        'lime',
                        'yellow',
                        'orange',
                        'orangered',
                        'red',
                        'darkred',
                         ])

cmap2=matplotlib.colors.ListedColormap([
                        (0,0,255), #0 
                        (0,113,255), #1
                        (0,227,255), #2
                       (0,255,170), #3
                       (0,255,57), #4
                       (57,255,0),#5
                       (170,255,0),#6
                       (255,227,0),#7
                       (255,113,0),#8
                       (255,0,0)#9
                         ])

plt.imshow(y_pred_patch_prob, 
           cmap='RdYlBu_r',
           interpolation='bicubic',
            vmin=0.0, 
            vmax=1.0
           )
plt.colorbar()


#%%
y_val_true = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-tr-weights-2c-att-res-unet-100ep-20221020-1049/y_val_true.npy')
print(y_val_true.shape)

# y_val_pred = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-att-unet-model-100ep-20221012-0959/y_val_pred.npy')
# print(y_val_pred.shape)

y_val_pred = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-4c-SL-val-unet-model-100ep-20221009-2042/y_val_pred.npy')
print(y_val_pred.shape)

y_val_pred_argmax = np.argmax(y_val_pred, axis=3).astype('uint8')
print(y_val_pred_argmax.shape)

del y_val_pred

#%%
for i in range(0,16):
    image_number = random.randint(0, len(val_pair_RBF))
    # image_number = 984
    print(i, 'random image number: ', image_number)
    utility.plot_crosscheck_4c(val_pair_RBF, 
                               y_val_true, 
                               y_val_pred_argmax, 
                               image_number,
                               'Validation')

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
start_exe1 = datetime.now() 
print("\nPredict for the test set")
test_generator_RBF_inf1 = DataGenerator(test_pair_RBF,
                                        num_classes=n_classes,
                                batch_size=batch_size, 
                                dim=(img_size,img_size,3),
                                shuffle=False,
                                inference=True,
                                )

test_steps_RBF = test_generator_RBF_inf1.__len__()
print('test_steps_RBF: ', test_steps_RBF)


#%%
image_number = random.randint(0, test_steps_RBF)
# image_number = 170
print('random batch number: ', image_number)
X_test, y_test, pair_idx_test = test_generator_RBF_inf1.__getitem__(image_number)
print(X_test.shape, y_test.shape, len(pair_idx_test))
y_test_argmax = np.argmax(y_test, axis=3).astype('uint8')

for i in range(0, len(y_test_argmax)):
    (unique, counts) = np.unique(y_test_argmax[i], return_counts=True)
    print(i, unique, counts, pair_idx_test[i], test_pair_RBF[pair_idx_test[i]])

print('random batch number: ', image_number) #170, 312, 331

#%%
y_test_pred = model1.predict(
                    # test_generator_RBF.__getitem__(image_number),
                    X_test,   
                    batch_size=batch_size, 
                    verbose=verbose)

y_test_pred_argmax = np.argmax(y_test_pred, axis=3).astype('uint8')
print('y_test_pred_argmax.shape: ', y_test_pred_argmax.shape)

(unique, counts) = np.unique(y_test_pred_argmax, return_counts=True)
print(unique, counts)

start_exe2 = datetime.now() - start_exe1
print('Execution times: ', start_exe2, '\n')

#%%
print("\nInference Plot\n")
utility.inference_plot_3(X_test[6:7], 
                         y_test_argmax[6:7], 
                         y_test_pred_argmax[6:7], 
                           note='Test ',
                           pair_idx=pair_idx_test[6:7],
                           pairs=test_pair_RBF,
                           )

#%%
import utility
utility.inference_plot_prob(X_test, y_test_argmax, pred_att_res_unet, 
                note='Test ',
                pair_idx=pair_idx_test,
                pairs=test_pair_RBF,
                )

#%%
import utility
utility.inference_plot_prob_grid(X_test,
                                 y_test_argmax, 
                                 y_test_pred, 
                                note='UNet-Test ',
                                # note='Att-UNet-Test ',
                                # note='Att-Res-UNet-Test ',
                                pair_idx=pair_idx_test,
                                pairs=test_pair_RBF,
                                )

#%%  
image_number = random.randint(0, test_steps_RBF)
# image_number = 170
print('random image number: ', image_number)
X_test, y_test, pair_idx_test = test_generator_RBF_inf1.__getitem__(image_number)
print(X_test.shape, y_test.shape, len(pair_idx_test))
y_test_argmax = np.argmax(y_test, axis=3).astype('uint8')

for i in range(0, len(y_test_argmax)):
    (unique, counts) = np.unique(y_test_argmax[i], return_counts=True)
    print(i, unique, counts, pair_idx_test[i], test_pair_RBF[pair_idx_test[i]])

print('random image number: ', image_number) #170, 312, 331


#%%
print('\nInference test pairs for Kappa evaluation')
random_ratio = 0.03

random_size = int(test_steps_RBF * random_ratio)
batch_idx = random.sample(range(1, test_steps_RBF), random_size)
print('len(batch_idx): ', len(batch_idx))
print('samples: ', batch_size * len(batch_idx))
print(batch_idx)

for batch in tqdm(batch_idx):
    print('\nRandom batch number: ', batch)
    X_test, y_test, pair_idx_test = test_generator_RBF_inf1.__getitem__(batch)
    print(X_test.shape, y_test.shape, len(pair_idx_test))
    y_test_argmax = np.argmax(y_test, axis=3).astype('uint8')
    
    print("\nEnsemble prediction") 
    preds = [model.predict(X_test,   
                batch_size=batch_size, 
                verbose=2) for model in model_list]
    
    
    preds_arr = np.array(preds)
    print(preds_arr.shape)

    #%
    # weights_arr = np.array([0.0, 0.2, 0.2])
    # wted_preds = np.tensordot(preds_arr, weights_arr, axes=0) 
    # print(wted_preds.shape)
    
    wted_preds_sum = np.sum(preds_arr, axis=0)
    print(wted_preds_sum.shape)
    
    wted_preds_sum_avg = wted_preds_sum/len(model_list)
    print(wted_preds_sum_avg.shape)
    
    wted_preds_sum_avg_argmax = np.argmax(wted_preds_sum_avg, axis=3)
    print(wted_preds_sum_avg_argmax.shape)
    
    wted_preds_sum_avg_prob = np.zeros((patch_size,patch_size))
    wted_preds_sum_avg_prob = np.where(wted_preds_sum_avg_argmax==1,
                                wted_preds_sum_avg[:,:,:,1], 
                                wted_preds_sum_avg_prob)
    print(wted_preds_sum_avg_prob.shape)

    #%
    pred_unet = preds_arr[0]
    print(pred_unet.shape)
    pred_unet_argmax = np.argmax(pred_unet, axis=3)
    pred_unet_prob = np.zeros((patch_size,patch_size))
    pred_unet_prob = np.where(pred_unet_argmax==1,
                                pred_unet[:,:,:,1], 
                                pred_unet_prob)
    print(pred_unet_prob.shape)
    
    pred_att_unet = preds_arr[1]
    print(pred_att_unet.shape)
    pred_att_unet_argmax = np.argmax(pred_att_unet, axis=3)
    pred_att_unet_prob = np.zeros((patch_size,patch_size))
    pred_att_unet_prob = np.where(pred_att_unet_argmax==1,
                                pred_att_unet[:,:,:,1], 
                                pred_att_unet_prob)
    print(pred_att_unet_prob.shape)
    
    pred_att_res_unet = preds_arr[2]
    print(pred_att_res_unet.shape)
    pred_att_res_unet_argmax = np.argmax(pred_att_res_unet, axis=3)
    pred_att_res_unet_prob = np.zeros((patch_size,patch_size))
    pred_att_res_unet_prob = np.where(pred_att_res_unet_argmax==1,
                                pred_att_res_unet[:,:,:,1], 
                                pred_att_res_unet_prob)
    print(pred_att_res_unet_prob.shape)
    
    pred_prob_list = [pred_unet_prob, pred_att_unet_prob, pred_att_res_unet_prob,
                      wted_preds_sum_avg_prob]
    #%
    import utility
    utility.inference_plot_prob_all(X_test, y_test_argmax, pred_prob_list, 
                    note='Test',
                    pair_idx=pair_idx_test,
                    pairs=test_pair_RBF,
                    SAVE=True
                    )
    
    # break

#%%
utility.inference_plot_prob_grid3(X_test,
                        y_test_argmax, 
                        wted_preds_sum_avg, 
                        note='Ensemble-Test ',
                        pair_idx=pair_idx_test,
                        pairs=test_pair_RBF,
                        )

        
#%%

utility.inference_plot_3(X_test[6:7], 
                    y_test_argmax[6:7], 
                    wted_preds_sum_avg_argmax[6:7], 
                    note='Test ',
                    pair_idx=pair_idx_test[6:7],
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
