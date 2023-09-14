import tensorflow as tf
import segmentation_models as sm
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
import tensorflow as tf
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

#%
default_n_classes = 5
default_classes = [0, 1, 2, 3, 4]
n_classes = 2
classes = [0, 1]
labels = ['BG+FG+Normal','GP3+GP4']
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
data_note = f'train-2c-SL-val-unet-model-{epochs}ep'
LOGSCREEN = False
# LOGSCREEN = True
CW = False
# PROB = True
# PROB = False
# BACKBONE = 'resnet34'
# preprocess_input = sm.get_preprocessing(BACKBONE)

dataset_path = ""
model_best_path = ""
if platform.system() == 'Darwin':
    dataset_directory = Path('/Volumes/MyPassport/PCa-Kasikrit/segmentation/slides/')
    slide_no_pos = 5
    file_name_pos = 7
else:
    # model_best_path = Path(os.path.abspath('G:/My Drive/PCa-Kasikrit/segmentation/'))  
    model_best_path = Path(os.path.abspath('E:/PCa-Kasikrit/42slides-data-models/'))
    dataset_directory = Path(os.path.abspath('C:/67slides-data/train55slides'))
    dataset_directory_test = Path(os.path.abspath('C:/67slides-data/test12slides'))
    slide_no_pos = 3
    file_name_pos = 5

#%%
def make_pair1(img,label,dataset):
    pairs = []
    for im in img:
        pairs.append( (im , dataset/ label / (im.stem + ".png")) )
    
    return pairs

def make_pair(imgs,labels):
    pairs = []
    for img, mask in zip(imgs, labels):
        pairs.append( (img, mask) )
    
    return pairs

def read_patches(slide_list, patch_type):
    patches = []
    for slide in slide_list:
        image_directory = os.path.join(
        dataset_directory,
        slide,
        patch_type,
        '*.png'
        )
        images = list(glob.glob(image_directory, recursive=True ))
        images.sort()
        # print(images[:5])
        print(images[0].split(os.path.sep)[slide_no_pos])
        print('\n')
        for patch in images:
            patches.append(patch)
        
    return patches

def read_patches_test(slide_list, patch_type):
    patches = []
    for slide in slide_list:
        image_directory = os.path.join(
        dataset_directory_test,
        slide,
        patch_type,
        '*.png'
        )
        images = list(glob.glob(image_directory, recursive=True ))
        images.sort()
        # print(images[:5])
        print(images[0].split(os.path.sep)[slide_no_pos])
        print('\n')
        for patch in images:
            patches.append(patch)
        
    return patches

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
print('dataset_path: ', dataset_path)
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
image_patches_train = read_patches(slide_list_train, 'image20x')
mask_patches_train = read_patches(slide_list_train, 'mask20x')
print('Before RBF:', len(image_patches_train), len(image_patches_train))

print(image_patches_train[199])
print(mask_patches_train[199])

#%%
train_pairs = make_pair(image_patches_train, mask_patches_train)
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
image_patches = read_patches(slide_list_val, 'image20x')
mask_patches = read_patches(slide_list_val, 'mask20x')
print('Before RBF:', len(image_patches), len(mask_patches))

print(image_patches[199])
print(mask_patches[199])

#%%
val_pairs = make_pair(image_patches, mask_patches)
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
image_patches_test = read_patches_test(slide_list_test, 'image20x')
mask_patches_test = read_patches_test(slide_list_test, 'mask20x')
print('Before RBF:', len(image_patches_test), len(mask_patches_test))

print(image_patches_test[1999])
print(mask_patches_test[1999])

#%
test_pairs = make_pair(image_patches_test, mask_patches_test)
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
class DataGenerator(Sequence):
    'Generates data for Keras'
    
    def __init__(self, pair, 
                 batch_size=32, 
                 dim=(patch_size,patch_size,3), 
                 shuffle=True,
                 augmentation=None, 
                 preprocessing=None,
                 inference=False,
                 ):
        'Initialization'
        self.dim = dim
        self.pair = pair
        # self.class_map = class_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.inference = inference

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pair) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        if self.inference:
            # print(list_IDs_temp)
            return X, y, list_IDs_temp
        else:
            return X, y

    def __preprocess_mask_2c(self, mask_data):
        # convert to
        # 0 not change
        # FG: 1 => 0
        # N 2 => 0
        # 3, 4 => 1
        # print('mask_data.shape: ', mask_data.shape)
        mask_data = mask_data.astype('uint8')
        train_masks_input_2c = np.where(mask_data==1, 0, mask_data)
        train_masks_input_2c = np.where(train_masks_input_2c==2, 0, train_masks_input_2c)
        train_masks_input_2c = np.where(train_masks_input_2c==3, 1, train_masks_input_2c)
        train_masks_input_2c = np.where(train_masks_input_2c==4, 1, train_masks_input_2c)
    
        return train_masks_input_2c

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pair))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __preprocess_mask_3c_sumCa_v2(self, mask_data):
        # convert to
        # 0 not change
        # FG: 1 => 0
        # N 2 => 1
        # 3, 4 => 2
        # print('mask_data.shape: ', mask_data.shape)
        mask_data = mask_data.astype('uint8')
        train_masks_input_3c = np.where(mask_data==1, 0, mask_data)
        train_masks_input_3c = np.where(train_masks_input_3c==2, 1, train_masks_input_3c)
        train_masks_input_3c = np.where(train_masks_input_3c==3, 2, train_masks_input_3c)
        train_masks_input_3c = np.where(train_masks_input_3c==4, 2, train_masks_input_3c)
    
        return train_masks_input_3c
    
    def __preprocess_mask_4c(self, mask_data):
        # convert to
        # 0 not change
        # FG: 1 => 0
        # N 2 => 1
        # P3 => 2
        # P4 => 3
        # print('mask_data.shape: ', mask_data.shape)
        mask_data = mask_data.astype('uint8')
        train_masks_input = np.where(mask_data==1, 0, mask_data)
        train_masks_input = np.where(train_masks_input==2, 1, train_masks_input)
        train_masks_input = np.where(train_masks_input==3, 2, train_masks_input)
        train_masks_input = np.where(train_masks_input==4, 3, train_masks_input)
    
        return train_masks_input
    
    def __scale_img(self, single_patch_img):
        single_patch_img_scaled = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        return single_patch_img_scaled
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_labels = list()
        
        # Generate data
        for i in list_IDs_temp:
            # Store sample
            # print(self.pair[i])
            # print(self.pair[i])
            
            # processing image
            image = load_img(self.pair[i][0], target_size=self.dim)
            image = img_to_array(image, dtype='uint8')
            
            
            # processing mask
            dsize = (self.dim[0], self.dim[1])
            # print(dsize)
            # mask = cv2.imread(self.pair[i][1].as_posix(), 0)
            mask = cv2.imread(self.pair[i][1], 0)
            mask = cv2.resize(mask, dsize, interpolation = cv2.INTER_NEAREST)
            
             # apply augmentation                     
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']  

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                # image = self.__scale_img(image)
                image = image * (1/255.0)
                
            mask = self.__preprocess_mask_2c(mask)
            mask = to_categorical(mask, num_classes = n_classes)
            
            batch_imgs.append(image)
            batch_labels.append(mask)
                       
        return np.array(batch_imgs), np.array(batch_labels)
    
#%
import albumentations as A
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ]
    
    return A.Compose(train_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

#% sanity check
def sanity_check(X, y, note, batch_size=32):   
    for i in range(batch_size):
        # image_number = random.randint(0, len(X)-1)
        plt.figure(figsize=(12, 6), dpi=600)
        plt.subplot(121)
        plt.imshow(X[i])
        plt.title(note + ' Image: ' + str(i))
        plt.subplot(122)
        plt.imshow(y[i], cmap='gray')
        (unique, counts) = np.unique(y[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Mask: ' + str(i))
        plt.show()
        print(np.unique(y[i], return_counts=True))
        
#%
def sanity_check_v2(X, y, batch_size=32):   
    for i in range(batch_size):
        # image_number = random.randint(0, len(X)-1)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=600) 
        plt.subplot(121)
        plt.imshow(X[i])
        plt.title('Image: ' + str(i))
        
        plt.subplot(122)
        cmap1 = matplotlib.colors.ListedColormap(['white', #0
                                          'green', #1
                                          'yellow', #2
                                          ])
        cax = ax.imshow(y[i], cmap1,
              interpolation='nearest',
              vmin=0, 
              vmax=2,
              )
        plt.imshow(y[i], cmap=cmap1)
        (unique, counts) = np.unique(y[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Mask: ' + str(i))
        fig.tight_layout()
        plt.show()
        print(np.unique(y_argmax[i], return_counts=True))
        
def sanity_check_v3(X, y, note, pair_idx, pairs):   
    for i, pair_id in enumerate(pair_idx):
        image_file = pairs[pair_id][0].split(os.path.sep)[file_name_pos]
        mask_file = pairs[pair_id][1].split(os.path.sep)[file_name_pos]
        slide_id = pairs[pair_id][1].split(os.path.sep)[slide_no_pos]
        image_id = str(slide_id + '_' + image_file)
        mask_id = str(slide_id + '_' + mask_file)
        
        plt.figure(figsize=(12, 6), dpi=600)
        plt.subplot(121)
        plt.imshow(X[i])
        plt.title(note + ' Image: ' + image_id )
        plt.subplot(122)
        plt.imshow(y[i], cmap='gray')
        (unique, counts) = np.unique(y[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Mask: ' + mask_id )
        plt.show()
        print(np.unique(y[i], return_counts=True))
        
def sanity_check_color(X, y, note, batch_size=32):   
    for i in range(batch_size):
        # image_number = random.randint(0, len(X)-1)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=600) 
        plt.subplot(121)
        plt.imshow(X[i])
        #plt.title('Image: ' + str(i))
        plt.title(note + ' Image: ' + str(i))
        
        plt.subplot(122)
        cmap1 = matplotlib.colors.ListedColormap(['white', #0
                                          'green', #1
                                          'yellow', #2
                                          'orange', #3
                                          ])
        cax = ax.imshow(y[i], cmap1,
              interpolation='nearest',
              vmin=0, 
              vmax=3,
              )
        plt.imshow(y[i], cmap=cmap1)
        (unique, counts) = np.unique(y[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Mask: ' + str(i))
        fig.tight_layout()
        plt.show()
        print(np.unique(y[i], return_counts=True))
    
        
from sklearn.utils import class_weight
def cal_class_weight(labels, classes):
    class_weights = class_weight.compute_class_weight('balanced',
                                        classes,
                                        labels)
    # print("class_weights: ", class_weights) #list
    
    class_weight_dict = dict(enumerate(class_weights))   
    print('class_weight_dict: ', class_weight_dict)

    return class_weight_dict.values()

def get_all_labels(data_generator):
    # X_test_list = []
    y_test_list = []
    
    # for i in range(10):
    for i in tqdm(range(data_generator.__len__())):
        X_test, y_test = data_generator.__getitem__(i)
        # print(i, y_test.shape)
        y_test_list.append(y_test)
        # X_test_list.append(X_test)
         
    # X_test1 = np.concatenate(X_test_list, axis=0)
    y_test1 = np.concatenate(y_test_list, axis=0).astype('uint8')
    y_test1_argmax = np.argmax(y_test1, axis=3).astype('uint8')
    
    return y_test1_argmax

def inference_plot(X, y, y_pred, note, batch_size=16):   
    for i in range(batch_size):
        # image_number = random.randint(0, len(X)-1)
        plt.figure(figsize=(12, 6), dpi=600)
        plt.subplot(131)
        plt.imshow(X[i])
        plt.title(note + ' Image: ' + str(i))
        
        plt.subplot(132)
        plt.imshow(y[i], cmap='gray')
        (unique, counts) = np.unique(y[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Mask: ' + str(i))
        
        plt.subplot(133)
        plt.imshow(y_pred[i], cmap='gray')
        (unique, counts) = np.unique(y_pred[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Predicted: ' + str(i))
        
        
        plt.show()
        print(np.unique(y[i], return_counts=True))
        

def inference_plot_v2(X, y, y_pred, note, pair_idx, pairs):   
    for i, pair_id in enumerate(pair_idx):
        image_file = pairs[pair_id][0].split(os.path.sep)[file_name_pos]
        mask_file = pairs[pair_id][1].split(os.path.sep)[file_name_pos]
        slide_id = pairs[pair_id][1].split(os.path.sep)[slide_no_pos]
        image_id = str(slide_id + '_' + image_file)
        mask_id = str(slide_id + '_' + mask_file)
        
        plt.figure(figsize=(12, 6), dpi=600)
        plt.subplot(131)
        plt.imshow(X[i])
        plt.title(note + ' Image: ' + image_id)
        
        plt.subplot(132)
        plt.imshow(y[i], cmap='gray')
        (unique, counts) = np.unique(y[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Mask: ' + mask_id)
        
        plt.subplot(133)
        plt.imshow(y_pred[i], cmap='gray')
        (unique, counts) = np.unique(y_pred[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Predicted mask')
             
        plt.show()
        print(np.unique(y[i], return_counts=True))
        

        
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
print('\nCreate val_generator_RBF')
val_generator_RBF_train = DataGenerator(val_pair_RBF,
                                batch_size=batch_size, 
                                dim=(patch_size, patch_size, 3),
                                # dim=(target_size,target_size,3),
                                shuffle=True,
                                augmentation=get_training_augmentation(),
                                # preprocessing=get_preprocessing(preprocess_input),
                                )
val_steps_RBF = val_generator_RBF_train.__len__()
print('val_steps_RBF: ', val_steps_RBF)

#%
image_number = random.randint(0, val_steps_RBF)
print('random image number: ', image_number)
X_val, y_val = val_generator_RBF_train.__getitem__(image_number)
print(X_val.shape, y_val.shape)
y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')

sanity_check(X_val, y_val_argmax, note='Val RBF ', batch_size=batch_size)



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
# class_weights = [0.500981666256899, 1.0815061468805043, 
#                  2.0159053357520116, 1.7145968330578576]

#binary class
    class_weights = [0.6847553952019375, 1.8531404575587638]
    
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

# # actulally total_loss can be imported directly from library, 
# # above example just show you how to manipulate with losses
# # total_loss = sm.losses.binary_focal_dice_loss

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


# metrics = ['accuracy', sm_metrics, jacard_coef, dice_coef, 
#            iou_coef1, dice_coef1] 

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


#%%
print("\n\nPerform training...");
print(data_note)
t1 = datetime.now()
with tf.device('/device:GPU:0'):
    model_history = model.fit(
            train_generator_RBF, 
            # steps_per_epoch=train_steps_RBF,
            validation_data=val_generator_RBF,   
            # validation_steps=val_steps_RBF,
            epochs=epochs,
            verbose=verbose,
            callbacks=[
                        reLR,
                        mcp2,
                        # early_stop, 
                        tqdm_callback, 
                        PlotLossesKeras(),
                        ],
            )
t2 = datetime.now() - t1
print("\nTraining time: ", t2)

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

# print('\nbackup_model_best: ', backup_model_best)
# model1 = load_model(backup_model_best, compile=False)
# print(model1.summary())
# print('\nLoaded: ' , backup_model_best)

#%%
from models import (Attention_ResUNet, UNet, Attention_UNet, 
                    dice_coef, dice_coef_loss,
                    jacard_coef, jacard_coef_loss, 
                    iou_coef1, dice_coef1)

sm_metrics = [
            sm.metrics.IOUScore(
            threshold=0.5, 
            class_weights=class_weights,
            class_indexes = classes,
            ), 
            sm.metrics.FScore(
            threshold=0.5, 
              class_weights=class_weights,
              class_indexes = classes,
             )
           ]


metrics = ['accuracy', sm_metrics, 
           #jacard_coef, dice_coef, 
           iou_coef1, dice_coef1]  

#%%
# model1.compile(Adam(lr = lr), 
#                total_loss, 
#                metrics= metrics)

# #%%
# print('\nCreate val_generator_RBF')
# val_generator_RBF_eva = DataGenerator(val_pair_RBF,
#                                 batch_size=batch_size, 
#                                 dim=(patch_size, patch_size, 3),
#                                 # dim=(target_size,target_size,3),
#                                 shuffle=True,
#                                 # augmentation=get_training_augmentation(),
#                                 # preprocessing=get_preprocessing(preprocess_input),
#                                 )
# val_steps_RBF = val_generator_RBF_eva.__len__()
# print('val_steps_RBF: ', val_steps_RBF)

# #%
# image_number = random.randint(0, val_steps_RBF)
# print('random image number: ', image_number)
# X1, y1 = val_generator_RBF_eva.__getitem__(image_number)
# print(X1.shape, y1.shape)
# y_argmax1 = np.argmax(y1, axis=3).astype('uint8')

# sanity_check(X1, y_argmax1, note='Val RBF ', batch_size=batch_size//2)


#%%
# start_exe1 = datetime.now() 
# print("\nEvaluate model for the val set")
# with tf.device('/device:GPU:0'):
#     verbose=1
#     scores = model1.evaluate(val_generator_RBF_eva, 
#                              batch_size=batch_size, 
#                              verbose=verbose)
    
# for metric, value in zip(model1.metrics_names, scores):
#     print("mean {}: {:.4}".format(metric, value))


# start_exe2 = datetime.now() - start_exe1
# print('Execution times: ', start_exe2, '\n')    


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
# print('\nCreate val_generator_RBF_inf')
# val_generator_RBF_inf = DataGenerator(val_pair_RBF,
#                                 batch_size=batch_size, 
#                                 dim=(patch_size, patch_size, 3),
#                                 # dim=(target_size,target_size,3),
#                                 shuffle=False,
#                                 inference=True,
#                                 # augmentation=get_training_augmentation(),
#                                 # preprocessing=get_preprocessing(preprocess_input),
#                                 )
# val_steps_RBF = val_generator_RBF_inf.__len__()
# print('val_steps_RBF: ', val_steps_RBF)

# image_number = random.randint(0, val_steps_RBF)
# print('random image number: ', image_number)
# X_val, y_val, pair_idx_val = val_generator_RBF_inf.__getitem__(image_number)
# print(X_val.shape, y_val.shape)
# y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')

# for i in range(0, len(y_val_argmax)):
#     (unique, counts) = np.unique(y_val_argmax[i], return_counts=True)
#     print(unique, counts)


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
#%%
# print('\Getting X...')
# X_val_list= []
# for batch in tqdm(range(val_generator_RBF_inf.__len__())): 
#     X_val, y_val, pair_idx_val = val_generator_RBF_inf.__getitem__(batch)
#     X_val_list.append(X_val)
#     # pair_idx_val_list.append(pair_idx_val)

# X_val = np.concatenate(X_val_list, axis=0)
# del X_val_list
# print(X_val.shape)

# del X_val

#%%
# print("\nEvaluate the whole val set")
# start_exe1 = datetime.now()
# # X_val_list= []
# y_val_list = []
# y_val_pred_list = []
# pair_idx_val_list = []
   
# # for batch in range(1, val_steps_RBF+1):
# print('\nPredicting...')
# for batch in tqdm(range(val_generator_RBF_inf.__len__())): 
#     X_val, y_val, pair_idx_val = val_generator_RBF_inf.__getitem__(batch)
    
#     # print('\n', X_val.shape, y_val.shape, len(pair_idx_val))
#     y_val_argmax = np.argmax(y_val, axis=3).astype('uint8')
#     verbose=2   
#     y_val_pred = model1.predict(
#                     # test_generator_RBF.__getitem__(image_number),
#                     X_val,   
#                     batch_size=batch_size, 
#                     verbose=verbose)

#     # y_pred_argmax = np.argmax(y_pred, axis=3).astype('uint8')
#     # print('\ny_pred_argmax.shape: ', y_pred_argmax.shape)
#     # print('y_val_argmax.shape: ', y_val_argmax.shape)

#     # X_val_list.append(X_val)
#     y_val_list.append(y_val_argmax)
#     y_val_pred_list.append(y_val_pred)
#     # y_val_pred_argmax_list.append(y_pred_argmax)
#     pair_idx_val_list.append(pair_idx_val)
#     # for idx in pair_idx_val:
#     #     pair_idx_val_list.append(idx)  

# #%
# # X_val1 = np.concatenate(X_val_list, axis=0)
# y_val_true = np.concatenate(y_val_list, axis=0)
# y_val_pred = np.concatenate(y_val_pred_list, axis=0)
# pair_idx_val = np.concatenate(pair_idx_val_list, axis=0)
# print('y_val_true.shape, y_val_pred.shape, pair_idx_val.shape')
# print(y_val_true.shape, y_val_pred.shape, pair_idx_val.shape)

# y_val_pred_argmax = np.argmax(y_val_pred, axis=3) #int64
# print(y_val_pred_argmax.shape)

# np.save("y_val_true.npy", y_val_true)
# np.save("y_val_pred.npy", y_val_pred)

# #%
# # print("\nVal: Calculate Jacard and Dice manually")
# # with tf.device('/device:GPU:0'):
# #     jacard_coef_val = jacard_coef(np.array(y_val_true, dtype='float32'), 
# #                                   np.array(y_val_pred_argmax, dtype='float32')
# #                                 )
                                  
# #     dice_coef_val  = dice_coef(np.array(y_val_true, dtype='float32'), 
# #                                 np.array(y_val_pred_argmax, dtype='float32')
# #                                 )

# # print('jacard_coef:', jacard_coef_val)
# # print('dice_coef:', dice_coef_val)

# start_exe2 = datetime.now() - start_exe1
# print('Execution times: ', start_exe2, '\n')


#%%
# print("\nVal: Calculate iou_coef1 and dice_coef1")
# with tf.device('/device:GPU:0'):
#     val_iou_coef1 = iou_coef1(np.array(y_val_true, dtype='float32'), 
#                                   np.array(y_val_pred_argmax, dtype='float32')
#                                 )
                                  
#     val_dice_coef1  = dice_coef1(np.array(y_val_true, dtype='float32'), 
#                                 np.array(y_val_pred_argmax, dtype='float32')
#                                 )

# print('val_iou_coef1:', val_iou_coef1)
# print('val_dice_coef1:', val_dice_coef1)

# dice_coef2_val = dice_coef2(np.array(y_val_true, dtype='float32'),
#                        np.array(y_val_pred_argmax, dtype='float32')
#                        )
# print(dice_coef2_val)

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


#%%
y_val_true = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-att-res-unet-model-100ep-20221014-1727/y_val_true.npy')

y_val_pred_attresunet = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-att-res-unet-model-100ep-20221014-1727/y_val_pred.npy')
print(y_val_true.shape, y_val_pred_attresunet.shape)

#%%
y_val_pred_attunet = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-att-unet-model-100ep-20221022-2137/y_val_pred.npy')
print(y_val_pred_attunet.shape)

#%%
y_val_pred_unet = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-unet-model-100ep-20221024-1114/y_val_pred.npy')
print(y_val_pred_unet.shape)

#%%
y_test_true = np.load('E:/PCa-Kasikrit/42slides-data-models/backup_model_train-2c-SL-val-att-res-unet-model-100ep-20221014-1727/y_test_true.npy')
print(y_test_true.shape)

#%%
sns.countplot( y_val_true.flatten() )

sns.countplot( y_test_true.flatten() )

#%%
# y_val_pred_argmax = np.argmax(y_val_pred, axis=3).astype('uint8')
# print(y_val_pred_argmax.shape)

# plt.imshow(y_val_pred_argmax[1999], cmap='gray')

#%%
# indx = random.randint(0, len(y_val_true))
# print(indx)
# plt.figure(dpi=300)
# plt.subplot(151)
# plt.imshow(X_val[indx])
# plt.subplot(152)
# plt.imshow(y_val_true[indx], cmap='gray')
# plt.subplot(153)
# plt.imshow(np.argmax(y_val_pred_attresunet[indx], axis=2), cmap='gray')
# plt.subplot(154)
# plt.imshow(np.argmax(y_val_pred_attunet[indx], axis=2), cmap='gray')
# plt.subplot(155)
# plt.imshow(np.argmax(y_val_pred_unet[indx], axis=2), cmap='gray')

#%%
preds = y_val_pred_attresunet + y_val_pred_attunet + y_val_pred_unet
print(preds.shape)

# y_val_pred_attresunet_argmax = np.argmax(y_val_pred_attresunet, axis=3)
# y_val_pred_attunet_argmax = np.argmax(y_val_pred_attunet, axis=3)
# y_val_pred_unet_argmax = np.argmax(y_val_pred_unet, axis=3)

# ensemble_prediction = np.argmax(preds, axis=3)
# print(ensemble_prediction.shape)

del y_val_pred_attresunet
del y_val_pred_attunet
del y_val_pred_unet

# del prediction1
# del prediction2
# del prediction3

#%%
preds_sum_avg = preds/3.0
print(preds_sum_avg.shape)

preds_sum_avg_argmax = np.argmax(preds_sum_avg, axis=3)
print(preds_sum_avg_argmax.shape)

#%%,
random_ratio = 0.10 #828

print("\nCompute MeanIoU for Val ratio ", random_ratio)
random_size = int(len(y_val_true) * random_ratio)
batch_idx = random.sample(range(1, len(y_val_true)), random_size)
print('len(batch_idx): ', len(batch_idx))

#%%
import utility
utility.unique_counts(preds_sum_avg_argmax[batch_idx])

preds_samples = preds_sum_avg_argmax[batch_idx]
pred_attresunet_samples_argmax = np.argmax(y_val_pred_attresunet[batch_idx], axis=3)
print(preds_samples.shape, pred_attresunet_samples_argmax.shape)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, cohen_kappa_score

y_true = y_val_true[batch_idx].flatten()
y_hat = preds_samples.flatten()

sns.countplot(y_true)

cm = confusion_matrix(y_true, y_hat)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
bal_acc = (sensitivity+specificity)/2.0
kappa = cohen_kappa_score(y_true, y_hat)

print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))
print("Balance acc: {:.4f}".format( bal_acc ))
print('Cohen_Kappa {:.4}'.format( kappa ))

# [[35991998  2752050]
#  [ 3264626 12255134]]
# acc: 0.8891
# sensitivity: 0.9290
# specificity: 0.7896
# Balance acc: 0.8593
# Cohen_Kappa 0.7258

#%%
target_names = ['Benign', 'Malignant']
cf = classification_report(y_pred=y_hat,
                            y_true=y_true,
                            target_names=target_names,
                            output_dict = True
                          )
print(type(cf))
print(cf.keys())

# {'Benign': {'precision': 0.9168388499224998,
#   'recall': 0.9289684443917683,
#   'f1-score': 0.9228637927632214,
#   'support': 38744048},
#  'Malignant': {'precision': 0.8166178278349889,
#   'recall': 0.7896471337185627,
#   'f1-score': 0.8029060491610297,
#   'support': 15519760},
#  'accuracy': 0.8891217512784949,
#  'macro avg': {'precision': 0.8667283388787443,
#   'recall': 0.8593077890551655,
#   'f1-score': 0.8628849209621255,
#   'support': 54263808},
#  'weighted avg': {'precision': 0.8881750633752514,
#   'recall': 0.8891217512784949,
#   'f1-score': 0.8885551907784962,
#   'support': 54263808}
#  }

#%%
# https://www.mathsisfun.com/data/standard-deviation-formulas.html
# Standard Deviation of F1-score
mean_f1score = cf['weighted avg']['f1-score']
sum_msd = 0
for key, value in cf.items():
    if key in target_names:
        #print(type(value), value)
        mean_square_dif = (value['f1-score'] - mean_f1score)**2
        sum_msd += mean_square_dif
        print(key, 'f1-score: {:2.4f}'.format(value['f1-score']),
              'mean_square_dif: {:2.5f}'.format(mean_square_dif )
             )
        
std_f1score = (sum_msd*(1/len(target_names)))**(1/2)
print('mean_f1score: {:2.4f}'.format(mean_f1score), 
      'sd_f1score: {:2.4f}'.format(std_f1score))


#%%
import seaborn as sns
#target_names = labelList

plt.figure(figsize=(3, 2), dpi=300)
accuracy = np.trace(cm) / float(np.sum(cm))
misclass = 1 - accuracy
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, fontsize=8)
plt.yticks(tick_marks, target_names, fontsize=8)

#sns.set(font_scale=1.4)
cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True)
cm_nor = cm.astype('float32') / cm.sum(axis=1)[:, np.newaxis]
ax = sns.heatmap(cm_nor, annot=True,
                 cbar=True,
                 fmt='.2f', 
                 linewidths=.05,
                 xticklabels=target_names,
                 yticklabels=target_names,
                 cmap=cmap
                )

plt.tight_layout()
plt.ylabel('True label', fontsize=10)
plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass),
          fontsize=10)

#%%
# from sklearn.metrics import jaccard_score
# y_true = y_val_true[batch_idx].flatten()
# y_pred = ensemble_prediction[batch_idx].flatten()
# print(y_true.shape, y_pred.shape)
# jaccardScore = jaccard_score(y_true, y_pred, average="micro")


#%%
# values = compute_MeanIoU(y_val_true[batch_idx], y_val_pred_argmax[batch_idx])
# print('len(ba♣tch_idx): ', len(batch_idx))

#%%
# values1 = compute_MeanIoU(y_val_true[batch_idx], prediction1[batch_idx])
# values2 = compute_MeanIoU(y_val_true[batch_idx], prediction2[batch_idx])
# values3 = compute_MeanIoU(y_val_true[batch_idx], prediction3[batch_idx])
# values4 = compute_MeanIoU(y_val_true[batch_idx], ensemble_prediction[batch_idx])

# print(values1[0])
# print(values2[0])
# print(values3[0])
# print(values4[0])

#%%
print("\nFind weighted for models")
random_ratio = 0.33
batch_idx = generate_batch_idx(random_ratio)
print(len(batch_idx))

t1 = datetime.now()
import pandas as pd
df = pd.DataFrame([])

with tf.device('/device:GPU:0'):
    for w1 in tqdm(range(0, 5)):
        for w2 in range(0,5):
            for w3 in range(0,5):
                batch_idx = generate_batch_idx(random_ratio)
                wts = [w1/10., w2/10., w3/10.]
                # print(wts)               
                wted_preds = np.tensordot(preds[batch_idx], wts, axes=0) 
                print('\n', wted_preds.shape)
                wted_preds_sum = K.sum(wted_preds, axis=4)
                print(wted_preds_sum.shape)
                wted_ensemble_pred = np.argmax(wted_preds_sum, axis=3)
                print(wted_ensemble_pred.shape)
                # plt.imshow(wted_ensemble_pred[80],cmap='gray')
                
                iou_wted = compute_MeanIoU(y_val_true[batch_idx], wted_ensemble_pred)            
                
                df = df.append(pd.DataFrame({'wt1':wts[0],
                                     'wt2':wts[1],
                                     'wt3':wts[2],
                                     'iou': iou_wted[0],
                                      'iou_class0': iou_wted[1],
                                      'iou_class1': iou_wted[2],
                                     }, 
                                    index=[0]), 
                                ignore_index=True)
t2 = datetime.now() - t1
print('\nExecution time: ', t2)

#ratio 0.1=> 828 Execution time:  0:08:44.344358

#%%
max_acc_row = df.iloc[df['iou'].idxmax()]
print(max_acc_row)

df.to_csv('max_row.csv')
    
#%%
max_acc_row0 = df.iloc[df['iou_class0'].idxmax()]
print(max_acc_row0)

#%%
max_acc_row1 = df.iloc[df['iou_class1'].idxmax()]
print(max_acc_row1)


#%%
print('\nApplication test for weighted ensemble')
weights_arr = np.array([max_acc_row[0],
           max_acc_row[1],
           max_acc_row[2],
           ])
# weights = weights_arr + [1]

wted_preds = np.tensordot(preds[batch_idx], weights_arr, axes=0) 
print(wted_preds.shape)

# wted_preds = np.tensordot(preds[batch_idx], weights, axes=0) 
# print(wted_preds.shape) => same as weights_arr

wted_preds_sum = K.sum(wted_preds, axis=4)
print(wted_preds_sum.shape)

wted_ensemble_pred = np.argmax(wted_preds_sum, axis=3)
print(wted_ensemble_pred.shape)

iou_wted = compute_MeanIoU(y_val_true[batch_idx], wted_ensemble_pred)  
print(iou_wted)


