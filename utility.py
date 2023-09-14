# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:18:23 2022

@author: kasikrit
"""
import tensorflow as tf
import segmentation_models as sm
import albumentations as A

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
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


#wted1
wted_unet = 0.8
wted_att_unet = 0.8
wted_att_res_unet = 1.0

#test for Glas-dataset
# wted_unet = 1.0
# wted_att_unet = 1.0
# wted_att_res_unet = 1.0

'''
A few useful metrics and losses
'''

def dice_coef(y_true, y_pred): #F1-score
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred): #IoU
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
axis = [0]
def iou_coef1(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=axis)
  union = K.sum(y_true, axis=axis)+K.sum(y_pred, axis=axis)-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_coef1(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=axis)
  union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice


#https://www.kaggle.com/code/hashbanger/skin-lesion-segmentation-using-fcn/notebook
def iou(y_true, y_pred, smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.square(y_true), axis = -1) + K.sum(K.square(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jaccard_distance(y_true, y_pred, smooth=100):
    return (1 - iou(y_true, y_pred))

def dice_coef2(y_true, y_pred, smooth = 100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return K.mean(K.equal(y_true, K.round(y_pred)))

#%%
def get_weighted_for_model():
    return np.array([wted_unet, wted_att_unet, wted_att_res_unet])

def get_single_cancer_ensemble_prob(single_patch_preds_arr):
    
    wted_preds = np.tensordot(single_patch_preds_arr, get_weighted_for_model(), axes=0) 
    # print(wted_preds.shape)
    
    wted_preds_sum = np.sum(wted_preds, axis=0)
    # print(wted_preds_sum.shape)
    wted_preds_sum1 = np.sum(wted_preds_sum, axis=4)
    
    wted_preds_sum_avg = wted_preds_sum1/len(model_list)
    # print(wted_preds_sum_avg.shape)
    
    wted_preds_sum_avg_argmax = np.argmax(wted_preds_sum_avg, axis=3)
    # print(wted_preds_sum_avg_argmax.shape)
    
    wted_preds_sum_avg_prob = np.zeros((patch_size,patch_size))
    wted_preds_sum_avg_prob = np.where(wted_preds_sum_avg_argmax==1,
                                wted_preds_sum_avg[:,:,:,1], 
                                wted_preds_sum_avg_prob)
    # print(wted_preds_sum_avg_prob.shape)
    return wted_preds_sum_avg_prob

def detect_bg(patch, prob=False):
        # image = cv2.imread('test24.png')
        # image = np.array(patch * 255, dtype='uint8')
            
        h, w, _ = patch.shape
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
            
        pixels = cv2.countNonZero(thresh)
        pixel_ratio = (pixels/(h * w)) * 100
        title = 'Pixel ratio: {:.2f}%'.format(pixel_ratio)
           
        if prob:
            print(title)
            plt.figure()  
            plt.subplot(1,2,1)
            plt.title("Original Patch")
            plt.imshow(patch)
            plt.subplot(1,2,2)
            plt.title(title)
            plt.imshow(thresh)
        
        return pixel_ratio
    
def example_function(thumb_gray):
    assert len(thumb_gray.shape) == 2
    assert isinstance(thumb_gray, np.ndarray)
    thresh = 100
    thumb_gray[thumb_gray > thresh] = 1
    thumb_gray[thumb_gray <= thresh] = 0
    assert np.sum((thumb_gray == 0) | (thumb_gray == 1)) == len(thumb_gray)
    return thumb_gray


def extract_annotation_region(slide_id, anotation_file, rule_file, output_dir):
    slide = wp.slide(slide_id)
    annotation = wp.annotation(annotation_file)
    rule = wp.rule(rule_file)
    annotation.make_masks(slide, rule,
                         foreground='otsu',
                         size=2000,
                        )

    annotation.export_thumb_masks(output_dir)

#% Extract large image and mask
#% patch slide and mask
def find_targetDims(TILE_SIZE, width, height, lower, upper):
    df0 = []
    df1 = []
    for i in np.arange(lower, upper):
        dif0 = np.abs(width - i*TILE_SIZE)
        df0.append([dif0, i*TILE_SIZE])

    for i in np.arange(lower, upper):
        dif1 = np.abs(height - i*TILE_SIZE)
        df1.append([dif1, i*TILE_SIZE])

    df0.sort()
    df1.sort()
    s0 = df0[0][1]
    s1 = df1[0][1]

    return s0, s1

def scale_img(single_patch_img):
        single_patch_img_scaled = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        return single_patch_img_scaled
    
def normalize_img(single_patch_img):
    return single_patch_img * (1/255.)


def plot_predicted_4c(slide_id, mask_data, dpi=1200, filename=None):    
    
    fig, ax = plt.subplots(dpi=dpi)     
    
    cmap1 = matplotlib.colors.ListedColormap([
                        'white', #0
                         '#00b300', #1 Benign, Green
                         'yellow', #2 GP3
                         'orange', #3 GP4
                         ])

    cax = ax.imshow(mask_data, cmap1,
              interpolation='nearest',
              vmin=0, 
              vmax=3
              )
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([
                            'Forground',
                            'Normal',
                            'P3', 
                            'P4',
                             ],
                            fontsize=4,
                            )  # vertically oriented colorbar
    fig.tight_layout()

    # fig.colorbar(plot)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Prediction of slide: ' + slide_id, fontsize=8)
    _ = plt.show()
    
    if filename is not None:
        fig.savefig(filename)
        print("Saved the plot file")
        
#%%%
    
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

def read_patches(slide_list, patch_type, dataset_directory):
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

#%%
class DataGenerator(Sequence):
    'Generates data for Keras'
    
    def __init__(self, 
                 pair, 
                 num_classes,
                 batch_size=32, 
                 dim=(256,256,3), 
                 shuffle=True,
                 augmentation=None, 
                 preprocessing=None,
                 inference=False,
                 ):
        'Initialization'
        self.dim = dim
        self.pair = pair
        self.num_classes = num_classes
        # self.class_map = class_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.inference = inference

        print('num_classes: ', num_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pair) / self.batch_size))
    
    def foo(self):
        print('foo')

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
        
    def __preprocess_mask_3c_sumCa(self, mask_data):
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
    
    def __preprocess_mask_3c_g3(self, mask_data):
        # convert to
        # 0 not change
        # FG: 1 => 0
        # N 2 => 1
        # GP3 => 2
        # print('mask_data.shape: ', mask_data.shape)
        mask_data = mask_data.astype('uint8')
        train_masks_input_3c = np.where(mask_data==1, 0, mask_data)
        train_masks_input_3c = np.where(train_masks_input_3c==2, 1, train_masks_input_3c)
        train_masks_input_3c = np.where(train_masks_input_3c==3, 2, train_masks_input_3c)
        #train_masks_input_3c = np.where(train_masks_input_3c==4, 2, train_masks_input_3c)
    
        return train_masks_input_3c
    
    def __preprocess_mask_3c_g3_g4(self, mask_data):
        #labels = ['BG+FG+Normal', 'GP3', 'GP4']
        # convert to
        # 0 not change
        # FG: 1 => 0
        # N 2 => 0
        # 3 => 1
        # 4 => 2
        # print('mask_data.shape: ', mask_data.shape)
        mask_data = mask_data.astype('uint8')
        train_masks_input_3c = np.where(mask_data==1, 0, mask_data)
        train_masks_input_3c = np.where(train_masks_input_3c==2, 0, train_masks_input_3c)
        train_masks_input_3c = np.where(train_masks_input_3c==3, 1, train_masks_input_3c)
        train_masks_input_3c = np.where(train_masks_input_3c==4, 2, train_masks_input_3c)
    
        return train_masks_input_3c
    
    def __preprocess_mask_2c_g3(self, mask_data):
        # convert to
        # 0 not change
        # FG: 1 => 0
        # N 2 => 0
        # 3 => 1
        # print('mask_data.shape: ', mask_data.shape)
        mask_data = mask_data.astype('uint8')
        train_masks_input_2c = np.where(mask_data==1, 0, mask_data)
        train_masks_input_2c = np.where(train_masks_input_2c==2, 0, train_masks_input_2c)
        train_masks_input_2c = np.where(train_masks_input_2c==3, 1, train_masks_input_2c)
    
        return train_masks_input_2c
    
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
    
    def __preprocess_mask_3c_BgFg_N_GP3(self, mask_data):
        # convert to
        # 0 not change
        # FG: 1 => 0
        # N 2 => 1
        # 3 => 2
        # print('mask_data.shape: ', mask_data.shape)
        mask_data = mask_data.astype('uint8')
        train_masks_input_3c = np.where(mask_data==1, 0, mask_data)
        train_masks_input_3c = np.where(train_masks_input_3c==2, 1, train_masks_input_3c)
        train_masks_input_3c = np.where(train_masks_input_3c==3, 2, train_masks_input_3c)
        #train_masks_input_3c = np.where(train_masks_input_3c==4, 2, train_masks_input_3c)
    
        return train_masks_input_3c
     
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
            
            # apply masking
            if self.num_classes==2:
                # mask = self.__preprocess_mask_2c(mask) 
                mask = self.__preprocess_mask_2c_g3(mask)
            elif self.num_classes==3:
                # mask = self.__preprocess_mask_3c_p3(mask)
                # mask = self.__preprocess_mask_3c_g3_g4(mask)
                mask3 = self.__preprocess_mask_3c_BgFg_N_GP3(mask)
            else:
                mask4 = self.__preprocess_mask_4c(mask)
            
            # print('mask: ', mask.shape)
            # print(np.unique(mask, return_counts=True))
            mask3c = to_categorical(mask3, self.num_classes)
            # print('after mask: ', mask3c.shape)
            
            batch_imgs.append(image)
            batch_labels.append(mask3c)
                       
        return np.array(batch_imgs), np.array(batch_labels)
    
#%%
def preprocess_mask_2c(mask_data):
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
    
def preprocess_mask_4c(mask_data):
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
def sanity_check(X, y, note, batch_size=16):   
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
def sanity_check_v2(X, y, batch_size=16):   
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

def get_all_labels(data_generator, steps):
    # X_test_list = []
    y_test_list = []
    
    # for i in tqdm(range(data_generator.__len__())):
    for i in tqdm(steps):    
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
        

def inference_plot_3(X, y, y_pred, note, pair_idx, pairs):   
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
        plt.title('Ground truth: ' + mask_id)
        
        plt.subplot(133)
        plt.imshow(y_pred[i], cmap='gray')
        (unique, counts) = np.unique(y_pred[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Probability predicted mask' + ' (' + str(pair_id) + ')')
             
        plt.show()
        print(np.unique(y[i], return_counts=True))
        
 #%%
from sklearn.metrics import confusion_matrix  
import numpy as np

def compute_iou(y_pred, y_true, classes):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=classes)
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

from tensorflow.keras.metrics import MeanIoU
def compute_MeanIoU_4c(y_pred, y_true, sample_weight=None):
    t3 = datetime.now()
    print("\nCompute MeanIoU using keras.metrics.MeanIoU")
    print(y_pred.shape, y_true.shape)
    n_classes=4
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_pred, y_true)
    
    if sample_weight is not None:
        print('Use class weights:')
        print(sample_weight)
        IOU_keras.update_state(y_pred, y_true, sample_weight)
    
    meanIoU = IOU_keras.result().numpy()
    print("Mean IoU = ", meanIoU)
   
    print('To calculate IoU for each class...')
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    print(values)
    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
    class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])
    
    print("IoU for class0 is: ", class1_IoU)
    print("IoU for class1 is: ", class2_IoU)
    print("IoU for class2 is: ", class3_IoU)
    print("IoU for class3 is: ", class4_IoU)
   
    t4 = datetime.now() - t3
    print("Exe time: ", t4)
    
    return [meanIoU,class1_IoU, class2_IoU, class3_IoU, class4_IoU]

def compute_MeanIoU_2c(y_pred, y_true, sample_weight=None):
    t3 = datetime.now()
    # print("\nCompute MeanIoU using keras.metrics.MeanIoU")
    # print(y_pred.shape, y_true.shape)
    n_classes=2
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_pred, y_true)
    
    if sample_weight is not None:
        # print('Use class weights:')
        # print(sample_weight)
        IOU_keras.update_state(y_pred, y_true, sample_weight)
    
    meanIoU = IOU_keras.result().numpy()
    # print("Mean IoU = ", meanIoU)
   
    print("To calculate IoU for each class...")
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    # print(values)
    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[1,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[0,1])
    # class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
    # class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])
    
    # print("IoU for class0 is: ", class1_IoU)
    # print("IoU for class1 is: ", class2_IoU)
    # print("IoU for class2 is: ", class3_IoU)
    # print("IoU for class3 is: ", class4_IoU)
    
    t4 = datetime.now() - t3
    print("Exe time: ", t4)
    
    return [meanIoU, class1_IoU, class2_IoU]

def compute_MeanIoU_3c(y_pred, y_true, sample_weight=None):
    t3 = datetime.now()
    # print("\nCompute MeanIoU using keras.metrics.MeanIoU")
    # print(y_pred.shape, y_true.shape)
    n_classes=3
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_pred, y_true)
    
    if sample_weight is not None:
        # print('Use class weights:')
        # print(sample_weight)
        IOU_keras.update_state(y_pred, y_true, sample_weight)
    
    meanIoU = IOU_keras.result().numpy()
    print("Mean IoU = ", meanIoU)
   
    print("To calculate IoU for each class...")
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    # print(values)
    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[0,2]+ values[1,2])
    #class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])
    
    print("IoU for class0 is: ", class1_IoU)
    print("IoU for class1 is: ", class2_IoU)
    print("IoU for class2 is: ", class3_IoU)
    # print("IoU for class3 is: ", class4_IoU)
    
    t4 = datetime.now() - t3
    print("Exe time: ", t4)
    
    return [meanIoU, class1_IoU, class2_IoU, class3_IoU]

def dice_coef2(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred)
  union = K.sum(y_true) + K.sum(y_pred)
  dice = K.mean((2. * intersection + smooth)/(union + smooth) )
  return dice

def unique_counts(array):
    (unique, counts) = np.unique(array, return_counts=True)
    print(unique, counts)
    


#%%  
def getPatchMask4c(pairs, image_number):
    pair = pairs[image_number]
    # processing image
    image = load_img(pair[0], target_size=(patch_size,patch_size,3))
    image = img_to_array(image, dtype='uint8')
    # plt.figure()
    # plt.imshow(image)
    
    # processing mask
    dsize = (patch_size, patch_size)
    # print(dsize)
    # mask = cv2.imread(self.pair[i][1].as_posix(), 0)
    mask = cv2.imread(pair[1], 0)
    mask = cv2.resize(mask, dsize, interpolation = cv2.INTER_NEAREST)
    # plt.figure()
    # plt.imshow(mask)
    mask4c = preprocess_mask_4c(mask)
    
    return image, mask4c

def plot_crosscheck_4c(pairs, y_true, y_pred, image_number, note):
    image, mask = getPatchMask4c(pairs, image_number)
    plt.figure(figsize=(12, 8), dpi=600)
    plt.subplot(141)
    plt.imshow(image)
    plt.title(note + ' image_number: ' + str(image_number))
       
    plt.subplot(142)
    plt.imshow(mask, cmap='gray')
    plt.title('mask')
    (unique, counts) = np.unique(mask, return_counts=True)
    xlabel = str(unique) + "\n" + str(counts)
    plt.xlabel(xlabel)
     
    plt.subplot(143)
    plt.imshow(y_true[image_number], cmap='gray')
    plt.title('mask (Datagen)')
    (unique, counts) = np.unique(y_true[image_number], return_counts=True)
    xlabel = str(unique) + "\n" + str(counts)
    plt.xlabel(xlabel)
    
    plt.subplot(144)
    plt.imshow(y_pred[image_number], cmap='gray')
    plt.title('prediction 4c')
    (unique, counts) = np.unique(y_pred[image_number], return_counts=True)
    xlabel = str(unique) + "\n" + str(counts)
    plt.xlabel(xlabel)

def generate_batch_idx(random_ratio):
    random_size = int(len(y_val_true) * random_ratio)
    batch_idx = random.sample(range(1, len(y_val_true)), random_size)
    # print('len(batch_idx): ', len(batch_idx))
    return batch_idx

def inference_plot_prob(X, y, y_pred, note, pair_idx, pairs):   
    for i, pair_id in enumerate(pair_idx):
        # print(i, pair_id)
        image_file = pairs[pair_id][0].split(os.path.sep)[file_name_pos]
        mask_file = pairs[pair_id][1].split(os.path.sep)[file_name_pos]
        slide_id = pairs[pair_id][1].split(os.path.sep)[slide_no_pos]
        image_id = str(slide_id + '_' + image_file)
        mask_id = str(slide_id + '_' + mask_file)
        
        plt.figure(figsize=(12, 8), dpi=600)
        # plt.figure(dpi=600)
        # fig, ax = plt.subplots(figsize=(12, 8), dpi=600) 
        
        plt.subplot(131)
        plt.imshow(X[i])
        plt.title(note + ' Image: ' + image_id)
        
        plt.subplot(132)
        plt.imshow(y[i], cmap='RdYlBu_r', vmin=0.0, vmax=1.0)
        (unique, counts) = np.unique(y[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Ground truth: ' + mask_id)
        
        y_pred_patch = y_pred[i]
        y_pred_patch_argmax = np.argmax(y_pred_patch, axis=2)
        
              
        y_pred_patch_prob = np.zeros((patch_size,patch_size))
        # print(y_pred_patch_prob.shape)
        # y_pred_patch_prob = np.where(y_pred_patch_argmax==0,
        #                               y_pred_patch[:,:,1], 
        #                               y_pred_patch_prob)
        y_pred_patch_prob = np.where(y_pred_patch_argmax==1,
                                     y_pred_patch[:,:,1], 
                                     y_pred_patch_prob)

        xlabel = str(i) + ': index ' + str(pair_id)  
        print('\n', xlabel)
        print(np.min(y_pred_patch_prob), np.max(y_pred_patch_prob) )
   
        plt.subplot(133)
        plt.imshow(y_pred_patch_prob, 
                    cmap='RdYlBu_r',
                    interpolation='bicubic',
                    vmin=0.0, 
                    vmax=1.0)
        # plt.colorbar(label = 'Cancer Probability')
        # plt.xlabel(xlabel)
        plt.title('Probability prediction')
        
        plt.show()
        

#%%
def inference_plot_prob_grid3(X, y, y_pred, note, pair_idx, pairs):   
    for i, pair_id in enumerate(pair_idx):
        print(i, pair_id)
        image_file = pairs[pair_id][0].split(os.path.sep)[file_name_pos]
        mask_file = pairs[pair_id][1].split(os.path.sep)[file_name_pos]
        slide_id = pairs[pair_id][1].split(os.path.sep)[slide_no_pos]
        image_id = str(slide_id + '_' + image_file)
        mask_id = str(slide_id + '_' + mask_file)
        
        fig = plt.figure(figsize=(12, 12), dpi=600)

        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,3),
                  axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                  cbar_mode="edge",
                  cbar_size="10%",
                 # cbar_pad=0.15,
                 )
        
        y_pred_patch = y_pred[i]
        y_pred_patch_argmax = np.argmax(y_pred_patch, axis=2)
        
              
        y_pred_patch_prob = np.zeros((patch_size,patch_size))
        # print(y_pred_patch_prob.shape)
        # y_pred_patch_prob = np.where(y_pred_patch_argmax==0,
        #                               y_pred_patch[:,:,1], 
        #                               y_pred_patch_prob)
        y_pred_patch_prob = np.where(y_pred_patch_argmax==1,
                                     y_pred_patch[:,:,1], 
                                     y_pred_patch_prob)

        xlabel = str(i) + ': index ' + str(pair_id)  
        print('\n', xlabel)
        print(np.min(y_pred_patch_prob), np.max(y_pred_patch_prob) )

        patch_list = [X[i], y[i], y_pred_patch_prob]
        
        for i, ax in enumerate(grid):
            # print(ax)
            # im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)
            im = ax.imshow(patch_list[i], 
                            cmap='RdYlBu_r',
                            interpolation='bicubic',
                            vmin=0.0, 
                            vmax=1.0)
            
        grid[0].set_title(note + '-Image: ' + image_id)
        grid[1].set_title('Ground truth: ' + mask_id)
        grid[2].set_title('Probability prediction' + ' (' + str(pair_id) + ')')
        # Colorbar
        cb = ax.cax.colorbar(im)
        ax.cax.toggle_label(True)
        cb.set_label_text('Cancer Probability')
        plt.show()
        
def inference_plot_prob_all(X, y, pred_prob_list, note, pair_idx, pairs, SAVE=None):
    #y = > y in argmax
    #y_preds => predicted array using all models
    print(X.shape)
    print(y.shape)
    print(len(pred_prob_list))
    #Unet=0, Att-unet=1, Att-res-unet=2, ensemble=3
    unet_prob = pred_prob_list[0]
    att_unet_prob = pred_prob_list[1]
    att_res_unet_prob = pred_prob_list[2]
    ensemble_prob = pred_prob_list[3]
    
    for i, pair_id in enumerate(pair_idx):
        print(i, pair_id)
        image_file = pairs[pair_id][0].split(os.path.sep)[file_name_pos]
        mask_file = pairs[pair_id][1].split(os.path.sep)[file_name_pos]
        slide_id = pairs[pair_id][1].split(os.path.sep)[slide_no_pos]
        image_id = str(slide_id + '_' + image_file)
        mask_id = str(slide_id + '_' + mask_file)
        
        fig = plt.figure(figsize=(12, 8), dpi=600)
        # plt.figure(dpi=600)
        # fig, ax = plt.subplots(figsize=(12, 8), dpi=600) 
        
        plt.subplot(231)
        plt.imshow(X[i])
        title = str(i) + '-' + note + '-Image-' + image_id
        plt.title(title)
        
        plt.subplot(232)
        plt.imshow(y[i], cmap='RdYlBu_r', vmin=0.0, vmax=1.0)
        (unique, counts) = np.unique(y[i], return_counts=True)
        xlabel = str(unique) + "\n" + str(counts)
        plt.xlabel(xlabel)
        plt.title('Ground truth: ' + mask_id)
        
        patch_label = str(i) + ': index ' + str(pair_id)  
        print('\n', patch_label)
   
        #Unet
        plt.subplot(233)
        plt.imshow(unet_prob[i], 
                    cmap='RdYlBu_r',
                    interpolation='bicubic',
                    vmin=0.0, 
                    vmax=1.0)
        plt.title('Unet probability prediction')
        print(np.min(unet_prob[i]), np.max(unet_prob[i]) )
        
        #att-unet
        plt.subplot(234)
        plt.imshow(att_unet_prob[i], 
                    cmap='RdYlBu_r',
                    interpolation='bicubic',
                    vmin=0.0, 
                    vmax=1.0)
        plt.title('Att-unet probability prediction')
        print(np.min(att_unet_prob[i]), np.max(att_unet_prob[i]) )
        
        #att-res-unet
        plt.subplot(235)
        plt.imshow(att_res_unet_prob[i], 
                    cmap='RdYlBu_r',
                    interpolation='bicubic',
                    vmin=0.0, 
                    vmax=1.0)
        plt.title('Att-res-unet probability prediction')
        print(np.min(att_res_unet_prob[i]), np.max(att_res_unet_prob[i]) )
        
        #att-res-unet
        plt.subplot(236)
        im = plt.imshow(ensemble_prob[i], 
                    cmap='RdYlBu_r',
                    interpolation='bicubic',
                    vmin=0.0, 
                    vmax=1.0)
        plt.title('Ensemble probability prediction')
        print(np.min(ensemble_prob[i]), np.max(ensemble_prob[i]) )
        cb = plt.colorbar(im)
        fig.tight_layout()
        plt.show()
        
        if SAVE:
            fig_file = title + '.png'
            fig_file_path = os.path.join('G:\My Drive\PCa-Kasikrit\segmentation\inference-ensemble-2c' , 
                                         fig_file)
            fig.savefig(fig_file_path, dpi=600)
            print('saved: ', fig_file_path)


def inference_plot_v2(X, y, y_pred, note, pair_idx, pairs, slide_no_pos):
    file_name_pos = slide_no_pos + 2
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
        
def read_patches_test(slide_list, patch_type):
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






     



