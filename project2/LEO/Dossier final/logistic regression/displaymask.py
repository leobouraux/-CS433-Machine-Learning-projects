# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import nose.tools
import scipy.misc
import scipy.ndimage
import skimage.filters
import sklearn.metrics
import cv2
from cross_validation import *
from logistic_regression import *

def predictions(model, Y, X, k_fold, k, patch_size):
    seed = 1
    k_indices = build_k_indices(Y, k_fold, seed)
    non_k_indices = k_indices[np.arange(k_indices.shape[0]) != k].ravel()
    tx_tr = X[non_k_indices]
    y_tr = Y[non_k_indices]
    tx_te = X[k_indices[k]]
    y_te = Y[k_indices[k]]
    
    model.initialize_model() # Reset model
    model.train(y_tr, tx_tr,patch_size)
    
    # Run model
    Z = model.logisticReg(tx_te, patch_size)
    return Z , tx_te


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# Convert array of labels to an image

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img