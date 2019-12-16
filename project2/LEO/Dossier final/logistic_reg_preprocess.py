# -*- coding: utf-8 -*-

import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import re
import scipy.misc
import scipy.ndimage


def load_images(path, n):
    files = os.listdir(image_dir)
    print("Loading " + str(n) + " images")
    imgs = np.asarray([mpimg.imread(image_dir + files[i]) for i in range(n)])


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches
    
def create_patches(X, patch_size):
    img_patches = np.asarray([img_crop(X[i], patch_size, patch_size) for i in range(X.shape[0])])
    # Linearize list
    img_patches = img_patches.reshape(-1, img_patches.shape[2], img_patches.shape[3])
    return img_patches
    
def group_patches(patches, num_images):
    return patches.reshape(num_images, -1)


def median(imgs,Size,n):
    median = []
    for i in range(n):
        median.append(scipy.ndimage.median_filter(imgs[i], size=Size))
    return median

def initialize_clusters(data, k):
    num_sample, num_feature = data.shape[0], data.shape[1]
    return data[np.random.choice(num_sample,k)]

def build_distance_matrix(data, mu):
    """build a distance matrix.
    return
        distance matrix:
            row of the matrix represents the data point,
            column of the matrix represents the k-th cluster.
    """
    distance_list = []
    num_cluster, _ = mu.shape
    for k_th in range(num_cluster):
        sum_squares = np.sum(np.square(data - mu[k_th, :]), axis=1)
        distance_list.append(sum_squares)
    return np.array(distance_list).T
    
def update_kmeans_parameters(data, mu_old):
    """update the parameter of kmeans
    return:
        losses: loss of each data point with shape (num_samples, 1)
        assignments: assignments vector z with shape (num_samples, 1)
        mu: mean vector mu with shape (k, num_features)
    """
    _, num_features = data.shape
    num_clusters, _ = mu_old.shape
    distance_matrix = build_distance_matrix(data, mu_old)
    losses = np.min(distance_matrix, axis=1)
    assignments = np.argmin(distance_matrix, axis=1)

    # update the mu
    mu = np.empty((num_clusters, num_features))
    for k_th in range(num_clusters):
        rows = np.where(assignments == k_th)
        mu[k_th, :] = np.mean(np.squeeze(data[rows, :]), axis=0)
    return losses, assignments, np.nan_to_num(mu)  
    
def preprocess_image_rgb(original_image):
    """preprocess the image."""
    processed_image = original_image.reshape(
        (original_image.shape[0] * original_image.shape[1], 3))
    processed_image = processed_image.astype(float)
    
    return processed_image

def kmean_compression(original_image, processed_image, k, max_iter, threshold):
    """using k-means for image compression."""
    mu_old = initialize_clusters(processed_image, k)
    
    # init some empty lists to store the result.
    loss_list = []
    
    # start the kmeans
    for iter in range(max_iter):
        losses, assignments, mu = update_kmeans_parameters(processed_image, mu_old)
        
        # calculate the average loss over all points
        average_loss = np.mean(losses)
        loss_list.append(average_loss)
        # check converge
        if iter > 0 and np.abs(loss_list[-1] - loss_list[-2]) < threshold:
            break

        # update mu
        mu_old = mu
    # Clustered Image  
    image_reconstruct = mu[assignments].reshape(original_image.shape)
    
    return image_reconstruct