# -*- coding: utf-8 -*-

import numpy as np
from helpers import *

def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold cross-validation.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def classification_results(y, y_pred):
    
    y = y.reshape(-1) # Linearize
    y_pred = y_pred.reshape(-1) # Linearize
    diff = y - y_pred
    correct = np.sum(diff == 0)
    return correct / len(y_pred)

def iterations(model, Y, X, k_indices, k, patch_size):
    """
    Execute a single run of cross-validation.
    Returns the ratio of correct answers on the validation set.
    """
    non_k_indices = k_indices[np.arange(k_indices.shape[0]) != k].ravel()
    tx_tr = X[non_k_indices]
    y_tr = Y[non_k_indices]
    tx_te = X[k_indices[k]]
    y_te = Y[k_indices[k]]
    
    model.initialize_model() # Reset model
    model.train(y_tr, tx_tr,patch_size)
    
    # Run model
    Z = model.logisticReg(tx_te, patch_size)
    
    # Calculate ground-truth labels
    img_patches_gt = create_patches(y_te, 16)
    y_real = np.mean(img_patches_gt, axis=(1, 2)) > 0.25
    
    return classification_results(y_real, Z)    
    
def cross_validation(model, Y, X, k_fold, seed,patch_size):
    """
    Run a full k-fold cross-validation and print mean accuracy and standard deviation.
    """
    np.random.seed(seed)
    k_indices = build_k_indices(Y, k_fold, seed)
    results = np.zeros(k_fold)
    for k in range(k_fold):
        results[k] = iterations(model, Y, X, k_indices, k, patch_size)
        print('Accuracy: ' + str(results[k]))
    print(results)
    print('Average accuracy: ' + str(np.mean(results)))
   