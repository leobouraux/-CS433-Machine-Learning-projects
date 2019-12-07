# -*- coding: utf-8 -*-
"""
Authors : Clélie de Witasse, David Sollander, Léopold Bouraux
Implementation of functions to preprocess dataset, split datasets, build a polynomial basis on dataset
and create a batch iteration for stochastic methods.
"""

import numpy as np

def standardize(x):
    """Standardize the dataset x"""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def preprocess_data(y, tX0):
    MISSING_DATA = -999
    MISSING_DATA2 = 0
    tX1 = np.delete(tX0, 22, axis=1)
    
    # replace missing data for all columns
    x=tX1
    replacement_values_by_col = np.ma.array(x, mask=[x==MISSING_DATA]).mean(axis=0)
    replacement_values = np.tile(replacement_values_by_col, (len(x), 1))
    x2= x.copy()
    x2[x == MISSING_DATA] = replacement_values[x == MISSING_DATA]
    
    # replace missing data for last column
    last = tX1[:,-1]
    rvbc = np.ma.array(last, mask=[last==MISSING_DATA2]).mean(axis=0)
    rv = np.tile(rvbc, (len(last), 1))
    last2 = last.copy()
    last2[last == MISSING_DATA2] = rv[last== MISSING_DATA2].flatten()
    x2[:, -1] = last2
    
    standardized_x, _, _ = standardize(x2)

    return y, standardized_x

def split_data(x, y, train_ratio, seed=3):
    """Split the inputs x and y into 4 small dataset x_tr, x_te, y_tr, y_te"""
    np.random.seed(seed)
    indices = np.random.permutation(len(x))
    end = int(train_ratio*len(x))
    x_tr = x[indices[:end]]
    x_te = x[indices[end:]]
    y_tr = y[indices[:end]]
    y_te = y[indices[end:]]
    return x_tr, x_te, y_tr, y_te

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly_matrix = np.ones(len(x))
    for n in range(1, degree+1):
        poly_matrix = np.c_[poly_matrix, x**n] 
    return poly_matrix

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Creates batch for stochastic methods."""
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y      = y[shuffle_indices]
        shuffled_tx     = tx[shuffle_indices]
    else:
        shuffled_y  = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index   = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]