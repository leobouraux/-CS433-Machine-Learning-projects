# -*- coding: utf-8 -*-
"""
Authors : Clélie de Witasse, David Sollander, Léopold Bouraux
Functions to implement cross validation on the classification models.
"""

import numpy as np
import matplotlib.pyplot as plt

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_visualization(param, accuracy_tr, accuracy_te):
    """visualization the curves of accuracy_tr and accuracy_te."""
    plt.semilogx(param, accuracy_tr, marker=".", color='b', label='train accuracy')
    plt.semilogx(param, accuracy_te, marker=".", color='r', label='test accuracy')
    plt.xlabel("gamma")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")