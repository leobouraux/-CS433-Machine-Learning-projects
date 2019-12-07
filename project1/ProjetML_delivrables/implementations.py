# -*- coding: utf-8 -*-
"""
Authors : Clélie de Witasse, David Sollander, Léopold Bouraux
Implementation of the 6 basic method implementations.
"""
import numpy as np
from proj1_helpers import *

##### Loss computation
def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_loss(y, tx, w):
    """Calculate the mean square error."""
    return np.sqrt(2*calculate_mse(y-tx@w))

def sigmoid(t):
    """Calculate the sigmoid function (for logistic regressions)."""
    return 1. / (1. + np.exp(-t))

def compute_loss_loglike(y, tx, w, lambda_):
    A    = sigmoid(tx@w)
    loss = y.T@(np.log(A)) + (1-y).T@np.log(1-A)
    return np.sum(np.log(1+np.exp(tx.dot(w))) - y*tx.dot(w)) + \
        lambda_ * w[np.newaxis, :].dot(w[:, np.newaxis])[0, 0]

##### Gradient computation
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error    = y-(tx@w)
    gradient = -1/len(error) * tx.T@error
    return gradient, error

def compute_gradient_loglike(y, tx, w, lambda_):
    """Compute the gradient for logistic regressions."""
    A    = sigmoid(tx@w)
    grad = tx.T@(A-y)
    return grad + lambda_ / y.shape[0] * w

##### Method 1 : Least squares gradient descent
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using Least squares"""   
    w = initial_w
    for n_iter in range(max_iters):
        grad, _ = compute_gradient(y,tx,w)    # Compute gradient
        loss    = compute_loss(y, tx, w)      # Compute loss
        w      -= gamma * grad                # Update w by gradient
    return loss, w

##### Method 2 : Least squares stochastic gradient descent
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad, _ = compute_gradient(y, tx, w)    # Compute a stochastic gradient
            loss    = compute_loss(y, tx, w)        # Compute a stochastic loss
            w      -= gamma * grad                  # Update w 
    return loss, w

##### Method 3 : Least squares
def least_squares(y, tx):
    """calculate the least squares solution."""
    A    = tx.T@tx
    B    = tx.T@y
    w    = np.linalg.solve(A,B)
    loss = compute_loss(y, tx, w)
    return loss, w

##### Method 4 : Ridge regression
def ridge_regression(y, tx, lambda_):
    """Implement ridge regression."""
    A    = tx.T@tx +  (2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1]))
    B    = tx.T@y
    w    = np.linalg.solve(A, B)
    loss = compute_loss(y, tx, w)
    return loss, w

##### Method 5 : Logistic regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Implement logistic regression."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_loglike(y, tx, w, lambda_=0)  # Compute gradient
        loss = compute_loss_loglike(y, tx, w, lambda_=0)      # Compute loss
        w   -= gamma * grad                                   # Update w                     
    return loss, w

##### Method 6 : Regularized logistic regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Implement regularized logistic regression."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_loglike(y, tx, w, lambda_)   # Compute gradient
        loss = compute_loss_loglike(y, tx, w, lambda_)       # Compute loss
        w   -= gamma * grad                                  # Update w                     
    return loss, w


