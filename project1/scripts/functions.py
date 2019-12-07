import numpy as np
from proj1_helpers import *

###################
## Least squares ##
###################

def least_squares(y, tx):
    N = y.shape[0]
    a = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(a, b)
    e = y - tx@w
    loss = e.T@e / (2*N)
    return loss, w

####################################
## Least squares Gradient Descent ##    
####################################
                        
def least_squares_GD(y, tx, w0, max_iters, gamma):
    w = w0
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = np.sqrt(2* compute_mse(y, tx, w))
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}, w={w}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w=w[0]))
    return loss, w
                
###############################################
## Least squares Stochastic Gradient Descent ##
###############################################
            
def least_squares_SGD(y, tx, w0, max_iters, gamma):
    batch_size = 1
    w = w0
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = np.sqrt(2*compute_mse(y, tx, w))
        print("SGD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return loss, w
    
######################
## Ridge regression ##
######################

def ridge_regression(y, tx, lambda_):
    a = tx.T@tx +  (2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1]))
    b = tx.T@y
    w = np.linalg.solve(a, b)
    e = y - tx@w
    loss = e.T@e / (2*len(e))
    return loss, w

#########################
## Logistic regression ##
#########################

def log_reg(y, tx, w0, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = w0
    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_loss_loglike(y, tx, w)
        grad = compute_gradient_loglike(y, tx, w)
        w = - grad * gamma    
        # log info
        #if iter % 100 == 0:
           # print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            return loss[-1], w

#####################################
## Regularized logistic regression ##
#####################################

def reg_log_reg(y, tx, lambda_, w0, max_iters, gamma):
    threshold = 1e-5
    loss = []
    w = w0
    for iter in range(max_iters):
        # get loss and update w.
        A = sigmoid(tx@w)
        loss = compute_loss_loglike(y, tx, w) + lambda_*w.T@w
        gradient = compute_gradient_loglike(y, tx, w) + 2*lambda_*w
        print(gradient)
        w = w - gradient * gamma
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            return loss[-1], w