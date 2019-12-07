# -*- coding: utf-8 -*-
"""
Authors : Clélie de Witasse, David Sollander, Léopold Bouraux
"""

from proj1_helpers import *
from implementations import *
from preprocess import *

# Download training dataset and testing sataset
data_path_train     = './data/train.csv'
yTrain, xTrain, ids = load_csv_data(data_path_train)


# Parameters
seed       = 3
degree     = 7
trainRatio = 0.72
lambda_    = 2e-4

# Classification method on the training set
pyTrain, pxTrain            = preprocess_data(yTrain, xTrain)
xTr, xTe, yTr, yTe = split_data(pxTrain, pyTrain, trainRatio)

xPolyTrain         = build_poly(xTr, degree)
xPolyTest          = build_poly(xTe, degree)

loss, w            = ridge_regression(yTr, xPolyTrain, lambda_)

y_validation_tr = predict_labels(w, xPolyTrain)
accuracy_tr = sum(y_validation_tr == yTr)/len(yTr)
print('Accuracy for LS (S train):', accuracy_tr)

y_validation_te = predict_labels(w, xPolyTest)
accuracy_te = sum(y_validation_te == yTe)/len(yTe)
print('Accuracy for LS (S test):', accuracy_te)
