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

data_path_test     = './data/test.csv'
_ , xTest, ids_test = load_csv_data(data_path_test)

# Parameters
seed       = 3
degree     = 7
trainRatio = 0.72
lambda_    = 2e-4

# Classification method on the training set
pyTrain, pxTrain   = preprocess_data(yTrain, xTrain)
xTr, xTe, yTr, yTe = split_data(pxTrain, pyTrain, trainRatio)

xPolyTrain         = build_poly(xTr, degree)
xPolyTest          = build_poly(xTe, degree)

loss, w            = ridge_regression(yTr, xPolyTrain, lambda_)

# Apply the learnt model on the testing set xTest
pxTest  = preprocess_data(_, xTest)
xPoly   = build_poly(pxTest, degree)
yPred   = predict_labels(w, xPoly)
create_csv_submission(ids_test, yPred, "submission.csv")
