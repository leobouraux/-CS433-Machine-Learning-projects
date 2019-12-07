# -*- coding: utf-8 -*-
"""
Authors : Clélie de Witasse, David Sollander, Léo Bouraux
"""

from proj1_helpers import *
from implementations import *
from preprocess import *

# Download training dataset and testing sataset
data_path_train     = '../data/train.csv'
yTrain, xTrain, ids = load_csv_data(data_path_train)

data_path_test     = '../data/test.csv'
_, xTest, ids_test = load_csv_data(data_path)

# Parameters
seed       = 3
degree     = 12
trainRatio = 0.45

# Classification method on the training set
pxTrain            = preprocessing(xTrain)
xTr, xTe, yTr, yTe = split_data(pxTrain, yTrain, trainRatio, seed)

xPolyTrain         = build_poly(xTr, degree)
xPolyTest          = build_poly(xTe, degree)

loss, w            = least_squares(yTr, xPolyTrain)

# Apply the learnt model on the testing set xTest
pxTest  = preprocessing(xTest)
xPoly   = build_poly(pxTest, degree)
yPred   = predict_labels(w, xPoly)
create_csv_submission(ids_test, yPred, "submission.csv")
