from constants import *
from cross_validation import *
from data_preprocess import *
from data_postprocess import *
from unet import *
from fit_model import *

import os
import sys

# --- Choose whether train a new model (1) or load a pretrained model
train_model = 1

# --- Indicate the path of the pretrained model (if it exists) and the images to load
#WEIGHT_PATH     =  # Indicate the path where yout weights are pretrained
DATA_PATH       = root_path+"data/"
DATA_TEST_PATH  = root_path+"data/test/"
MODEL_PATH      = root_path+"models/"
PRED_PATH       = DATA_PATH+"predictions/"

# --- Cross validation
k_fold = 5
rotation_angles = [5,90,180]
for k in range(k_fold):
    
    # Creates a list to store the predictions 
    IMGS = []

    # Create the k folds
    create_validation_train_directory(DATA_PATH+'train/', 'images', 'groundtruth', 1, 1)
    
    for i in rotation_angles:
        # --- Train or load a pretrained model
        if train_model:
            MODEL_NAME = f'Model_k{k}_rot{i}'
            model = fit_unet(i, MODEL_PATH, MODEL_NAME)

        else:
            # Load pretrained weights
            model = unet256(input_size = (SIDE,SIDE,3), verbose=False)
            model.load_weights(WEIGHT_PATH)

        # --- Load the testing set
        data_test = data_load(DATA_PATH+'test_resized/')

        test_datagenerator  = ImageDataGenerator()
        testGene            = test_datagenerator.flow(data_test, batch_size=1)

        # --- Predict the the segmentation of the testing images
        # Reshape the images to 256x256 px and predict the segmentation
        reshape_img(DATA_PATH+"test/", DATA_PATH+"test_resized/", SIDE)
        results = model.predict(data_test,verbose=1)

        # Create a prediction folder if it doesn't exist
        if not os.path.exists(PRED_PATH):
                os.mkdir(PRED_PATH)

        # Save the predictions
        PRED_FILE = PRED_PATH + f'Model_k{k}_rot{i}/'
        savePredictedImages(DATA_PATH+"test_resized/", 
                            PRED_FILE, 
                            results, 
                            concat=False)

        # Resize the prediction to their final size (608x608 px)
        reshape_img(PRED_FILE, PRED_FILE, SIDE_FINAL)

        # Add the reshaped prediction to the prediction list
        IMGS.append((1,data_load_for_prediction(PRED_FILE)))
        
# --- Compute the mean of the predicted images
means = average_image(IMGS)

# --- Create submission in a .csv file
if not os.path.exists(PATH_ROOT+"submissions/"):
        os.mkdir(PATH_ROOT+"submissions/")
create_csv_submission(patched_imgs, vs, PATH_ROOT+'submissions/' + f'Model_k{k}_rot{i}.csv')