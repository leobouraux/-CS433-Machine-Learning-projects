from constants import *
from cross_validation import *
from data_preprocess import *
from data_postprocess import *
from unet import *
from fit_model import *

import os,sys


# --- Dataset augmentation
dataset_augmentation(root_path+"data/train/images/", root_path+"data/train/img_aug/")
dataset_augmentation(root_path+"data/train/groundtruth/", root_path+"data/train/gts_aug/")


# --- Creates a list to store the predictions 
IMGS = []

# --- Cross validation
k_fold = 5
rotation_angles = [5,90]
for k in range(k_fold):

    # Create the k folds
    create_validation_train_directory(DATA_PATH+'train/', 'img_aug', 'gts_aug', 1, 1)
    
    for i in rotation_angles:
        # --- Train or load a pretrained model
        MODEL_NAME = f'Model_k{k}_rot{i}'
        model = fit_unet(i, MODEL_PATH, MODEL_NAME)
	
	# Uncomment if you want to save your model for a further use
	#save_model(model, MODEL_PATH, MODEL_NAME)

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