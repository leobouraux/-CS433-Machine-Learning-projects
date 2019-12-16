from constants import *
from cross_validation import *
from data_preprocess import *
from data_postprocess import *
from unet import *

import os
import sys

# --- Choose whether train a new model (1) or load a pretrained model
train_model = 0

# --- Indicate the path of the pretrained model (if it exists) and the images to load
WEIGHT_PATH     = "../" # A remplir
DATA_PATH       = "../../Data/training"
DATA_TEST_PATH  = "!!"

create_validation_train_directory(DATA_ROOT+'train/', 'images', 'groundtruth', 1, 1)

# --- Train or load a pretrained model
if train_model:
    # Load dataset
    data_gen_args = dict(rotation_range=180,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    train_generator = dataGenerator(2, DATA_PATH+'train',
                                    'images_tr',
                                    'groundtruth_tr'
                                    ,data_gen_args,
                                    (SIDE,SIDE))

    validation_generator = dataGenerator(2, DATA_PATH+'train',
                                    'images_te',
                                    'groundtruth_te'
                                    ,data_gen_args,
                                    (SIDE,SIDE))

    filepath = "weights.{epoch:02d}-{val_f1_m:.2f}.hdf5"

    csv_logger = CSVLogger("AccuracyHistory.csv")
    cp_callback = ModelCheckpoint(filepath=filepath, verbose=1, save_weights_only=True, period=1)
    
    # Load and fit the model
    model = unet256((SIDE,SIDE,3),lr=0.001, verbose=False)
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=200,#2000
                    epochs=100, #10
                    verbose=1,
                    validation_data = validation_generator,
                    validation_steps = 70,#700
                    validation_freq=1,
                    initial_epoch=0,
                    callbacks=[cp_callback, csv_logger])
    
    # Save the trained model
    save_model(model, PATH_ROOT+"model_saved", "Model_UNET_256_k_fold_0_f1_874")

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
if not os.path.exists(DATA_PATH+"predictions/"):
        os.mkdir(DATA_PATH+"predictions/")
            
# Save the predictions
savePredictedImages(DATA_PATH+"test_resized/", 
                    DATA_PATH+"predictions/Model_UNET_256_k_fold_0_f1_874a", 
                    results, 
                    concat=False)
savePredictedImages(DATA_PATH+"test_resized/", 
                    DATA_PATH+"predictions/Model_UNET_256_k_fold_0_f1_874b", 
                    results, 
                    concat=True)


# Resize the prediction to their final size (608x608 px)
reshape_img(DATA_PATH+"predictions/Model_UNET_256_k_fold_0_f1_874a/",
            DATA_PATH+"predictions/Model_UNET_256_k_fold_0_f1_874a/",
            SIDE_FINAL)

# Average the predictions 
IMGS = []
IMGS.append(data_load_for_prediction(DATA_ROOT+"predictions/Model_UNET_256_k_fold_0_f1_874a/"))
means = average_image(IMGS)

# --- Create submission in a .csv file
if not os.path.exists(PATH_ROOT+"submissions/"):
        os.mkdir(PATH_ROOT+"submissions/")
create_csv_submission(patched_imgs, vs, PATH_ROOT+'submissions/Model_UNET_256_k_fold_0_f1_874a.csv')