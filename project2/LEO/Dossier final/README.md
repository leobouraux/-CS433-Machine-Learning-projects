# Project Road Segmentation 

___

***Authors***

• Léopold Bouraux

• David Sollander

• Clélie de Witasse

___

## Requirements

### Python

### Libraries

### Data


## How to launch `run.py`

The data must be uploaded in the `Data`folder, where the subfolder `training` contains the satellite images and their groundtruthes and `test_set_images`  contains the test satellite images. In a terminal, the command `python run.py` launch the script and a prediction folder will be created in the root folder.

## Scripts

#### `run.py`

#### `data_preprocess.py`
Contains functions to preprocess the data properly:
* **`sorted_aphanumeric`**: 
* **`data_load`**: 
* **`dataGenerator`**:
* **`create_validation_train_directory`**:

#### `data_postprocess.py`
Contains functions to postprocess the data properly:
* **`save_model`**:
* **`load_model`**:
* **`reshape_img`**:
* **`img_float_to_uint8`**
* **`concatenate_images`**
* **`savePredictedImages`**
* **`data_load_for_prediction`**
* **`average_image`**
* **`color_patch`**
* **`img_crop`**
* **`patched_imgs_and_vs`**
* **`create_csv_submission`**

#### `unet.py` 
Contains functions that are used to create the U-Net model:
* **`recall_m`**
* **`precision_m`**
* **`f1_m`**
* **`convolution_down`**
* **`convolution_up`**
* **`unet256`**
* **`unet512`**

#### `logistic_regression.py`
???

#### `constants.py
Contains useful constant values that are used in other scripts.


#### `cross_validation.py`
Contains useful functions to implement cross validation on the segmentation models:
* **`build_k_indices`**: Creates the indices for the k folds
* **`cross_validation_visualization`**: Visualize the result of the cross validation



