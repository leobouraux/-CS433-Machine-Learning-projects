# Project Road Segmentation 

___

***Authors***

• Clélie de Witasse

• Léopold Bouraux

• David Sollander
___

Blabla but du projet

## Requirements

### Python version and libraries
The script have been executed on Python 3, using the libraries listed below. VERIFIER LES VERSIONS
* **keras v2.3.1**: This minimalist open-source library is used to build our neural networks. See the documentation on the [keras](https://keras.io/) website.
* **Tenserflow**: ???
* **scikit-learn v0.21.2**: This open-source machine learning library supports supervised and unsupervised learning, as well as tools for model fitting, data preprocessing, model selection and evaluation. It has been used to train the logistic regression model. See the documentation on the [scikit-learn](https://scikit-learn.org/stable/user_guide.html) website.
* **scikit-image v0.15.0**: This library offers a collection of algorithms for image processing and has been used ??. See the documentation on the [scikit-image](https://scikit-image.org/docs/stable/).
* **Python Imaging Library (PIL) v6.1.0**: This second library has also been used for image processing. You can find a documentation on this [website](https://python.developpez.com/cours/pilhandbook/).


### Data
The datasets are available either from the [EPFL private challenge page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation) or in the folder `Data`on this GitHub.

## How to launch `run.py`
The data must be uploaded in the `Data` folder, where the subfolder `training` contains the satellite images and their groundtruthes and `test_set_images`  contains the test satellite images. In a terminal, the command `python run.py` launch the script and a prediction folder will be created in the root folder.

## Scripts

#### `run.py`:

#### `data_preprocess.py`:
Contains functions to preprocess the data properly:
* **`sorted_aphanumeric`**: 
* **`data_load`**: 
* **`dataGenerator`**:
* **`create_validation_train_directory`**:

#### `data_postprocess.py`:
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

#### `unet.py`:
Contains functions that are used to create the U-Net model:
* **`recall_m`**
* **`precision_m`**
* **`f1_m`**
* **`convolution_down`**
* **`convolution_up`**
* **`unet256`**
* **`unet512`**

#### `logistic_regression.py`:
???

#### `constants.py:
Contains useful constant values that are used in other scripts.


#### `cross_validation.py`:
Contains useful functions to implement cross validation on the segmentation models:
* **`build_k_indices`**: Creates the indices for the k folds
* **`cross_validation_visualization`**: Visualize the result of the cross validation



