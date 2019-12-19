# Project Road Segmentation 

___
***Authors***

• Clélie de Witasse

• Léopold Bouraux

• David Sollander
___

The aim of this project is to segment satellite images from Google Maps into a binary outputs, where white pixels represent the roads and black pixels represent the background. First, check that you have already installed the external libraries lised below and that their versions match. The datasets are available in the `Data` folder, the python scripts are in the `src` folder. The architecture of the code is presented in this README file. Different methods of segmentations have been coded, and their results are shown on Fig 1.

![alt-text-1](Pictures/test_10.png "Satellite image") ![alt-text-2](Pictures/pred_10.png "U-Net prediction") ![alt-text-3](Pictures/patched_10.png "Patched prediction for submission")


## Requirements

### Python version and libraries
The script have been executed on Python 3, using the libraries listed below. To install them, you can follow the following instructions.

`pip install --upgrade pip`

`pip install tensorflow`

`pip install keras`

`pip install sklearn` 

`pip install scikit-image` 

`pip install Pillow`

* **TenserFlow v1.15.0**: This open-source platform for machine learning must be installed to use properly the `keras` library. 

See the documentation on the [TenserFlow](https://www.tensorflow.org/overview) website.

* **keras v2.3.1**: This minimalist open-source library is used to build our neural networks. 

See the documentation on the [keras](https://keras.io/) website.
 
* **scikit-learn v0.21.2**: This open-source machine learning library supports supervised and unsupervised learning, as well as tools for model fitting, data preprocessing, model selection and evaluation. It has been used to train the logistic regression model. 

See the documentation on the [scikit-learn](https://scikit-learn.org/stable/user_guide.html) website.
* **scikit-image v0.15.0**: This library offers a collection of algorithms for image processing. 

See the documentation on the [scikit-image](https://scikit-image.org/docs/stable/).
* **Python Imaging Library (PIL) v6.1.0**: This second library has also been used for image processing.
 
You can find a documentation on this [website](https://python.developpez.com/cours/pilhandbook/).


### Data
The datasets are available either from the [EPFL private challenge page](https://www.crowdai.org/challenges/epfl-ml-road-segmentation) or in the folder `Data`on this GitHub.

## How to launch `run.py`
The data must be uploaded in the `Data` folder, where the subfolder `training` contains the satellite images and their groundtruthes and `test_set_images`  contains the test satellite images. In a terminal, the command `python run.py` launch the script and a prediction folder will be created in the root folder. Please make sure that all the constants and the different paths are well defined.

## Scripts

#### `run.py`:
This script run the overall U-Net segmentation. You can either choose to train a model or load pretrained weights by choosing the value of `train_model`. A 5-fold cross-validation will be executed. In each cross-validation iteration, 2 predictions will be computed for each image of the dataset.

* In the first situation, images are rotated by a random angle inferior to 5°, to handle with vertical and horizontal roads.
* In the second situation, images are rotated by a random angle inferior to 90°, to handle with slanting roads.

The mean of those predictions will give the final prediction for each image.
Eventually, a submission is created for the AI-crowd challenge. The predictions are patched to respect the criterion of the challenge, but the results seems better without any patch.

#### `data_preprocess.py`:
Contains functions to preprocess the data properly:
* **`sorted_aphanumeric`**: to be sure to load images in the right order.
* **`data_load`**: loads the png images from a specified directory and returns them as a numpy array.
* **`dataGenerator`**: creates a generator of training images and groundtruthes.
* **`dataset_augmentation`**: copies the entire dataset in a new folder and creates a rotated copy (by 45°) for each image in the dataset.
* **`create_validation_train_directory`**: splits the dataset into a validation set and a training set to implement a cross validation

#### `data_postprocess.py`:
Contains functions to postprocess the data properly:
* **`save_model`**: saves a trained model to a json file and saves the weights in a h5 file as well.
* **`load_model`**: loads a saved neural net model from a json file.
* **`reshape_img`**: reshapes images from a directory and puts the reshaped images into a new folder.
* **`img_float_to_uint8`**: converts an image from float values to uint8 values.
* **`concatenate_images`**: concatenates two images. To be used to show an image and its predicted groundtruth.
* **`savePredictedImages`**: saves predicted images into a specified folder.
* **`data_load_for_prediction`**: loads the groundtruth images that have been predicted. To be used to average the predictions.
* **`average_image`**: averages a bunch of predictions.
* **`median_image`**: computes the median of a bunch of predictions.
* **`color_patch`**: colors a patch in black or white depending on its mean.
* **`color_patch_full`**: creates a binary patched image.
* **`img_crop`**: crops an image.
* **`patched_imgs_and_vs`**: creates patched images from a list of predictions, also returns the predictions to be submitted.
* **`create_csv_submission`**: creates a csv file for submission on AI-Crowd.

#### `unet.py`:
Contains functions that are used to create the U-Net model:
* **`recall_m`**: to be used to compute the f1-score.
* **`precision_m`**: to be used to compute the f1-score.
* **`f1_m`**: computes the F1-score to be displayed at each epochs of the fit of neural models.
* **`convolution_down`**: defines a set of convolutional layers and a maxPooling operation. Is to be used in the `unet256` and `unet512` functions. 
* **`convolution_up`**: defines a set of convolutional layers and merge different layers of the model. Is to be used in the `unet256` and `unet512` functions. 
* **`unet256`**: builds a U-Net model that is tu be used with 256x256 images.
* **`unet512`**: builds a U-Net model that is tu be used with 512x512 images.

#### `fit_model.py`:
Contains a function **`fit_unet`** that fits a U-Net model.

#### `constants.py`:
Contains useful constant values and paths that are used in other scripts.

#### `logistic_regression.py`:
* **`initialize_model`** : initializes or reset the logistic model from scikit-learn.
* **`extract_features_2d`** : extracts the average color, the standard deviation and the variation from each image patch.
* **`polynomial_feature`** : generates a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree.
* **`logisticReg`** : generates predictions using the logistic regression model. 

#### `cross_validation.py`:
Contains useful functions to implement cross validation on the segmentation models:
* **`build_k_indices`**: Creates the indices for the k folds.
* **`classification_results`** : Gives the accuracy of the k fold.
* **`iterations`** : trains the model and returns the prediction.
* **`cross_validation`** : Runs a full k-fold cross-validation and print mean accuracy and standard deviation.

