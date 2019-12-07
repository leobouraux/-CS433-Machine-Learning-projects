# ML-Project-1: 

___

***Authors***

• Léopold Bouraux

• David Sollander

• Clélie de Witasse

___


## How to launch `run.py`

The `train.csv` and `test.csv` files must be added to a `data` folder at the root of the project. In a terminal, the command `python run.py` launch the script and a CSV file will be created in the same folder.

## Scripts

#### `run.py`

This script creates the CSV file submitted AIcrowd. It preprocesses the datta, learns a classification model, apply this model to the dataset `test.csv` and creates the submission file.
4 parameters can be changed here:
* the seed for the reproducibility (default = 3)
* the degree of the polynomial features (default = 7)
* the ratio of the dataset which will be used for training (default = 0.72)
* the tuning parameter for Ridge regression (default = 2e-4)

#### `preprocess.py`

Contains function to preprocess the data properly:
* **`least_squares_GD`**: Linear regression using gradient descent


#### `implementations.py` 

Contains the six regression methods needed for the project and useful functions to compute losses and gradients,

* **`standardize`**: Standardize the data
* **`preprocess_data`**: Remove all missing values and standardize the data
* **`split_data`**: Split the data into a training set and a testing set given a ration
* **`build_poly`**: Build polynomial bases for the features
* **`batch_iter`**: Creates batch for stochastic classification methods

#### `proj1_helpers.py` 

Contains given helper functions,

* **`load_csv_data`**: Loads data
* **`predict_labels`**: Generates class predictions
* **`create_csv_submission`**: Creates an output file in CSV format for submission

#### `cross_validation.py`

Contains useful functions to implement cross validation on the classification models, 
* **`build_k_indices`**: Creates the indices for the k folds
* **`cross_validation_visualization`**: Visualize the result of the cross validation



