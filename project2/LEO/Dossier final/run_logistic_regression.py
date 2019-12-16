omport numpy as np
from logistic_regression import *
from logistic_reg_preprocess import *

# --- Useful constants
k_cluster         = 5     # Nb of cluster
max_iter_cluster  = 100    # Max nb of iterations while clustering
threshold_cluster = 1e-4  # Threshold fo clustering

# --- Load the training set
nb_images = 100
image_dir = "training/images/"
groudtruth_dir = "training/groundtruth/"

imgs    = load_images(image_dir, nb_images)
gt_imgs = load_images(groundtruth_dir, nb_images)

# --- Preprocess the images
imgs_med = median(imgs,3,n)
img_pre = np.array([preprocess_image_rgb(imgs_med[i]) for i in range(n)])

img_cluster = np.array([kmean_compression(imgs_med[i], img_pre[i], k_cluster, max_iter_cluster, threshold_cluster) for i in range(n)])

# --- Load the logistic regression model
model = LogisticModel()

# --- Cross validation 
np.random.seed(1) 
cross_validation(model, gt_imgs, img_cluster, 4, 1,16)