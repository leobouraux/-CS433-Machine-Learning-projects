import os
import matplotlib.image as mpimg

def load_images(nb_images):
    # Loaded a set of images
    root_dir = "../Datasets/training/"

    image_dir = root_dir + "images/"
    files = os.listdir(image_dir)

    n = min(nb_images, len(files))
    imgs = [mpimg.imread(image_dir + files[i]) for i in range(n)]

    gt_dir = root_dir + "groundtruth/"
    gt_imgs = [mpimg.imread(gt_dir + files[i]) for i in range(n)]
    
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data