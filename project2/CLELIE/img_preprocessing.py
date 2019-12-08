import os
import sys
import random
import matplotlib.image as mpimg
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img


from constant_values import *

def data_augmentation(directory_name, seed): 
    datagen = ImageDataGenerator()
    filenames = os.listdir(directory_name)
    
    #create 24 rotated images for one image
    angls = np.arange(0, 360, 15)
    zooms = np.array([1., 0.85, 0.8, 0.75, 0.8, 0.85,
                      1., 0.85, 0.8, 0.75, 0.8, 0.85,
                      1., 0.85, 0.8, 0.75, 0.8, 0.85,
                      1., 0.85, 0.8, 0.75, 0.8, 0.85])
    imgs = []
    
    for i, fileNb in enumerate(filenames):
        full_name = directory_name+fileNb
        img=mpimg.imread(full_name)
        imgr = img_to_array(img)
        for j, angle in enumerate(angls):
            zoom = zooms[j]
            img2 = datagen.apply_transform(x=imgr, transform_parameters={'theta':angle, 'zx':zoom, 'zy':zoom})
            imgs.append(img2)
        sys.stdout.write("\rImage {}/{} is being loaded".format(i+1,len(filenames)))
        sys.stdout.flush()
        
    print(' ... Shuffle data ...')
    imgs1 = np.asarray(imgs)
    np.random.seed(seed)
    rand = np.random.randint(imgs1.shape[0], size=imgs1.shape[0])
    return imgs1[rand, :, :, :]

#Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.5
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):    
            im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


# Extract label images
def extract_labels(gt_imgs):
    """Extract the labels into a 1-hot matrix [image index, label index]."""            
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(gt_imgs.shape[0])]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)

    
def create_patches(im):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    for i in range(0, imgheight, IMG_PATCH_SIZE):
        for j in range(0, imgwidth, IMG_PATCH_SIZE):
            im_patch = im[j:j+IMG_PATCH_SIZE, i:i+IMG_PATCH_SIZE, :]
            list_patches.append(im_patch)
    return list_patches


def extract_data(tr_imgs):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """#todo
        
    #list of images (=list windows (=list pixels))
    img_patches = [create_patches(tr_imgs[i]) for i in range(tr_imgs.shape[0])]
    
    # j in range number of patch per image
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data)

