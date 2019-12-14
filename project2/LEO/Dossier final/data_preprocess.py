import sys, re, os, shutil
import matplotlib.image as mpimg
import numpy as np
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return np.asarray(sorted(data, key=alphanum_key))

def data_load(directory_name): 
    filenames = sorted_aphanumeric(os.listdir(directory_name))
    imgs = []
    i = 0
    nb_imgs = len(filenames)
    for j, fileName in enumerate(filenames):
        if(fileName[0]=='.'):
            nb_imgs-=1
        else:
            full_name = directory_name+fileName
            img=mpimg.imread(full_name)
            imgr = img_to_array(img)
            imgs.append(imgr)
            sys.stdout.write("\rImage {}/{} is being loaded".format(i+1,nb_imgs))
            sys.stdout.flush()
            i+=1
    print()
    return np.asarray(imgs)

def dataGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, target_size,
                  image_color_mode = "rgb", mask_color_mode = "grayscale",
                  image_save_prefix  = "image",mask_save_prefix  = "mask",
                  flag_multi_class = False,num_class = 2,save_to_dir = None, seed = 1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen  = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    
    for (img,mask) in train_generator:
        if(np.max(img) > 1):
            img = img / 255
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        yield (img,mask)
        
def create_validation_train_directory(path, dir_images, dir_labels, i, seed):
    # 0<=i<5 
    origin_dirs = [dir_images, dir_labels]
    for name in origin_dirs:
        filenames = sorted_aphanumeric(os.listdir(path+name))
        if('.DS_Store' in filenames):
            filenames = np.delete(filenames, 0)
        np.random.seed(seed)
        permut = np.random.permutation(len(filenames))
        index = int(len(permut)*0.2)
        test_ind = permut[i*index:(i+1)*index]
        test_filenames = filenames[test_ind]
        trai_ind = permut[index:]
        trai_filenames = filenames[trai_ind]
        new = [name+'_te', name+'_tr']
        for n in new:  
            if os.path.exists(path+n):
                shutil.rmtree(path+n)
            os.mkdir(path+n)
        for test_f in test_filenames:
            img = Image.open(path+name+'/'+test_f)
            img.save(path+new[0]+'/'+test_f)
        for trai_f in trai_filenames:
            img = Image.open(path+name+'/'+trai_f)
            img.save(path+new[1]+'/'+trai_f)