# data_postprocess
import os, re, sys, csv
import skimage.io as io
import numpy as np
import matplotlib.image as mpimg
from skimage import img_as_ubyte
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array


def save_model(model, save_path, output_filename):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model_json = model.to_json()
    with open(save_path+output_filename + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(save_path+output_filename + ".h5")
    print("Saved model to disk")

def load_model(path):
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model

def reshape_img(path_from, path_to, shape):
    if not os.path.exists(path_to):
        os.mkdir(path_to)
    filenames = os.listdir(path_from)
    for i, fileNb in enumerate(filenames):
        if(fileNb!='.DS_Store'):        
            im1 = Image.open(path_from+fileNb)
            # use one of these filter options to resize the image
            im2 = im1.resize((shape, shape), Image.NEAREST)
            im2.save(path_to+fileNb)
            
def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def savePredictedImages(from_path, save_path, predictions, concat=True):
    #if os.path.exists(save_path):
    #    shutil.rmtree(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    list_im_name = sorted_aphanumeric(os.listdir(from_path))
    if(list_im_name[0]=='.DS_Store'):
        list_im_name=list_im_name[1:]
    for name, pred in zip(list_im_name, predictions):
        img = mpimg.imread(from_path+name)
        if(concat):
            if(len(pred.shape)==3 and pred.shape[2]==1):
                pred = pred.reshape(pred.shape[0],pred.shape[1])
            cimg = concatenate_images(img, pred)
            io.imsave(save_path+"/prediction"+name[4:], img_as_ubyte(cimg))
        else:
            io.imsave(save_path+"/prediction"+name[4:], img_as_ubyte(pred))
    print("save pred ok")
            
def data_load_for_prediction(directory_name, RGB_image=False): 
    filenames = sorted_aphanumeric(os.listdir(directory_name))
    imgs = []
    i=0
    nb_imgs = len(filenames)
    for fileNb in filenames:
        if(fileNb!='.DS_Store'):  
            full_name = directory_name+fileNb
            img=mpimg.imread(full_name)
            if RGB_image:
                imgr = img_to_array(img)
            else:
                imgr = img_to_array(img).reshape(SIDE_FINAL, SIDE_FINAL)
            imgs.append(imgr)
            i+=1
            sys.stdout.write("\rImage {}/{} is being loaded".format(i,nb_imgs))
            sys.stdout.flush()
        else:
            nb_imgs-=1
    print()

    return np.asarray(imgs)

def average_image(IMGS_weighted_folders):
    total = IMGS_weighted_folders[0][0]
    means = IMGS_weighted_folders[0][1] * total
    print('Size should be (50, 608, 608) and currently is:', means.shape) 
    for i in range(1,len(IMGS_weighted_folders)):
        tupl = IMGS_weighted_folders[i]
        weight = tupl[0]
        total+=weight
        for j, img in enumerate(tupl[1]):
            means[j]+=img*weight
    return means/total

def median_image(IMGS_weighted_folders):
    images = []
    for i in range(len(IMGS_weighted_folders)):
        images.append(IMGS_weighted_folders[i][1])
    #return np.median(np.asarray(images), axis=0)
    return np.percentile(np.asarray(images), 70, axis=0)

def color_patch(patch, thresh=0.25):
    m = np.mean(patch)
    if(m>thresh):
        return 1, np.ones(16*16).reshape(16,16)
    else:
        return 0, np.zeros(16*16).reshape(16,16)

def color_patch_full(img):
    vs=[]
    patchs=[]
    img_patches = img_crop(img, PATCH_SIZE, PATCH_SIZE)
    for i in range(len(img_patches)):
        v, X = color_patch(img_patches[i])
        vs.append(v)
        patchs.append(X)
    patchs = np.asarray(patchs)
    return vs, patchs

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def patched_imgs_and_vs(means):
    vs_for_each_imgs = []
    imgs_means_patched = []
    for index, mean_img in enumerate(means):
        vs, patchs = color_patch_full(mean_img)
        vs_for_each_imgs.append(vs)
        
        nb_patch_per_side = int(SIDE_FINAL/PATCH_SIZE)
        patchs = patchs.reshape(nb_patch_per_side, nb_patch_per_side, PATCH_SIZE, PATCH_SIZE)
        
        for j in range(0,nb_patch_per_side):
            tmp = patchs[j,0]
            for i in range(1,nb_patch_per_side):
                tmp = np.concatenate((tmp, patchs[j,i]), axis=1) 
            if(j==0):
                TMP = tmp
            else:
                TMP = np.concatenate((TMP, tmp), axis=0)
        imgs_means_patched.append(np.rot90(np.rot90(np.rot90(np.flip(TMP, 0)))))
        sys.stdout.write("\rImage {}/{} is being processed".format(index+1,len(means)))
        sys.stdout.flush()
    return imgs_means_patched, vs_for_each_imgs

def create_csv_submission(imgs, vs_for_each_imgs, name):
    labels = np.arange(0, SIDE_FINAL, PATCH_SIZE)
    nb_patch_per_side = int(SIDE_FINAL/PATCH_SIZE)
    with open(name, 'w') as csvfile:
        fieldnames = ['id', 'prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for nb, img in enumerate(imgs):
            ID = str(nb+1).zfill(3)
            row = 0
            col = 0
            for v in vs_for_each_imgs[nb]:
                col_ = col%nb_patch_per_side
                ID2 = ID+'_'+str(labels[row])+'_'+str(labels[col_])    
                writer.writerow({'id':ID2,'prediction':v})
                if(col == 37):
                    col=0
                    row+=1
                else:
                    col+=1