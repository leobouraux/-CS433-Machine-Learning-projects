import numpy as np
from helpers_model import *

#pad an image 
def pad_image(img, padSize):
    is_2d = len(img.shape) < 3
    if is_2d:
        return np.lib.pad(img,((padSize,padSize),(padSize,padSize)),'reflect')
    else:
        return np.lib.pad(img,((padSize,padSize),(padSize,padSize),(0,0)),'reflect')
# source for rotation
#https://stackoverflow.com/questions/46657423/rotated-image-coordinates-after-scipy-ndimage-interpolation-rotate
def rot(image, xy, angle):
    im_rot = rotate(image,angle, reshape=False) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot

def image_generator(images, ground_truths, window_size, batch_size = 64, upsample=False):
    np.random.seed(0)
    imgWidth = images[0].shape[0]
    imgHeight = images[0].shape[1]
    half_patch = IMG_PATCH_SIZE // 2
    
    padSize = (window_size - IMG_PATCH_SIZE) // 2
    paddedImages = []
    for image in images:
        paddedImages.append(pad_image(image,padSize))
        
    while True:
        batch_input = []
        batch_output = [] 
        
        #rotates the whole batch for better performance
        randomIndex = np.random.randint(0, len(images))  
        img = paddedImages[randomIndex]
        gt = ground_truths[randomIndex]
        
        # rotate with probability 10 / 100
        random_rotation = 0
        if (np.random.randint(0, 100) < 10):
            rotations = [90, 180, 270, 45, 135, 225, 315]
            random_rotation = np.random.randint(0, 7)
            img = rot(img, np.array([imgWidth+2*padSize, imgHeight+2*padSize]), rotations[random_rotation])
            gt = rot(gt, np.array([imgWidth, imgHeight]), rotations[random_rotation]) 
        
        background_count = 0
        road_count = 0
        while len(batch_input) < batch_size:
            x = np.empty((window_size, window_size, 3))
            y = np.empty((window_size, window_size, 3))
            
            
            # we need to limit possible centers to avoid having a window in an interpolated part of the image
            # we limit ourselves to a square of width 1/sqrt(2) smaller
            if(random_rotation > 2):
                boundary = int((imgWidth - imgWidth / np.sqrt(2)) / 2)
            else:
                boundary = 0
            center_x = np.random.randint(half_patch + boundary, imgWidth  - half_patch - boundary)
            center_y = np.random.randint(half_patch + boundary, imgHeight - half_patch - boundary)
            
            x = img[center_x - half_patch:center_x + half_patch + 2 * padSize,
                    center_y  - half_patch:center_y + half_patch + 2 * padSize]
            y = gt[center_x - half_patch : center_x + half_patch,
                   center_y - half_patch : center_y + half_patch]
            
            # vertical
            if(np.random.randint(0, 2)):
                x = np.flipud(x)
            
            # horizontal
            if(np.random.randint(0, 2)):
                x = np.fliplr(x)
            
            label = [0., 1.] if (np.array([np.mean(y)]) >  FG_THRESH) else [1., 0.]
            
            # makes sure we have an even distribution of road and non road if we oversample
            if not upsample:
                batch_input.append(x)
                batch_output.append(label)
            elif label == [1.,0.]:
                # case background
                background_count += 1
                if background_count != batch_size // 2:
                    batch_input.append(x)
                    batch_output.append(label)
            elif label == [0.,1.]:
                # case road
                road_count += 1
                if road_count != batch_size // 2:
                    batch_input.append(x)
                    batch_output.append(label)
                
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )

        yield( batch_x, batch_y )    
        