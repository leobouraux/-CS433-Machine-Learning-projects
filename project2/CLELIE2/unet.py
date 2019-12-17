from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Convolution Blocks
def convolution_down(prev_layer, nb_channels, acti='relu'):
    kernel_size = 3
    conv = Conv2D(nb_channels, kernel_size, activation=acti, padding='same', kernel_initializer='he_normal')(prev_layer)
    conv = BatchNormalization()(conv)
    conv = Conv2D(nb_channels, kernel_size, activation=acti, padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool
    
def convolution_up(prev_layer, nb_channels, layer_merging, acti='relu'):
    kernel_size = 3
    up = Conv2D(nb_channels, 2, activation=acti, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(prev_layer))
    merged = concatenate([layer_merging, up], axis=3)
    conv = Conv2D(nb_channels, kernel_size, activation=acti, padding='same', kernel_initializer='he_normal')(merged)
    conv = BatchNormalization()(conv)
    conv = Conv2D(nb_channels, kernel_size, activation=acti, padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization()(conv)
    return conv

# U-Net model for 256x256 images
def unet256(input_size, lr=0.005, verbose=True):
    inputs = Input(shape=input_size)
    kernel_size = 3

    # Creation of the layers
    conv1, pool1 = convolution_down(inputs, 64)
    conv2, pool2 = convolution_down(pool1, 128)
    conv3, pool3 = convolution_down(pool2, 256)
    conv4, pool4 = convolution_down(pool3, 512)
    conv5, _ = convolution_down(pool4, 1024)
    
    conv6 = convolution_up(conv5, 512, conv4)
    conv7 = convolution_up(conv6, 256, conv3)
    conv8 = convolution_up(conv7, 128, conv2)
    conv9 = convolution_up(conv8, 64, conv1)
    
    conv10 = Conv2D(2, kernel_size, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
    
    model = Model(inputs = inputs, outputs = conv11)
    
    model.compile(optimizer = Adam(lr=lr), loss = 'binary_crossentropy', metrics = ['acc', f1_m])
    
    if(verbose == True):
        model.summary()

    return model

# U-Net model for 512x512 images
def unet512(input_size, lr=0.005, verbose=True):
    
    inputs = Input(shape=input_size)
    
    # Creation of the layers
    conv1, pool1 = convolution_down(inputs, 64)
    conv2, pool2 = convolution_down(pool1, 128)
    conv3, pool3 = convolution_down(pool2, 256)
    conv4, pool4 = convolution_down(pool3, 512)
    conv5, pool5 = convolution_down(pool4, 1024)
    conv6, _ = convolution_down(pool5, 2048)
    
    conv7  = convolution_up(conv6, 1024, conv5)
    conv8  = convolution_up(conv7, 512, conv4)
    conv9  = convolution_up(conv8, 256, conv3)
    conv9  = convolution_up(conv9, 128, conv2)
    conv10 = convolution_up(conv10, 64, conv1)
    
    conv11 = Conv2D(2, kernel_size, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv10)
    conv12 = Conv2D(1, 1, activation = 'sigmoid')(conv11)
    
    model = Model(inputs = inputs, outputs = conv12)
    
    model.compile(optimizer = Adam(lr=lr), loss = 'binary_crossentropy', metrics = ['acc', f1_m])
    
    if(verbose == True):
        model.summary()

    return model

