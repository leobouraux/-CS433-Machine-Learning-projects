from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

from data_postprocess import *
from data_preprocess import *
from unet import *
from constants import *

def fit_unet(rotation, MODEL_PATH, MODEL_NAME):
    # Load dataset
    data_gen_args = dict(rotation_range=rotation,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    train_generator = dataGenerator(2, DATA_PATH+'train',
                                    'img_aug_tr',
                                    'gts_aug_tr'
                                    ,data_gen_args,
                                    (SIDE,SIDE))

    validation_generator = dataGenerator(2, DATA_PATH+'train',
                                    'img_aug_te',
                                    'gts_aug_te'
                                    ,data_gen_args,
                                    (SIDE,SIDE))

    filepath = "weights.{epoch:02d}-{val_f1_m:.2f}.hdf5"

    csv_logger = CSVLogger("AccuracyHistory.csv")
    cp_callback = ModelCheckpoint(filepath=filepath, verbose=1, save_weights_only=True, period=1)
    
    # Load and fit the model
    model = unet256((SIDE,SIDE,3),lr=0.001, verbose=False)
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=200,
                    epochs=100,
                    verbose=1,
                    validation_data = validation_generator,
                    validation_steps = 70,
                    validation_freq=1,
                    initial_epoch=0,
                    callbacks=[cp_callback, csv_logger])

    return model
    
