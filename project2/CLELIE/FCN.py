from keras.models import Sequential
from keras.layers import Conv2DTranspose, Conv2D, Flatten, MaxPooling2D, LeakyReLU, Dropout, Activation, GlobalAveragePooling2D

def model_FCN(input_shape):
    model = Sequential()
    model.add(Conv2D(32, 2, input_shape=input_shape))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(32, 2))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 2))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(64, 2))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2DTranspose(1, kernel_size=1, padding='valid'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('sigmoid'))
    return model