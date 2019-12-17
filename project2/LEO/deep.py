from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, ZeroPadding2D, Convolution2D, Dropout

def model_deep(input_shape = (400,400,3)):

        ##Parameters
        model = Sequential()
        kernel_size = (3,3)
        pool_size = (2,2)
        alpha_relu = 0.1
        regularizer = 1e-6
        
        model = Sequential()

        #Add convolution 
        model.add(Convolution2D(64, kernel_size, padding='same', input_shape=shape))
        model.add(LeakyReLU(alpha_relu))
        model.add(MaxPooling2D(pool_size))
        
        model.add(Convolution2D(64, kernel_size, padding='same', input_shape=shape))
        model.add(Dropout(0.1))
        model.add(LeakyReLU(alpha_relu))
        model.add(MaxPooling2D(pool_size))

        model.add(Convolution2D(128, kernel_size, padding='same', input_shape=shape))
        model.add(LeakyReLU(alpha_relu))
        model.add(MaxPooling2D(pool_size))
        
        model.add(Convolution2D(128, kernel_size, padding='same', input_shape=shape))
        model.add(Dropout(0.1))
        model.add(LeakyReLU(alpha_relu))
        model.add(MaxPooling2D(pool_size))

        model.add(Convolution2D(256, kernel_size, padding='same', input_shape=shape))
        model.add(LeakyReLU(alpha_relu))
        model.add(MaxPooling2D(pool_size))
        
        model.add(Convolution2D(256, kernel_size, padding='same', input_shape=shape))
        model.add(Dropout(0.1))
        model.add(LeakyReLU(alpha_relu))

        model.add(Flatten())

        model.add(Dense(2))
        model.add(Activation('softmax'))
        
        return model