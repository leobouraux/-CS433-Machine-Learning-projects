from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, ZeroPadding2D, Convolution2D, Dropout

def model_LIDAR(input_shape=(16, 16, 3)):
    K_size = (3,3) # specifying the height and width of the 2D convolution window
    #input_shape = (16,16,3)
    mod = Sequential()
    
    # Encoder 
    # Convolutional 3*3, stride=1, zero-padding + ELU
    mod.add(Conv2D(32, kernel_size=K_size, activation='elu', input_shape=input_shape, padding='same'))
    mod.add(ZeroPadding2D(1))
    # Convolutional 3*3, stride=1, zero-padding + ELU
    mod.add(Conv2D(32, kernel_size=K_size, activation='elu', padding='same'))
    mod.add(ZeroPadding2D(1))
    # Max-pooling 2*2, stride=2
    mod.add(MaxPooling2D((2,2), strides=(2,2), padding='same'))
     
    # Context Module
    # Dilated Convolution 3*3, stride=1, zero-padding + spatial dropout + ELU
    mod.add(Conv2D(128, kernel_size = K_size, dilation_rate = (1,1), activation='elu', padding='same'))        
    mod.add(ZeroPadding2D(2))
    mod.add(Dropout(0.3))
    # Dilated Convolution 3*3, stride=1, zero-padding + spatial dropout + ELU
    mod.add(Conv2D(128, kernel_size=K_size, dilation_rate = (1,2), activation='elu', padding='same'))        
    #mod.add(ZeroPadding2D())
    mod.add(Dropout(0.3))
    # Dilated Convolution 3*3, stride=1, zero-padding + spatial dropout + ELU
    mod.add(Conv2D(128, kernel_size=K_size, dilation_rate = (2,4), activation='elu', padding='same'))        
    mod.add(ZeroPadding2D(1))
    mod.add(Dropout(0.3))
    # Dilated Convolution 3*3, stride=1, zero-padding + spatial dropout + ELU
    mod.add(Conv2D(128, kernel_size=K_size, dilation_rate = (4,8), activation='elu', padding='same'))        
    mod.add(ZeroPadding2D(1))
    mod.add(Dropout(0.3))    
    # Dilated Convolution 3*3, stride=1, zero-padding + spatial dropout + ELU
    mod.add(Conv2D(128, kernel_size=K_size, dilation_rate = (8,16), activation='elu', padding='same'))        
    #mod.add(ZeroPadding2D())
    mod.add(Dropout(0.3))    
    # Dilated Convolution 3*3, stride=1, zero-padding + spatial dropout + ELU
    mod.add(Conv2D(128, kernel_size=K_size, dilation_rate = (16,32), activation='elu', padding='same'))        
    #mod.add(ZeroPadding2D())
    mod.add(Dropout(0.3))    
    # Dilated Convolution 3*3, stride=1, zero-padding + spatial dropout + ELU
    mod.add(Conv2D(128, kernel_size=K_size, dilation_rate = (32,64), activation='elu', padding='same'))        
    #mod.add(ZeroPadding2D())
    mod.add(Dropout(0.3))
    # Convolution 1*1
    mod.add(Conv2D(32, kernel_size=(1,1)))
            
    # Decoder : 
    # Max unpooling
    # mod.add(MaxUnpooling())
    # Convolutional 3*3, stride=1, zero-padding + ELU
    mod.add(Conv2D(32, kernel_size=K_size, activation='elu', padding='same'))
    mod.add(ZeroPadding2D())
    # Convolutional 3*3, stride=1, zero-padding + ELU
    mod.add(Conv2D(2, kernel_size=K_size, activation='elu', padding='same'))
    mod.add(ZeroPadding2D())
    mod.add(Flatten())
    # Softmax
    mod.add(Dense(2, activation='softmax'))

    return mod
