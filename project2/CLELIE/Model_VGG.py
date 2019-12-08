from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, ZeroPadding2D, Dropout, LeakyReLU

def model_VGG(input_shape):
    alpha = 0.0001
    
    model = Sequential()
    model.add(Conv2D(16, 2, input_shape=input_shape))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(16, 2))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, 2))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(32, 2))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 1))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(64, 1))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dense(2, activation='sigmoid'))
    
    return model