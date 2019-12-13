def unet(input_size, verbose=True):
    
    inputs = Input(shape=input_size)
    kernel_size = 3
    
    # Convolution Blocks
    def convolution_down(prev_layer, nb_channels, acti='relu'):
        conv = Conv2D(nb_channels, kernel_size, activation=acti, padding='same', kernel_initializer='he_normal')(prev_layer)
        conv = BatchNormalization()(conv)
        conv = Conv2D(nb_channels, kernel_size, activation=acti, padding='same', kernel_initializer='he_normal')(conv)
        conv = BatchNormalization()(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool
    
    def convolution_up(prev_layer, nb_channels, layer_merging, acti='relu'):
        up = Conv2D(nb_channels, 2, activation=acti, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(prev_layer))
        merged = concatenate([layer_merging, up], axis=3)
        conv2 = Conv2D(nb_channels, kernel_size, activation=acti, padding='same', kernel_initializer='he_normal')(merged)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(nb_channels, kernel_size, activation=acti, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        return conv2
    
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
    conv9 = Conv2D(2, kernel_size, activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    
    model = Model(inputs = inputs, outputs = conv10)
    
    model.compile(optimizer = Adam(lr = 0.005), loss = 'binary_crossentropy', metrics = ['acc', f1_m])
    if(verbose == True):
        model.summary()

    return model