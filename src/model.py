import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from layers.pixel_shuffle import PixelShuffle


def BilinearUpSampling2D(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (stride[0] * input_shape[1], stride[1] * input_shape[2])
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)


def conv_block(x, filters=64, batch_norm=False):
    x = Conv2D(64, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def up_block(x, y, filters=64, batch_norm=False):
    x = BilinearUpSampling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([x, y])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def create_model(input_shape=(64, 64, 3), scale_factor=3):
    '''
    U-Net based architecture with skip connections and bilinear interpolation upsampling.
    PixelShuffle is added as a final upsampling layer to enlarge to SR.
    '''

    img_input = Input(shape=input_shape)

    # Encoder
    enc1 = conv_block(img_input, 64)
    enc1 = conv_block(enc1, 64)
    down = MaxPooling2D()(enc1)

    enc2 = conv_block(down, 128)
    enc2 = conv_block(enc2, 128)
    down = MaxPooling2D()(enc2)

    enc3 = conv_block(down, 256)
    enc3 = conv_block(enc3, 256)
    down = MaxPooling2D()(enc3)

    enc4 = conv_block(down, 512)
    enc4 = conv_block(enc4, 512)
    down = MaxPooling2D()(enc4)

    enc5 = conv_block(down, 1024)
    enc5 = conv_block(enc5, 1024)

    # Center
    cent = Conv2D(256, (1, 1))(enc5)
    cent = ReLU()(cent)

    # Decoder
    dec = up_block(cent, enc4, 512)
    dec = up_block(dec, enc3, 256)
    dec = up_block(dec, enc2, 128)
    dec = up_block(dec, enc1, 64)

    dec = Conv2D(3 * (scale_factor ** 2), (3, 3), padding='same')(dec)
    dec = PixelShuffle(r=scale_factor)(dec)
    dec = Activation('sigmoid')(dec)

    return Model(img_input, dec)
