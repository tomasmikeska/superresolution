'''
Implementation of perceptual loss from paper Perceptual Losses for Real-Time Style Transfer and Super-Resolution
(link: https://arxiv.org/abs/1603.08155)

It measures distance (L1 distance in our case) between output of several layers of VGG16 pretrained model
for ground truth and output images.
'''
import keras.backend as K
from keras.models import Model
from keras.layers import MaxPooling2D
from keras.applications.vgg16 import VGG16


def l1_distance(a, b):
    return K.mean(K.abs(a - b))


def l2_distance(a, b):
    return K.mean(K.square(a - b))


def perceptual_loss(input_shape=(64, 64, 3),
                    loss_weights=[5, 15, 2]):
    # Base feature loss model
    vgg16 = VGG16(input_shape=input_shape,
                  include_top=False,
                  weights='imagenet')
    vgg16.trainable = False
    # Convert to model with output of 3 hook layers
    blocks = [i - 1 for i, l in enumerate(vgg16.layers) if isinstance(l, MaxPooling2D)]  # All blocks before maxpooling
    layer_ids = blocks[2:5]  # Filter only the higher layers
    loss_features = [vgg16.layers[i] for i in layer_ids]
    loss_model = Model(vgg16.inputs, list(map(lambda l: l.output, loss_features)))

    def loss(y_true, y_pred):
        in_feat = loss_model(y_true)
        out_feat = loss_model(y_pred)
        loss = l1_distance(y_true, y_pred)
        loss += [l1_distance(f_in, f_out) * w for f_in, f_out, w in zip(in_feat, out_feat, loss_weights)]
        return K.mean(loss)

    return loss
