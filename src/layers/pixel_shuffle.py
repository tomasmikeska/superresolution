import keras.backend as K
from keras.layers import Layer


class PixelShuffle(Layer):
    '''Rearranges elements in a tensor of shape (B, C * r^2, H, W) to a tensor of shape (B, C, H * r, W * r)'''

    def __init__(self, r=2, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.r = r

    def call(self, I):
        r = self.r
        _, h, w, c = I.get_shape().as_list()
        batch_size = K.shape(I)[0]
        X = K.reshape(I, [batch_size, h, w, int(c / r**2), r, r])  # (batch_size, a, b, c/(r*r), r, r)
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # (batch_size, a, b, r, r, c/(r*r))
        X = [X[:, i, :, :, :, :] for i in range(h)]  # [ (batch_size, b, r, r, c/(r*r)), ... ]
        X = K.concatenate(X, 2)  # (batch_size, b, a*r, r, c/(r*r))
        X = [X[:, i, :, :, :] for i in range(w)]  # [ (batch_size, a*r, r, c/(r*r)), ... ]
        X = K.concatenate(X, 2)  # (batch_size, a*r, b*r, c/(r*r))
        return X

    def compute_output_shape(self, input_shape):
        r = self.r
        batch_size, a, b, c = input_shape
        return (batch_size, a * r, b * r, c // (r * r))

    def get_config(self):
        config = super(Layer, self).get_config()
        config['r'] = self.r
        return config
