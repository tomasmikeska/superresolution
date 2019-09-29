import numpy as np
import skimage
from keras.callbacks import Callback
from dataset import read_pair


class LogImages(Callback):
    '''Log colorized images to CometML'''

    def __init__(self,
                 experiment,
                 paths=[],
                 model=None,
                 input_size=(32, 32),
                 scale=3,
                 log_iters=1000):
        '''Callback initializer

        Args:
            experiment (CometMl Experiment): CometML Experiment instance
            paths ([string]): List of paths to images that will be colorized and uploaded
            input_size (tuple): Image size tuple in format (width, height)
            scale (int): Superresolution scale
            log_iters (int): Number of batches after which callback is run
        '''
        self.experiment = experiment
        self.paths      = paths
        self.input_size = input_size
        self.scale      = scale
        self.log_iters  = log_iters
        self.iter       = 0
        if model:
            self.model = model

    def on_batch_end(self, batch, logs):
        self.iter += 1
        if self.iter % self.log_iters == 0:
            self.log_images(self.iter)

    def log_images(self, iter):
        ground_truth = []
        batch = []
        # Read images
        for path in self.paths:
            input_img, target_img = read_pair(path, self.input_size, self.scale)
            ground_truth.append(target_img)
            batch.append(input_img)
        # Predict superresolution images using trained model
        sr_images = self.model.predict(np.array(batch))
        # Concat ground truth and predicted images and log them to comet
        for i in range(len(sr_images)):
            input_upsampled = skimage.transform.resize(batch[i], sr_images[i].shape[:2])
            final = np.concatenate((input_upsampled, sr_images[i], ground_truth[i]), axis=1)
            final = np.rint(final * 255).astype(np.uint8)
            self.experiment.log_image(final, name=f'iter_{iter:06d}_image_{i:02d}')
