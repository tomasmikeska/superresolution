import os
import argparse
import numpy as np
from comet_ml import Experiment
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.optimizers import Adam
from model import create_model
from losses.perceptual_loss import perceptual_loss
from dataset import TrainDatasetSequence, TestDatasetSequence
from callbacks.log_images import LogImages
from utils import relative_path, file_listing


def psnr_metric(max_pixel=1.0):
    '''
    Computes PSNR metric

    Args:
        max_pixel: Max value pixel can take on. We scale inputs to (0, 1

    Note: 2.303 is natural and log10 conversion factor
    '''
    def psnr(y_true, y_pred):
        return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303
    return psnr


def train(model, args, experiment):
    output_shape = (args.input_h * args.scale, args.input_w * args.scale, 3)
    train_seq = TrainDatasetSequence(args.train_dataset,
                                     batch_size=args.batch_size,
                                     input_size=(args.input_w, args.input_h),
                                     scale=args.scale)
    test_seq = TestDatasetSequence(args.test_dataset,
                                   batch_size=args.batch_size,
                                   input_size=(args.input_w, args.input_h),
                                   scale=args.scale)
    model.compile(optimizer=Adam(lr=3e-4),
                  loss=perceptual_loss(output_shape),
                  metrics=['mse', psnr_metric()])
    model.summary()

    if args.weights:
        model.load_weights(args.weights)

    callbacks = [
        ModelCheckpoint(
            args.model_save_path + 'sr_{epoch:02d}_{val_loss:.3f}.h5',
            save_weights_only=True,
            verbose=1),
        TerminateOnNaN(),
        LogImages(
            experiment,
            paths=file_listing(args.validation_path),
            input_size=(args.input_w, args.input_h),
            scale=args.scale,
            log_iters=500)
    ]

    model.fit_generator(
        train_seq,
        epochs=args.epochs,
        validation_data=test_seq,
        use_multiprocessing=True,
        workers=8,
        callbacks=callbacks)


if __name__ == '__main__':
    # Command line arguments parsing
    parser = argparse.ArgumentParser(description='Train a colorization deep learning model')
    parser.add_argument('--train-dataset',
                        type=str,
                        default=relative_path('../data/imagenet-sample/train/'),
                        help='Train dataset base path. Folder should contain subfolder for each class.')
    parser.add_argument('--test-dataset',
                        type=str,
                        default=relative_path('../data/imagenet-sample/test/'),
                        help='Test dataset base path. Folder should contain images directly.')
    parser.add_argument('--validation-path',
                        type=str,
                        default=relative_path('../data/imagenet-sample/val/'),
                        help='Path to directory with validation images that will be uploaded to comet after each epoch')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='Batch size used during training')
    parser.add_argument('--input-w',
                        type=int,
                        default=128,
                        help='Image width')
    parser.add_argument('--input-h',
                        type=int,
                        default=128,
                        help='Image height')
    parser.add_argument('--scale',
                        type=int,
                        default=3,
                        help='Target img size / input img size scale factor')
    parser.add_argument('--weights',
                        type=str,
                        help='Model weights')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs')
    parser.add_argument('--model-save-path',
                        type=str,
                        default=relative_path('../model/'),
                        help='Base directory to save model during training')
    args = parser.parse_args()
    # CometML experiment
    experiment = Experiment(api_key=os.getenv('COMET_API_KEY'),
                            project_name=os.getenv('COMET_PROJECTNAME'),
                            workspace=os.getenv('COMET_WORKSPACE'))
    # Train
    model = create_model((args.input_h, args.input_w, 3), args.scale)
    train(model, args, experiment)
