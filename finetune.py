from os import path
from os.path import join
from scipy.misc import imresize
from python_utils.preprocessing import data_generator_s31
from python_utils.callbacks import callbacks
from keras.models import load_model
import layers_builder as layers
import numpy as np
import argparse
import os
from keras.optimizers import Adam, SGD
from metrics import iou_score
from losses import cce_jaccard_loss
from keras.utils.training_utils import multi_gpu_model
learning_rate = 1e-3

def train(datadir, logdir, input_size, nb_classes, resnet_layers, batchsize, initial_epoch, sep, gpu_num):
    if True:
        model = layers.build_pspnet(nb_classes=nb_classes,
                                    resnet_layers=resnet_layers,
                                    input_shape=input_size,
                                    freeze=True)
        model.layers[-3].name = 'conv7'
        model.load_weights(logdir + "/pspnet101_cityscapes.h5", by_name=True)
        print("Load pre-trained weights")
    parallel_model = multi_gpu_model(model, gpus=gpu_num)
    adam = Adam(lr=learning_rate, amsgrad=True)
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    parallel_model.compile(optimizer=sgd,
                  loss=cce_jaccard_loss,
                  metrics=[iou_score, 'accuracy'])

    train_generator, val_generator = data_generator_s31(
        datadir=datadir, batch_size=batchsize, input_size=input_size, nb_classes=nb_classes, separator=sep)
    parallel_model.fit_generator(
        generator=train_generator,
        epochs=100, verbose=True, steps_per_epoch=366//batchsize+1,
        validation_data=val_generator, validation_steps=233//batchsize+1,
        callbacks=callbacks(logdir, model, learning_rate), initial_epoch=initial_epoch)
    model.save_weights(logdir + '/trained_weights_stage_1.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=713)
    parser.add_argument('--classes', type=int, default=12)
    parser.add_argument('--resnet_layers', type=int, default=101)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--datadir', type=str, default='./CamVid')
    parser.add_argument('--logdir', type=str, default='./log')
    parser.add_argument('--initial_epoch', type=int, default=0)
    parser.add_argument('--sep', default=').')
    args = parser.parse_args()

    gpu_num = 2
    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

    train(args.datadir, args.logdir, (args.input_dim, args.input_dim), args.classes, args.resnet_layers,
          args.batch, args.initial_epoch, args.sep, gpu_num)
