import os
import random
import numpy as np
from scipy.misc import imresize, imread, imsave
from scipy.ndimage import zoom
from collections import defaultdict
from data_augmentation import randomRotation, randomCrop, randomColor, flip
from PIL import Image
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])

def preprocess_img(img, input_shape):
  img = imresize(img, input_shape)
  img = img - DATA_MEAN
  img = img[:, :, ::-1]
  img.astype('float32')
  return img

def update_inputs(batch_size = None, input_size = None, num_classes = None):
  return np.zeros([batch_size, input_size[0], input_size[1], 3]), \
    np.zeros([batch_size, input_size[0], input_size[1], num_classes])

def data_generator_s31(datadir='', nb_classes = None, batch_size = None, input_size=None, separator='_'):
  if not os.path.exists(datadir):
    print("ERROR!The folder is not exist")
  #listdir = os.listdir(datadir)
  data = defaultdict(dict)
  train_image_dir = os.path.join(datadir, "train")
  image_paths = os.listdir(train_image_dir)
  for image_path in image_paths:
    nmb = image_path.split(separator)[0]
    data[nmb]['image'] = image_path
  train_anno_dir = os.path.join(datadir, "trainannot")
  anno_paths = os.listdir(train_anno_dir)
  for anno_path in anno_paths:
    nmb = anno_path.split(separator)[0]
    data[nmb]['anno'] = anno_path
  train_values = data.values()
  train_values=list(train_values)

  data = defaultdict(dict)
  val_image_dir = os.path.join(datadir, "test")
  image_paths = os.listdir(val_image_dir)
  for image_path in image_paths:
    nmb = image_path.split(separator)[0]
    data[nmb]['image'] = image_path
  val_anno_dir = os.path.join(datadir, "testannot")
  anno_paths = os.listdir(val_anno_dir)
  for anno_path in anno_paths:
    nmb = anno_path.split(separator)[0]
    data[nmb]['anno'] = anno_path
  val_values = data.values()
  val_values=list(val_values)

  return generate(train_values, nb_classes, batch_size, input_size, train_image_dir, train_anno_dir, shuffle=True), \
      generate(val_values, nb_classes, batch_size, input_size, val_image_dir, val_anno_dir, shuffle=False)

def generate(values, nb_classes, batch_size, input_size, image_dir, anno_dir, shuffle):
  while 1:
    if(shuffle):
      random.shuffle(values)
    images, labels = update_inputs(batch_size=batch_size,
       input_size=input_size, num_classes=nb_classes)
    for i, d in enumerate(values):
      img = Image.open(os.path.join(image_dir, d['image']))
      img=img.resize(input_size, Image.BICUBIC)
      y = Image.open(os.path.join(anno_dir, d['anno']))
      y = y.resize(input_size, Image.NEAREST)
      if(shuffle):
        x = np.random.randint(0, 3)
        if(x!=1):
          x = np.random.randint(0, 2)
          if(x):
            img, y = flip(img, y)
          x = np.random.randint(0, 2)
          if(x):
            img, y = randomCrop(img, y, input_size)
          x = np.random.randint(0, 2)
          if(x):
            img, y = randomColor(img, y)
      img = np.array(img)
      y = np.array(y)
      h, w = input_size
      img = img - DATA_MEAN
      img = img[:, :, ::-1]
      img.astype('float32')
      y = (np.arange(nb_classes) == y[:,:,None]).astype('float32')
      assert y.shape[2] == nb_classes
      images[i % batch_size] = img
      labels[i % batch_size] = y
      if (i + 1) % batch_size == 0:
        yield images, labels
        images, labels = update_inputs(batch_size=batch_size,
          input_size=input_size, num_classes=nb_classes)