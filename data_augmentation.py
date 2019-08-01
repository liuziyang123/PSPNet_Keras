import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance

def randomRotation(image, label, mode=Image.BICUBIC):
  random_angle = np.random.randint(1, 360)
  return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

def flip(image, label):
  x = np.random.randint(0, 2)
  if(x):
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    label = label.transpose(Image.FLIP_LEFT_RIGHT)

  x = np.random.randint(0, 2)
  if (x):
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    label = label.transpose(Image.FLIP_TOP_BOTTOM)

  x = np.random.randint(0, 2)
  if (x):
    image = image.transpose(Image.ROTATE_90)
    label = label.transpose(Image.ROTATE_90)

  x = np.random.randint(0, 2)
  if (x):
    image = image.transpose(Image.ROTATE_270)
    label = label.transpose(Image.ROTATE_270)

  return  image, label

def randomCrop(image, label, input_size):
  image_width = image.size[0]
  image_height = image.size[1]
  crop_win_size = np.random.randint(360, 473)
  random_region = ((image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
            (image_height + crop_win_size) >> 1)
  image = image.crop(random_region)
  label = label.crop(random_region)
  return image.resize(input_size, Image.BICUBIC), label.resize(input_size, Image.NEAREST)

def randomColor(image, label):
  random_factor = np.random.randint(0, 31) / 10.
  color_image = ImageEnhance.Color(image).enhance(random_factor)
  random_factor = np.random.randint(10, 21) / 10.
  brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
  random_factor = np.random.randint(10, 21) / 10.
  contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
  random_factor = np.random.randint(0, 31) / 10.
  return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label