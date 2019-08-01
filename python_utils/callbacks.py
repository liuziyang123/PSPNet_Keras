import os
import time
import keras.backend as K
from keras.callbacks import Callback, TensorBoard, ReduceLROnPlateau, ModelCheckpoint

class LrReducer(Callback):
  def __init__(self, base_lr, max_epoch = 100, power=0.9, verbose=1):
    super(Callback, self).__init__()
    self.max_epoch = max_epoch
    self.power = power
    self.verbose = verbose
    self.base_lr = base_lr

  def on_epoch_end(self, epoch, logs={}):
    lr_now = K.get_value(self.model.optimizer.lr)
    new_lr = max(0.00001, min(self.base_lr * (1 - epoch / float(self.max_epoch))**self.power, lr_now))
    K.set_value(self.model.optimizer.lr, new_lr)
    if self.verbose:
        print(" - learning rate: %10f" % (new_lr))

class ParallelModelCheckpoint(ModelCheckpoint):
  def __init__(self, model, filepath, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=10):
    self.single_model = model
    super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

  def set_model(self, model):
    super(ParallelModelCheckpoint, self).set_model(self.single_model)



def callbacks(logdir, model, learning_rate):
  model_checkpoint = ParallelModelCheckpoint(model, filepath=(logdir + "/weights.{epoch:02d}-{loss:.2f}.h5"), monitor='loss', verbose=1, mode='auto')
  ##model_checkpoint = ModelCheckpoint("weights_train/weights.{epoch:02d}-{loss:.2f}.h5", monitor='loss', verbose=1, period=1)
  tensorboard_callback = TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
  plateau_callback = ReduceLROnPlateau(monitor='loss', factor=0.6, verbose=1, patience=2, min_lr=0.00001)
  #return [CheckPoints(), tensorboard_callback, LrReducer()]
  return [model_checkpoint, tensorboard_callback, plateau_callback, LrReducer(base_lr=learning_rate)]
