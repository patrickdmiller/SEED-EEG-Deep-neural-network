import logging
import os
import random
import time
import pickle
import numpy as np
import torch
from torcheeg.trainers import ClassificationTrainer
from torch import nn

import math

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


class Results:
  def __init__(self):
    self.epochs = {}

  def add(self, epoch, key, loss, accuracy):
    if epoch not in self.epochs:
      self.epochs[epoch] = {'train': Result(), 'val': Result()}

    self.epochs[epoch][key].add(loss, accuracy)

  def get_loss(self, epoch, key):
    return self.epochs[epoch][key].loss

  def get_accuracy(self, epoch, key):
    return self.epochs[epoch][key].accuracy

  def to_dict(self):
    d = {}
    for epoch in self.epochs:
      d[epoch] = {}
      for key in self.epochs[epoch]:
        d[epoch][key] = self.epochs[epoch][key].to_dict

    return d

  def save_pickle_with_dir_and_name(self, dir, name):
    #if dir doesn't exist, create it
    if not os.path.exists(dir):
      os.makedirs(dir)

    with open(os.path.join(dir, name), 'wb') as f:
      pickle.dump(self.to_dict(), f)


class Result:
  def __init__(self):
    self.count = 0
    self._loss = 0
    self._accuracy = 0

  @property
  def to_dict(self):
    return {'loss': self.loss, 'accuracy': self.accuracy}

  @property
  def loss(self):
    return self._loss / self.count

  @property
  def accuracy(self):
    return self._accuracy / self.count

  def add(self, loss, accuracy):
    self.count += 1
    self._loss += loss
    self._accuracy += accuracy

  def reset(self):
    self.count = 0
    self._loss = 0
    self._accuracy = 0


class Trainer(ClassificationTrainer):
  def __init__(self,
               model,
               num_classes=3,
               lr=1e-4,
               weight_decay=0.0,
               device_ids=[],
               ddp_sync_bn=True,
               ddp_replace_sampler=True,
               ddp_val=True,
               ddp_test=True,
               do_scheduler=True,
               scheduler_gamma=0.9,
               scheduler_step_size=10):
    super(Trainer,
          self).__init__(model=model,
                         num_classes=num_classes,
                         lr=lr,
                         weight_decay=weight_decay,
                         device_ids=device_ids,
                         ddp_sync_bn=ddp_sync_bn,
                         ddp_replace_sampler=ddp_replace_sampler,
                         ddp_val=ddp_val,
                         ddp_test=ddp_test)
    self.optimizer = torch.optim.Adam(model.parameters(),
                                      lr=lr,
                                      weight_decay=weight_decay)
    self.do_scheduler = do_scheduler
    self.scheduler = torch.optim.lr_scheduler.StepLR(
        self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    self.results = Results()
    self.epoch = 0


  def after_validation_epoch(self, epoch_id: int, num_epochs: int, **kwargs):
    if self.do_scheduler:
      self.scheduler.step()
    val_accuracy = 100 * self.val_accuracy.compute()
    val_loss = self.val_loss.compute()
    self.results.add(self.epoch, 'val', val_loss, val_accuracy)
    self.log(f"\nloss: {val_loss:>8f}, accuracy: {val_accuracy:>0.1f}%")
    self.epoch += 1

  def log(self, *args, **kwargs):
    if self.is_main:
      self.logger.info(*args, **kwargs)

  def on_training_step(self, train_batch, batch_id, num_batches, **kwargs):
    self.train_accuracy.reset()
    self.train_loss.reset()

    X = train_batch[0].to(self.device)
    y = train_batch[1].to(self.device)

    # compute prediction error
    pred = self.modules['model'](X)
    loss = self.loss_fn(pred, y)

    # backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # log five times
    log_step = math.ceil(num_batches / 5)
    #         if batch_id % log_step == 0:
    self.train_loss.update(loss)
    self.train_accuracy.update(pred.argmax(1), y)

    train_loss = self.train_loss.compute()
    train_accuracy = 100 * self.train_accuracy.compute()
    self.results.add(self.epoch, 'train', train_loss, train_accuracy)
    # if not distributed, world_size is 1

    batch_id = batch_id * self.world_size
    num_batches = num_batches * self.world_size
    if self.is_main and batch_id % log_step == 0:
      self.log(
          f"loss: {train_loss:>8f}, accuracy: {train_accuracy:>0.1f}% [{batch_id:>5d}/{num_batches:>5d}]"
      )