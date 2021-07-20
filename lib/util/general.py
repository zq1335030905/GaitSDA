from PIL import Image
import os
import json
import logging
import shutil
import csv
# from lib.network.munit import Vgg16
from torch.autograd import Variable
from torch.optim import lr_scheduler
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init
import time


def get_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def testJoints2Video(data, iter, flag, note="casiab_02_09_w/o_id"):
    """
    :param data:    shape: [K*D, T]
    :return:
    """
    print(data.shape, iter)
    if flag:
        T,K,D = data.shape
        # data = data.cpu().detach().numpy()
        # data = data.detach().numpy()
    else:
#        KD, T = data.shape
#        data = torch.from_numpy(data)
#        data = torch.Tensor(data.float())
        data = data.view(64, 15, 2)
        data = data.cpu().detach().numpy()
    limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [8, 12], [9, 10], [10, 11],
               [12, 13], [13, 14]]
    for t in range(data.shape[0]):
        fig = plt.figure(t, figsize=(4, 10))
        ax = fig.gca()
        print(t, data[t, 7, 0], data[t, 7, 1])
        for i in range(15):
            plt.plot(data[t, i, 0], data[t, i, 1], 'bo', color="black")
        for i in range(len(limbSeq)):
            x = [data[t, limbSeq[i][0], 0], data[t, limbSeq[i][1], 0]]
            y = [data[t, limbSeq[i][0], 1], data[t, limbSeq[i][1], 1]]
            plt.plot(x,y,marker='o')
        if not os.path.exists("save_video/{}/{}/".format(note, iter+1)):
            os.makedirs("save_video/{}/{}/".format(note, iter+1))
        plt.savefig("save_video/{}/{}/plt_{}.png".format(note, iter+1, t))
        plt.close(fig)
        # plt.show()
