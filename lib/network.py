import sys
thismodule = sys.modules[__name__]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.loss import SmoothCrossEntropyLoss, CenterLoss, TripletLoss

torch.manual_seed(123)

def get_autoencoder(config):
    ae_cls = getattr(thismodule, config.autoencoder.cls)
    return ae_cls(config.autoencoder)

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.float().cuda()
    else:
        return variable

class ConvEncoder(nn.Module):

    @classmethod
    def build_from_config(cls, config):
        conv_pool = None if config.conv_pool is None else getattr(nn, config.conv_pool)
        encoder = cls(config.channels, config.padding, config.kernel_size, config.conv_stride, conv_pool)
        return encoder

    def __init__(self, channels, padding=3, kernel_size=8, conv_stride=2, conv_pool=None):
        super(ConvEncoder, self).__init__()

        self.in_channels = channels[0]

        model = []
        acti = nn.LeakyReLU(0.2)

        nr_layer = len(channels) - 1

        for i in range(nr_layer):
            if conv_pool is None:
                model.append(nn.ReflectionPad1d(padding))
                model.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernel_size, stride=conv_stride))
                model.append(nn.LeakyReLU(0.2))
            else:
                model.append(nn.ReflectionPad1d(padding))
                model.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernel_size, stride=conv_stride))
                model.append(acti)
                model.append(conv_pool(kernel_size=2, stride=2))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x[:, :self.in_channels, :]
        x = trans_to_cuda(x)
        x = self.model(x)
        return x

class ConvDecoder(nn.Module):

    @classmethod
    def build_from_config(cls, config):
        decoder = cls(config.channels, config.kernel_size)
        return decoder

    def __init__(self, channels, kernel_size=7):
        super(ConvDecoder, self).__init__()

        model = []
        pad = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)
        tanh = nn.Tanh()

        for i in range(len(channels) - 1):
            model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            model.append(nn.ReflectionPad1d(pad))
            model.append(nn.Conv1d(channels[i], channels[i + 1],
                                            kernel_size=kernel_size, stride=2))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=0.2))
            if not i == len(channels) - 2:
                model.append(acti)          # whether to add tanh at last?
                #model.append(nn.Dropout(p=0.2))
            # if i == len(channels) - 2:
            #     model.append(tanh)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def conv7x7(input_size, output_size, stride = 1, padding = 1, group = 1, bn = False):
	"""
	7x7 convolution with padding

	nn.Conv2d
	    input: [N, Cin, H, W]
	    output: [N, Cout, Hout, Wout]
	"""
	layers = []
	conv = nn.Conv2d(input_size, output_size, kernel_size = 7, stride = stride,
					 padding = padding, groups=group, bias = True)
	layers.append(conv)
	if bn:
		bn_layer = nn.BatchNorm2d(output_size)
		layers.append(bn_layer)
	layers.append(nn.ReLU(inplace = True))
	return nn.Sequential(*layers)

def conv3x3(input_size, output_size, stride = 1, padding = 1, group = 1, bn = False):
	"""
	3x3 convolution with padding

	nn.Conv2d
	    input: [N, Cin, H, W]
	    output: [N, Cout, Hout, Wout]
	"""
	layers = []
	conv = nn.Conv2d(input_size, output_size, kernel_size = 3, stride = stride,
					 padding = padding, groups=group, bias = True)
	layers.append(conv)
	if bn:
		bn_layer = nn.BatchNorm2d(output_size)
		layers.append(bn_layer)
	layers.append(nn.ReLU(inplace = True))
	return nn.Sequential(*layers)

def conv1x1(in_planes, out_planes, stride=1, padding=1, group=1, bn=False):
    """1x1 convolution with padding"""
    layers = []
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=padding, groups=group, bias=False)
    layers.append(conv)

    if bn:
        bn_layer = nn.BatchNorm2d(out_planes)
        layers.append(bn_layer)
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class fc(nn.Module):
    def __init__(self, config):
        super(fc, self).__init__()
        self.output_dim = config.output_dim
        self.config = config
        self.bn = True
        self.conv1 = conv3x3(1, 32, stride=1, padding=1, group=1, bn=self.bn)
        self.conv2 = conv3x3(32, 64, stride=1, padding=1, bn=self.bn)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = conv3x3(64, 64, stride=1, padding=1, group=1, bn=self.bn)
        self.conv4 = conv3x3(64, 64, stride=1, padding=1, group=1, bn=self.bn)

        self.conv5 = conv3x3(64, 128, stride=1, padding=1, group=1, bn=self.bn)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv6 = conv3x3(128, 128, stride=1, padding=1, group=1, bn=self.bn)
        self.conv7 = conv3x3(128, 128, stride=1, padding=1, group=1, bn=self.bn)

        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(config.input_dim, config.output_dim)
        self.fc_cls = nn.Linear(config.output_dim, config.class_num)

        self.center_loss = CenterLoss(num_classes=self.config.class_num, feat_dim=self.config.output_dim, use_gpu=True)

    def forward(self, x, phase):
        Batch, feat_m, dim = x.shape
        x = x.view(Batch, 1, feat_m, dim)
        print("[FC x input]:{}".format(x.shape))

        x = self.conv1(x)
        x = self.conv2(x)
        pool1 = self.pool(x)
        x = self.conv3(pool1)
        x = self.conv4(x)
        x = pool1 + x

        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        if phase == "train":
            x = self.drop(x)
        feature = self.fc(x)
        out = self.fc_cls(feature)

        return feature, out

    def cal_id_loss(self, out, label):
        """计算id的loss"""
        # feat, out = self.forward(feature, phase)
        if self.config.loss == "CrossEntropy":
            loss_func = nn.CrossEntropyLoss()
        elif self.config.loss == "SmoothCrossEntropy":
            loss_func = SmoothCrossEntropyLoss(smoothing=self.config.smooth)
        else:
            raise NotImplementedError
        loss = loss_func(out, label.long())
        return loss

    def cal_center_loss(self, feat, label):
        """计算centerloss"""
        # feat, out = self.forward(feature, phase)
        loss_func = self.center_loss
        loss = loss_func(feat, label)
        return loss

class Autoencoder3f(nn.Module):

    def __init__(self, config):
        super(Autoencoder3f, self).__init__()

        assert config.motion_encoder.channels[-1] + config.body_encoder.channels[-1] + \
               config.view_encoder.channels[-1] == config.decoder.channels[0]

        self.n_joints = config.decoder.channels[-1] // 3

        motion_cls = getattr(thismodule, config.motion_encoder.cls)
        body_cls = getattr(thismodule, config.body_encoder.cls)
        view_cls = getattr(thismodule, config.view_encoder.cls)

        self.motion_encoder = motion_cls.build_from_config(config.motion_encoder)
        self.body_encoder = body_cls.build_from_config(config.body_encoder)
        self.view_encoder = view_cls.build_from_config(config.view_encoder)
        self.decoder = ConvDecoder.build_from_config(config.decoder)

        self.body_pool = getattr(F, config.body_encoder.global_pool) if config.body_encoder.global_pool is not None else None
        self.view_pool = getattr(F, config.view_encoder.global_pool) if config.view_encoder.global_pool is not None else None

    def forward(self, seqs):
        return self.reconstruct(seqs)

    def encode_motion(self, seqs):
        # print("seq shape: ", seqs.shape)
        motion_code_seq = self.motion_encoder(seqs)
        # print("motion_code_Seq.shape: ", motion_code_seq.size())
        return motion_code_seq

    def encode_body(self, seqs):
        body_code_seq = self.body_encoder(seqs)
        kernel_size = body_code_seq.size(-1)
        body_code = self.body_pool(body_code_seq, kernel_size)  if self.body_pool is not None else body_code_seq
        # print("before pool body_code shape: ", body_code.size())
        # print("after pool body_code shape: ", body_code_seq.size())
        return body_code, body_code_seq

    def encode_view(self, seqs):
        view_code_seq = self.view_encoder(seqs)
        kernel_size = view_code_seq.size(-1)
        view_code = self.view_pool(view_code_seq, kernel_size)  if self.view_pool is not None else view_code_seq
        # print("before pool view_code shape: ", view_code.size())
        # print("after pool view_code shape: ", view_code_seq.size())
        return view_code, view_code_seq

    def decode(self, motion_code, body_code, view_code):
        if body_code.size(-1) == 1:
            body_code = body_code.repeat(1, 1, motion_code.shape[-1])
        if view_code.size(-1) == 1:
            view_code = view_code.repeat(1, 1, motion_code.shape[-1])
        # print("[decode] body_code shape:", body_code.size())
        # print("[decode] view_code shape:", view_code.size())
        complete_code = torch.cat([motion_code, body_code, view_code], dim=1)
        out = self.decoder(complete_code)
        return out

