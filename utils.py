# The MIT License
#
# Copyright (c) 2020 Vincent Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from omegaconf import ListConfig


def show_tensor_images(image_tensor, show_n=1):
    ''' For visualizing images '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:show_n], nrow=show_n)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def parse_config(config):
    ''' Parses any unaddressed fields from yaml config '''
    if isinstance(config.generator.in_channels, ListConfig):
        config.generator.in_channels = sum(config.generator.in_channels)
    if isinstance(config.discriminator.in_channels, ListConfig):
        config.discriminator.in_channels = sum(config.discriminator.in_channels)
    return config


def get_lr_lambda(epochs, decay_after):
    ''' Function for scheduling learning '''
    def lr_lambda(epoch):
        return 1. if epoch < decay_after else 1 - float(epoch - decay_after) / (epochs - decay_after)
    return lr_lambda


def weights_init(m):
    ''' Initializes all model weights to N(0, 0.02) '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0., 0.02)


def freeze_encoder(encoder):
    ''' Freezes encoder weights and wraps it for high-res images '''
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    @torch.jit.script
    def forward(x, inst):
        x = F.interpolate(x, scale_factor=0.5, recompute_scale_factor=True)
        inst = F.interpolate(inst.float(), scale_factor=0.5, recompute_scale_factor=True)
        feat = encoder(x, inst.int())
        return F.interpolate(feat, scale_factor=2.0, recompute_scale_factor=True)

    return forward
