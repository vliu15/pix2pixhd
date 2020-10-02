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

import argparse
from datetime import datetime

import os
import yaml
import torch
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf, ListConfig

import dataset
from loss import Pix2PixHDLoss
from utils import get_lr_lambda, weights_init, freeze_encoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--high_res', action='store_true', default=False)
    return parser.parse_args()


def parse_config(config):
    if isinstance(config.generator.in_channels, ListConfig):
        config.generator.in_channels = sum(config.generator.in_channels)
    if isinstance(config.discriminator.in_channels, ListConfig):
        config.discriminator.in_channels = sum(config.discriminator.in_channels)
    return config


def train(dataloaders, models, optimizers, schedulers, train_config, start_epoch, device, high_res):
    ''' Training loop for Pix2PixHD '''
    # unpack all modules
    train_dataloader, val_dataloader = dataloaders
    encoder, generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers

    # initialize logging
    loss = Pix2PixHDLoss(device=device)
    log_dir = os.path.join(train_config.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, mode=0o775, exist_ok=False)

    for epoch in range(start_epoch, train_config.epochs):
        mean_g_loss = 0.0
        mean_d_loss = 0.0

        # training epoch
        if not high_res:
            encoder.train()
        generator.train()
        discriminator.train()
        pbar = tqdm(train_dataloader, position=0, desc='[G loss: -.----][D loss: -.----]')
        for (x_real, labels, insts, bounds) in pbar:
            x_real = x_real.to(device)
            labels = labels.to(device)
            insts = insts.to(device)
            bounds = bounds.to(device)

            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                g_loss, d_loss, x_fake = loss(
                    x_real, labels, insts, bounds, encoder, generator, discriminator,
                )

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item() / len(train_dataloader)
            mean_d_loss += d_loss.item() / len(train_dataloader)

            pbar.set_description(desc=f'[G loss: {mean_g_loss:.4f}][D loss: {mean_d_loss:.4f}]')

        if train_config.save_every % epoch == 0:
            torch.save({
                'e_state_dict': encoder.state_dict(),
                'g_state_dict': generator.state_dict(),
                'd_state_dict': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(log_dir, f'epoch={epoch}.pt'))

        g_scheduler.step()
        d_scheduler.step()
        mean_g_loss = 0.0
        mean_d_loss = 0.0

        # validation epoch
        if not high_res:
            encoder.eval()
        generator.eval()
        discriminator.eval()
        pbar = tqdm(val_dataloader, position=0, desc='[G loss: -.----][D loss: -.----]')
        for (x_real, labels, insts, bounds) in pbar:
            x_real = x_real.to(device)
            labels = labels.to(device)
            insts = insts.to(device)
            bounds = bounds.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    g_loss, d_loss, x_fake = loss(
                        x_real, labels, insts, bounds, encoder, generator, discriminator,
                    )
            
            mean_g_loss += g_loss.item() / len(train_dataloader)
            mean_d_loss += d_loss.item() / len(train_dataloader)

            pbar.set_description(desc=f'[G loss: {mean_g_loss:.4f}][D loss: {mean_d_loss:.4f}]')


def main():
    args = parse_arguments()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)
        config = parse_config(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = instantiate(config.encoder).to(device).apply(weights_init).train()
    generator = instantiate(config.generator).to(device).apply(weights_init).train()
    discriminator = instantiate(config.discriminator).to(device).apply(weights_init).train()

    if args.high_res:
        g_optimizer = torch.optim.Adam(
            list(generator.parameters()), **config.optim,
        )
    else:
        g_optimizer = torch.optim.Adam(
            list(generator.parameters()) + list(encoder.parameters()), **config.optim,
        )
    d_optimizer = torch.optim.Adam(list(discriminator.parameters()), **config.optim)
    g_scheduler = torch.optim.lr_scheduler.LambdaLR(
        g_optimizer,
        get_lr_lambda(config.train.epochs, config.train.decay_after),
    )
    d_scheduler = torch.optim.lr_scheduler.LambdaLR(
        d_optimizer,
        get_lr_lambda(config.train.epochs, config.train.decay_after),
    )

    start_epoch = 0
    if config.resume_checkpoint is not None:
        state_dict = torch.load(config.resume_checkpoint)

        encoder.load_state_dict(state_dict['e_model_dict'])
        generator.load_state_dict(state_dict['g_model_dict'])
        discriminator.load_state_dict(state_dict['d_model_dict'])
        g_optimizer.load_state_dict(state_dict['g_optim_dict'])
        d_optimizer.load_state_dict(state_dict['d_optim_dict'])
        start_epoch = state_dict['epoch']

        msg = 'high-res' if args.high_res else 'low-res'
        print(f'Starting {msg} training from checkpoints')

    elif args.high_res:
        state_dict = config.pretrain_checkpoint
        if state_dict is not None:
            encoder.load_state_dict(torch.load(state_dict['e_model_dict']))
            encoder = freeze_encoder(encoder)
            generator.g1.load_state_dict(torch.load(state_dict['g_model_dict']))
            print('Starting high-res training from pretrained low-res checkpoints')
        else:
            print('Starting high-res training from scratch (no valid checkpoint detected)')

    else:
        print('Starting low-res training from random initialization')

    train_dataloader = torch.utils.data.DataLoader(
        instantiate(config.train_dataset),
        collate_fn=dataset.CityscapesDataset.collate_fn,
        **config.train_dataloader,
    )
    val_dataloader = torch.utils.data.DataLoader(
        instantiate(config.val_dataset),
        collate_fn=dataset.CityscapesDataset.collate_fn,
        **config.val_dataloader,
    )

    train(
        [train_dataloader, val_dataloader],
        [encoder, generator, discriminator],
        [g_optimizer, d_optimizer],
        [g_scheduler, d_scheduler],
        config.train, start_epoch, device,
    )


if __name__ == '__main__':
    main()
