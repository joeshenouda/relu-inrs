#!/usr/bin/env python

import os
import sys
import tqdm
from scipy import io
import argparse
import time
import wget

import numpy as np
import wandb
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
plt.gray()

from skimage.metrics import structural_similarity as ssim_func

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from modules import models
from modules import utils

import scipy.io.wavfile as wavfile
import io
from IPython.display import Audio

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Fitting Audio Signal')

    # Training arguments
    parser.add_argument('-lr', '--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='LR decay rate')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--device', type=int, default=0, help='GPU to use')
    parser.add_argument('--lam', type=float, default=0, help='Weight decay/Path Norm param')
    parser.add_argument('--path-norm', action=argparse.BooleanOptionalAction)
    parser.add_argument('--stop-loss', type=float, default=0, help='Stop at this loss')
    parser.add_argument('--no-lr-scheduler', action='store_true', help='No LR scheduler')
    parser.add_argument('--learnable_c', action=argparse.BooleanOptionalAction, help='change c into a param or not')

    # Model arguments
    parser.add_argument('-af', '--act-func', type=str, default='bspline-w',
                        help='Activation function to use (wire, bspline-w, siren, relu, posenc)')
    parser.add_argument('--omega0', type=float, default=10, help='Global scaling for activation')
    parser.add_argument('--sigma0', type=float, default=10, help='Global scaling for activation')
    parser.add_argument('--c', type=float, default=1.0, help='Global scaling for BW-ReLU activation')
    parser.add_argument('--init-scale', type=float, default=1, help='Initial scaling to apply to weights')
    parser.add_argument('--width', type=int, default=256, help='Width for layers of MLP')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--lin-layers', action=argparse.BooleanOptionalAction, help='Adds linear layers in-between')
    parser.add_argument('--skip-conn', action=argparse.BooleanOptionalAction, help='Add skip connection')

    # Experiment arguments
    parser.add_argument('--rand-seed', type=int, default=40)
    parser.add_argument('--wandb', action='store_true', help='Turn on wandb')

    # Parse the arguments
    args = parser.parse_args()

    if args.wandb:
        now = datetime.now().strftime("%m%d%Y%H%M")
        name = ('act_{}_lr_{}_epochs_{}_{}'
                .format(args.act_func, args.lr, args.epochs, now))
        if args.act_func == 'bspline-w':
            name = ('act_{}_c_{}_lr_{}_epochs_{}_{}'
                    .format(args.act_func, args.c, args.lr, args.epochs, now))
        elif args.act_func == 'wire':
            name = ('act_{}_omega0_{}_sigma0_{}_lr_{}_epochs_{}_{}'
                    .format(args.act_func, args.omega0, args.sigma0, args.lr, args.epochs, now))
        elif args.act_func == 'siren':
            name = ('act_{}_omega0_{}_lr_{}_epochs_{}_{}'
                    .format(args.act_func, args.omega0, args.lr, args.epochs, now))
        project_name = 'audio-signal-fitting'
        # start a new wandb run to track this script
        run = wandb.init(
            entity='activation-func',
            # set the wandb project where this run will be logged
            project=project_name,
            name=name,
            # track hyperparameters and run metadata
            config=args,
            id=name
        )

    nonlin = args.act_func  # type of nonlinearity, 'bspline-w', 'wire', 'siren', 'relu', 'posenc'
    niters = args.epochs  # Number of SGD iterations
    learning_rate = args.lr  # Learning rate.
    device = 'cuda:{}'.format(args.device)
    lam = args.lam
    width = args.width
    layers = args.layers
    no_lr_scheduler = args.no_lr_scheduler

    # If you want the original values for omega0 and sigma0 use default of the args
    omega0 = args.omega0          # Frequency of sinusoid
    sigma0 = args.sigma0          # Sigma of Gaussian
    c = args.c  # Scaling for BW-ReLU

    # Network parameters
    hidden_layers = args.layers  # Number of hidden layers in the MLP
    hidden_features = args.width  # Number of hidden units per layer
    device = 'cuda:{}'.format(args.device)

    save_dir = 'results/audio/{}_c_{}_omega_{}_sigma_{}_lr_{}_lam_{}_PN_{}_width_{}_layers_{}_lin_{}_skip_{}_epochs_{}_seed_{}_{}'.format(
                                                                                    nonlin,
                                                                                    args.c,
                                                                                    args.omega0,
                                                                                    args.sigma0,
                                                                                    learning_rate,
                                                                                    args.lam,
                                                                                    args.path_norm,
                                                                                    args.width,
                                                                                    args.layers,
                                                                                    args.lin_layers,
                                                                                    args.skip_conn,
                                                                                    args.epochs,
                                                                                    args.rand_seed,
                                                                                    datetime.now().strftime('%m%d_%H%M'))
    os.makedirs(save_dir, exist_ok=True)

    ## Random seed
    torch.manual_seed(args.rand_seed)


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

if not os.path.exists('data/gt_bach.wav'):
    url_audio = "https://vsitzmann.github.io/siren/img/audio/gt_bach.wav"
    wget.download(url_audio, 'data/gt_bach.wav')

# Dataset for handling audio files
class AudioFile(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.rate, self.data = wavfile.read(filename)
        self.data = self.data.astype(np.float32)
        self.timepoints = get_mgrid(len(self.data), 1)

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        amplitude = self.data
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude).view(-1, 1)
        return self.timepoints, amplitude

bach_audio = AudioFile('data/gt_bach.wav')

dataloader = DataLoader(bach_audio, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

if nonlin == 'posenc':
    nonlin = 'relu'
    posencode = True
else:
    posencode = False

torch.manual_seed(args.rand_seed)

model = models.get_INR(
    nonlin=nonlin,
    in_features=1,
    out_features=1,
    hidden_features=hidden_features,
    hidden_layers=hidden_layers,
    first_omega_0=omega0,
    hidden_omega_0=omega0,
    c=c,
    scale=sigma0,
    pos_encode=posencode,
    sidelength=2,
    init_scale=args.init_scale,
    linear_layers=args.lin_layers)

print(model)

# Send model to CUDA
model.to(device)

## Train
total_steps = args.epochs
steps_til_summary = 100

# Create an optimizer
if args.path_norm:
    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
else:
    print('Adding weight decay param lam:{}!'.format(args.lam))
    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=args.lam)

# Schedule to 0.1 times the initial rate
scheduler = LambdaLR(optim, lambda x: args.lr_decay ** min(x / niters, 1))

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.to(device), ground_truth.to(device)

for step in range(total_steps):
    model_output = model(model_input)
    loss = F.mse_loss(model_output, ground_truth)

    print("Step {}, Total loss {}".format(step, loss))
    if args.wandb:
        wandb.log({'loss': loss.item()})
        # log learning rate
        wandb.log({'learning_rate': scheduler.get_last_lr()[0]})

        # fig, axes = plt.subplots(1, 2)
        # axes[0].plot(coords.squeeze().detach().cpu().numpy(), model_output.squeeze().detach().cpu().numpy())
        # axes[1].plot(coords.squeeze().detach().cpu().numpy(), ground_truth.squeeze().detach().cpu().numpy())
        # plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()
    scheduler.step()

final_model_output = model(model_input)
rate, _ = wavfile.read('data/gt_bach.wav')
os.makedirs('results/audio', exist_ok=True)
bspline_nn_audio = wavfile.write('results/audio/bspline_nn_audio.wav', rate, final_model_output.squeeze().detach().cpu().numpy())


