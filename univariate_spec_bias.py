import torch
import numpy as np
from torch.fft import rfft, fft
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.nn.utils import clip_grad_value_, clip_grad_norm_


import os
import argparse
from scipy import io

import matplotlib.pyplot as plt
import wandb
from datetime import datetime

from modules import models


from tqdm import tqdm

parser = argparse.ArgumentParser(description='Univariate Experiments for Spectral Bias')
# Training settings
parser.add_argument('--wd', type=float, default=0, help='Weight Decay Regularization')
parser.add_argument('--lr', type=float, default=3e-3, help='Step size for GD')
parser.add_argument('--optim', type=str, default='sgd', help='Optimizer to use [sgd or adam]')
parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs for experiment')
parser.add_argument('--lr-scheduler', type=str, default='none', help='Learning rate scheduler to use [none or reduce_plateau or lambda_lr] (default: none)')

# Data setup
parser.add_argument('--low-freq', type=int, default=1, help='Lowest frequency to use')
parser.add_argument('--mid-freq', type=int, default=3, help='Middle frequency to use')
parser.add_argument('--high-freq', type=int, default=5, help='Highest frequency to use')
parser.add_argument('--n', type=int, default=201, help='Number of data points to use')

# Model settings
parser.add_argument('--act-func', type=str, default='relu', help='Activation function for NN [bspline-w, siren, wire, relu]')
parser.add_argument('--width', type=int, default=8000, help='Width of NN')
parser.add_argument('--num-layers', type=int, default=2, help='Number of layers to use')
parser.add_argument('--model', type=str, default='none', help='Type of deep net to use [mlp or resnet]')
parser.add_argument('--omega', type=float, default=30, help='Hyperparam for SIREN')
parser.add_argument('--weight-norm', action='store_true', help='Turn on weight normalization')

# System settings
parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu, cuda:0 or cuda:1)')

# Logging settings
parser.add_argument('--log-freq', type=int, default=1000, help='Logging freq for print statements')
parser.add_argument('--wandb', action='store_true', help='Turn on wandb')

args = parser.parse_args()
now = datetime.now().strftime("%m%d%Y%H%M")
name = ('act_{}_lr_{}_wd_{}_optim_{}_epochs_{}_width_{}_layers_{}_{}'
        .format(args.act_func, args.lr, args.wd, args.optim, args.epochs, args.width, args.num_layers, now))

if args.wandb:
    # start a new wandb run to track this script
    run = wandb.init(
        entity='activation-func',
        # set the wandb project where this run will be logged
        project="active-func",
        name=name,
        # track hyperparameters and run metadata
        config=args,
        id=name
    )


def train_univ_net(net, x_train, y_train, freq_1, freq_2, freq_3, epochs=100001, lr=1e-2, lam=0, rand_seed=49,
                   optim_name='adam', device='cpu', log_freqs=1000, use_wandb=False, lr_scheduler='none'):

    torch.manual_seed(rand_seed)

    # Add everything to right device
    net.to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)

    err = []
    cond_num_arr = []

    fft_1 = []
    fft_3 = []
    fft_5 = []
    
    gradient_norms = []
    gradient_norms_outweights = []
    gradient_norms_hidden = []

    criterion = torch.nn.MSELoss()

    # pick optimization, a good optimizer
    if optim_name == 'adam':
        optim_func = torch.optim.Adam
    elif optim_name == 'sgd':
        optim_func = torch.optim.SGD
    else:
        raise NotImplementationError("Undefined optim_name")

    # optim param
    optimizer = optim_func(net.parameters(), lr=lr, weight_decay=lam)

    if lr_scheduler == 'reduce_plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True, min_lr=1e-6, threshold=1e-4)

    if lr_scheduler == 'lambda_lr':
        # Schedule to 0.1 times the initial rate
        scheduler = LambdaLR(optimizer, lambda x: 0.1**min(x/epochs, 1))
                             
    for t in (pbar := tqdm(range(epochs))):
        y_hat = net(x_train)
        atoms_dict = activation['last_feat'].detach().squeeze().T
        loss = criterion(y_hat, y_train)
        optimizer.zero_grad()
        loss.backward()
        
        # Calculate the total norm of the gradients
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float('inf'))
        gradient_norms.append(total_norm.item())
        
        norm_out_weights = torch.nn.utils.clip_grad_norm_(net.net[hidden_layers].parameters(), float('inf'))
        gradient_norms_outweights.append(norm_out_weights.item())

        norm_hidden = torch.nn.utils.clip_grad_norm_(net.net[:hidden_layers].parameters(), float('inf'))
        gradient_norms_hidden.append(norm_hidden.item())
        
        optimizer.step()
        if lr_scheduler != 'none':
            scheduler.step(loss)

        u_fft = torch.abs(rfft(y_train.flatten()))
        y_fft = torch.abs(rfft(y_hat.flatten().clone().detach()))

        # k = 1
        fft_1_data = (torch.abs(u_fft[freq_1] - y_fft[freq_1]) / torch.abs(u_fft[freq_1])).item()
        fft_1.append(fft_1_data)

        # k = 3
        fft_3_data = (torch.abs(u_fft[freq_2] - y_fft[freq_2]) / torch.abs(u_fft[freq_2])).item()
        fft_3.append(fft_3_data)

        # k = 5
        fft_5_data = (torch.abs(u_fft[freq_3] - y_fft[freq_3]) / torch.abs(u_fft[freq_3])).item()
        fft_5.append(fft_5_data)

        #print(h_relu)
        cond_num = torch.linalg.cond(atoms_dict)
        # cond_num = torch.as_tensor(0)
        
        pbar.set_description('{:.2e}'.format(loss.item()))
        pbar.refresh()

        err.append(loss.item())
        cond_num_arr.append(cond_num.item())
        if use_wandb:
            wandb.log({'fft_1': fft_1_data, 'fft_3': fft_3_data, 'fft_5': fft_5_data, 'loss': loss.item(),
                       'cond_num': cond_num})

        # if t % log_freqs == 0:
        #     print('Iter:{}, Loss: {}, Cond Num: {}'.format(t, loss.item(), cond_num.item()))
        #     print('FFT_1 Loss: {}'.format(fft_1_data))
        #     print('FFT_2 Loss: {}'.format(fft_3_data))
        #     print('FFT_3 Loss: {}'.format(fft_5_data), end='\n\n')

    return err, cond_num_arr, fft_1, fft_3, fft_5, gradient_norms, gradient_norms_outweights, gradient_norms_hidden


# Generate samples
freq_1 = torch.as_tensor(args.low_freq)
freq_2 = torch.as_tensor(args.mid_freq)
freq_3 = torch.as_tensor(args.high_freq)

n = args.n
x = torch.linspace(-np.pi, np.pi, n)

y = torch.sin(freq_1 * x) + torch.sin(freq_2 * x) + torch.sin(freq_3 * x)

data_train = torch.reshape(x, (n, 1))
resp_train = torch.reshape(y, (n, 1))

rand_seed = 50
W = args.width
device=args.device

total_iters = args.epochs
lr = args.lr
wd = args.wd
nonlin = args.act_func
omega0 = args.omega
optimizer_name = args.optim

hidden_layers = args.num_layers
torch.manual_seed(rand_seed)

weight_norm = args.weight_norm


model = models.get_INR(
                    nonlin=nonlin,
                    in_features=1,
                    out_features=1, 
                    hidden_features=W,
                    hidden_layers=hidden_layers,
                    first_omega_0=omega0,
                    hidden_omega_0=omega0,
                    weight_norm=weight_norm)

        
model = model.to(device)

activation = {}
def get_activation_hook(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
    
if nonlin == 'relu' or nonlin == 'siren' or nonlin=='wire':
    model.net[hidden_layers].register_forward_hook(get_activation_hook('last_feat'))
else:
    model.net[hidden_layers].act_func.register_forward_hook(get_activation_hook('last_feat'))

print(model)


err_list, cond_list, fft_1_list, fft_3_list, fft_5_list, gradient_norms, gradient_norms_outweights, gradient_hidden \
    = train_univ_net(model, data_train, resp_train, freq_1, freq_2, freq_3, epochs=total_iters, lr=lr,
                     lam=wd, rand_seed=rand_seed, optim_name=optimizer_name, log_freqs=args.log_freq,
                     use_wandb=args.wandb, device=args.device, lr_scheduler=args.lr_scheduler)

# Creating the plots
model.eval()
data_train = data_train.to(args.device)
y_f = model(data_train).detach().cpu().numpy()
y_f_dft = torch.abs(rfft(torch.from_numpy(y_f).flatten()))
x_extrapolate = torch.linspace(-6, 6, 2000).reshape(2000, 1).to(args.device)
y_extrapolate = model(x_extrapolate).detach().cpu().numpy()
#

# Plot 1
plt.figure(figsize=(8,8))
plt.title("Condition Number")
plt.semilogy(cond_list, label='condition number')
plt.ylabel("condition number")
plt.xlabel("epochs")
plt.legend()
if args.wandb:
    wandb.log({"Condition Number": wandb.Image(plt)})
else:
    plt.show()
plt.close()

# Plot 2
plt.figure(figsize=(8,8))
plt.title("frequency error")
plt.semilogy(fft_1_list, label="frequency {}".format(freq_1))
plt.semilogy(fft_3_list, label="frequency {}".format(freq_2))
plt.semilogy(fft_5_list, label="frequency {}".format(freq_3))
plt.yscale("log")
plt.ylabel("error")
plt.xlabel("epochs")
plt.legend()
if args.wandb:
    wandb.log({"Frequency Error": wandb.Image(plt)})
else:
    plt.show()
plt.close()

# Plot 3
plt.figure(figsize=(8,8))
plt.title("learned function")
plt.plot(x, y_f, label='learned function')
plt.scatter(x, y, label='origin')
plt.ylabel("y")
plt.xlabel("x")
plt.legend()
if args.wandb:
    wandb.log({"Learned Function": wandb.Image(plt)})
else:
    plt.show()
plt.close()

# Plot 4
plt.figure(figsize=(8,8))
plt.title("learned function outer")
plt.plot(x_extrapolate.cpu(), y_extrapolate, label='learned function')
plt.scatter(x, y, label='original')
plt.ylabel("y")
plt.xlabel("x")
plt.legend()
if args.wandb:
    wandb.log({"Learned Function Outer": wandb.Image(plt)})
else:
    plt.show()
plt.close()

# Plot 5
plt.figure(figsize=(8,8))
plt.title("DFT of the learned nn")
plt.plot(y_f_dft, label='DFT')
plt.ylabel("amplitude")
plt.xlabel("frequency")
plt.legend()
if args.wandb:
    wandb.log({"DFT of the Learned NN": wandb.Image(plt)})
else:
    plt.show()
plt.close()

# Plot 6
plt.figure(figsize=(8,8))
plt.title("Loss")
plt.semilogy(err_list, label='loss')
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend()
if args.wandb:
    wandb.log({"Loss": wandb.Image(plt)})
else:
    plt.show()
plt.close()

# Plot 7
plt.figure(figsize=(8,8))
plt.semilogy(gradient_norms)
plt.xlabel('Training Steps')
plt.ylabel('Gradient Norm Total')
plt.title('Gradient Norm During Training')
plt.show()
plt.close()


# Plot 8
plt.figure(figsize=(8,8))
plt.semilogy(gradient_norms_outweights)
plt.xlabel('Training Steps')
plt.ylabel('Gradient Norm Output Weights')
plt.title('Gradient Norm During Training')
plt.show()
plt.close()

# Plot 9
plt.figure(figsize=(8,8))
plt.semilogy(gradient_hidden)
plt.xlabel('Training Steps')
plt.ylabel('Gradient Norm Hidden Weights')
plt.title('Gradient Norm During Training')
plt.show()
plt.close()

err_list, cond_list, fft_1_list, fft_3_list, fft_5_list

mdict = {'tot_loss': err_list,
         'cond_num': cond_list,
         'fft_1_loss': fft_1_list,
         'fft_2_loss': fft_3_list,
         'fft_3_loss': fft_5_list
         }

os.makedirs('results/univ_spectral_bias', exist_ok=True)
io.savemat('results/univ_spectral_bias/%s.mat'%(nonlin), mdict)

if args.wandb:
    torch.save(model.state_dict(), "model.pth")
    wandb.save("model.pth")

