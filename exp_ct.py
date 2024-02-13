#!/usr/bin/env python

import os
import sys
import tqdm
from scipy import io
import argparse

import numpy as np
import wandb
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
plt.gray()

from skimage.metrics import structural_similarity as ssim_func

import torch
from torch.optim.lr_scheduler import LambdaLR

from modules import models
from modules import utils
from modules import lin_inverse

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='CT Reconstruction')

    # Add an argument
    parser.add_argument('-af', '--act-func', type=str, default='bspline-w', help='Activation function to use (wire, bspline-w)')
    parser.add_argument('-lr', '--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='LR decay rate')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--device', type=int, default=0, help='GPU to use')
    parser.add_argument('--width', type=int, default=256, help='Width for layers of MLP')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--lin-layers', action = argparse.BooleanOptionalAction, help='Adds linear layers in-between')
    parser.add_argument('--skip-conn', action = argparse.BooleanOptionalAction, help='Add skip connection')


    parser.add_argument('--meas', type=int, default=100, help='Number of projection measurements')
    
    parser.add_argument('--omega0', type=float, default=10, help='Global scaling for activation')
    parser.add_argument('--sigma0', type=float, default=10, help='Global scaling for activation')
    parser.add_argument('--init-scale', type=float, default=1, help='Initial scaling to apply to weights')
    
    parser.add_argument('--lam', type=float, default=0, help='Weight decay/Path Norm param')
    parser.add_argument('--path-norm', action=argparse.BooleanOptionalAction)
    parser.add_argument('--stop-loss', type=float, default=0, help='Stop at this loss')

    parser.add_argument('--weight-norm', action='store_true', help='Use weight normalization')
    parser.add_argument('--no-lr-scheduler', action='store_true', help='No LR scheduler')
    parser.add_argument('--rand-seed', type=int, default=40)
    parser.add_argument('--wandb', action='store_true', help='Turn on wandb')
    
    
    # Parse the arguments
    args = parser.parse_args()

    
    now = datetime.now().strftime("%m%d%Y%H%M")
    name = ('act_{}_lr_{}_epochs_{}_{}'
        .format(args.act_func, args.lr, args.epochs, now))
    
    if args.wandb:
        # start a new wandb run to track this script
        run = wandb.init(
            entity='activation-func',
            # set the wandb project where this run will be logged
            project="ct_wire_vs_bspline",
            name=name,
            # track hyperparameters and run metadata
            config=args,
            id=name
        )

    nonlin = args.act_func      # type of nonlinearity, 'bspline-w', 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = args.epochs            # Number of SGD iterations
    learning_rate = args.lr        # Learning rate. 
    device = 'cuda:{}'.format(args.device)
    lam = args.lam
    width = args.width
    layers = args.layers
    no_lr_scheduler = args.no_lr_scheduler

    save_dir = 'results/ct/{}_omega_{}_lr_{}_lam_{}_PN_{}_width_{}_layers_{}_lin_{}_skip_{}_epochs_{}_seed_{}_{}'.format(nonlin,
                                                                                                    args.omega0, 
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
    
    nmeas = args.meas                # Number of CT measurement
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3 

    
    # Gabor filter constants.
    omega0 = args.omega0    # Frequency of sinusoid or gloabal scaling B-spline wavelet
    sigma0= args.sigma0     # Sigma of Gaussian

    # Network parameters
    hidden_layers = layers       # Number of hidden layers in the MLP
    hidden_features = width   # Number of hidden units per layer
    
    # Generate sampling angles
    thetas = torch.tensor(np.linspace(0, 180, nmeas, dtype=np.float32)).to(device)

    # Create phantom
    img = cv2.imread('data/chest.png').astype(np.float32)[..., 1]
    img = utils.normalize(img, True)
    [H, W] = img.shape
    imten = torch.tensor(img)[None, None, ...].to(device)
    
    print('Image size is {}x{}'.format(H,W))
    
    # Create model
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
    else:
        posencode = False
    
    model = models.get_INR(
                    nonlin=nonlin,
                    in_features=2,
                    out_features=1, 
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=omega0,
                    hidden_omega_0=omega0,
                    scale=sigma0,
                    pos_encode=posencode,
                    sidelength=nmeas,
                    weight_norm=args.weight_norm,
                    init_scale = args.init_scale,
                    linear_layers = args.lin_layers,
                    skip_conn = args.skip_conn)
        
    model = model.to(device)
    print(model)
    
    # activation = {}
    # def get_activation_hook(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #     return hook

    # if nonlin == 'relu' or nonlin == 'siren' or nonlin=='wire':
    #     model.net[hidden_layers-1].register_forward_hook(get_activation_hook('last_feat'))
    # else:
    #     model.net[hidden_layers-1].act_func.register_forward_hook(get_activation_hook('last_feat'))
    
    with torch.no_grad():
        sinogram = lin_inverse.radon(imten, thetas).detach().cpu()
        sinogram = sinogram.numpy()
        
        # Set below to sinogram_noisy instead of sinogram to get noise in measurements
        sinogram_ten = torch.tensor(sinogram).to(device)
        
    x = torch.linspace(-1, 1, W).to(device)
    y = torch.linspace(-1, 1, H).to(device)
    
    X, Y = torch.meshgrid(x, y, indexing='xy')
        
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
        
    if args.path_norm:
        optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    else:
        optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=lam)

    
    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optimizer, lambda x: args.lr_decay**min(x/niters, 1))
    #scheduler = LambdaLR(optimizer, lambda x: 0.1**min(x/9000, niters/9000))

    
    best_loss = float('inf')
    loss_array = np.zeros(niters)
    lr_array = np.zeros(niters)
    loss_sinogram_array = np.zeros(niters)
    cond_num_array = np.zeros(niters)
    best_im = None
    path_norms_array = []
    
    increased_lr = False
    
    
    tbar = tqdm.tqdm(range(niters))
    for idx in tbar:
        # Estimate image       
        img_estim = model(coords).reshape(-1, H, W)[None, ...]
        
        # atoms_dict = activation['last_feat'].detach().squeeze().T
                
        # cond_num = torch.linalg.cond(atoms_dict)
        # cond_num_array[idx] = cond_num.item()
        
        # Compute sinogram
        sinogram_estim = lin_inverse.radon(img_estim, thetas)
        
        loss_sino = ((sinogram_ten - sinogram_estim)**2).mean()
        loss_sinogram_array[idx] = loss_sino.item()
        
        path_norm = 0

        if nonlin == 'bspline-w' and args.lin_layers:
            for l in range(hidden_layers):
               path_norm += omega0*torch.sum(torch.linalg.norm(model.net[l].block.linear.weight, dim=1) \
                                             * torch.linalg.norm(model.net[l].block.linear2.weight, dim=0))
            lam = args.lam
            path_norms_array.append(path_norm.item())
        
        elif nonlin=='bspline-w' and args.path_norm:
            for l in range(hidden_layers):
                if l == 0:
                    path_norm += omega0 * torch.sum(torch.linalg.norm(model.net[l].block.linear.weight, dim=1) \
                                             * torch.linalg.norm(model.net[l+1].block.linear.weight, dim=1))
                elif l > 1:
                    path_norm += omega0 * torch.sum(torch.linalg.norm(model.net[l].block.linear.weight, dim=1))
            
            path_norms_array.append(path_norm.item())

        if args.path_norm and args.lin_layers:
            loss_tot = loss_sino + lam*path_norm
        else:
            loss_tot = loss_sino
            
        
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()
        
        if not no_lr_scheduler:
            scheduler.step()
            lr_array[idx] = scheduler.get_last_lr()[0]
        else:
            lr_array[idx] = optimizer.param_groups[0]['lr']
        
        with torch.no_grad():
            img_estim_cpu = img_estim.detach().cpu().squeeze().numpy()
            if sys.platform == 'win32':
                cv2.imshow('Image', img_estim_cpu)
                cv2.waitKey(1)
            
            loss_gt = ((img_estim - imten)**2).mean()
            loss_array[idx] = loss_gt.item()
            
            if args.wandb:
                if idx % 100:
                    wandb.log({'loss_gt':loss_gt.item(), 'PSNR dB':-10*np.log10(loss_gt.item())})
            
            if loss_gt < best_loss:
                best_loss = loss_gt
                best_im = img_estim

            if nonlin == 'bspline-w' and args.path_norm:
                tbar.set_description('PSNR: {:2f},Loss: {:2e}, PN: {:2f}, LR: {:2e}'\
                                     .format(-10*np.log10(loss_array[idx]), loss_sino.item(), path_norm.item(), lr_array[idx]))
            else:
                tbar.set_description('PSNR: {:2f}, Loss: {:2e}, Learning Rate: {:2e}'.format(-10*np.log10(loss_array[idx]), loss_sino.item(), lr_array[idx]))                
            tbar.refresh()

            if loss_sino.item() < args.stop_loss:
                break
    
    img_estim_cpu = best_im.detach().cpu().squeeze().numpy()
    
    psnr2 = utils.psnr(img, img_estim_cpu)
    ssim2 = ssim_func(img, img_estim_cpu, data_range=1)
    
    np.save(os.path.join(save_dir, 'loss_array'), loss_array)
    np.save(os.path.join(save_dir, 'loss_sino_array'), loss_sinogram_array)
    np.save(os.path.join(save_dir, 'path_norm_array'), path_norms_array)
    np.save(os.path.join(save_dir, 'recon_img'), best_im.detach().cpu().numpy().squeeze())
    np.save(os.path.join(save_dir, 'orig_img'), imten.detach().cpu().numpy().squeeze())

    mdict = {'rec': best_im.detach().cpu().numpy().squeeze(),
             'gt': imten.detach().cpu().numpy().squeeze(),
             'mse_sinogram_array': loss_sinogram_array, 
             'mse_array': loss_array}
    
    io.savemat(os.path.join(save_dir, 'results.mat'), mdict)
    torch.save(model.state_dict(), os.path.join(save_dir,'model.pt'))

    # Contrast normalization
    best_im_np = best_im.detach().cpu().numpy().squeeze()
    best_im_cn = (best_im_np - np.min(best_im_np))/ (np.max(best_im_np) - np.min(best_im_np))

    plt.imsave(os.path.join(save_dir, 'recon.pdf'), best_im_cn, dpi=300)
    plt.imsave(os.path.join(save_dir, 'orig.pdf'), np.clip(imten.detach().cpu().numpy().squeeze(), 0, 1), dpi=300)
    

    print('PSNR: {:.1f} dB, SSIM: {:.3f}'.format(psnr2, ssim2))
    