#!/usr/bin/env python

import os
import sys
from tqdm import tqdm
import importlib
import argparse


import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import matplotlib.pyplot as plt
import cv2

from pytorch_msssim import ssim

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR

from modules import models
from modules import utils
from modules import wire

from datetime import datetime

models = importlib.reload(models)
    
if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='SuperRes')

    # Add an argument
    parser.add_argument('-af', '--act-func', type=str, default='wire', help='Activation function to use (wire, bspline-w, relu)')
    
    # Image arguments
    parser.add_argument('--scale', type=int, default=4, help='Downsampling scale')
    parser.add_argument('--scale-im', type=int, default=3, help='Initial image downsample for memory reasons')
    parser.add_argument('--image', type=str, default='butterfly', help='Which pic [butterfly, rhino, nemo]')

    # Model arguments
    parser.add_argument('--omega0', type=float, default=8.0, help='Global scaling for activation')
    parser.add_argument('--sigma0', type=float, default=6.0, help='Global scaling for activation')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--width', type=int, default=256, help='Width for layers of MLP')
    parser.add_argument('--lin-layers', action=argparse.BooleanOptionalAction)

    # Training arguments
    parser.add_argument('-lr', '--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of iterations')
    parser.add_argument('--lam', type=float, default=0, help='Weight decay/Path Norm param')
    parser.add_argument('--path-norm', action=argparse.BooleanOptionalAction)
    parser.add_argument('--weight-norm', action='store_true', help='Use weight normalization')
    parser.add_argument('--holdout', type=float, default=0, help='Number of points to use for holdout')
    parser.add_argument('--device', type=int, default=0, help='GPU to use')
    
    
    
    # Parse the arguments
    args = parser.parse_args()
    
    nonlin = args.act_func            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = args.epochs               # Number of SGD iterations
    learning_rate = args.lr        # Learning rate. 
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3 
    scale = args.scale                   # Downsampling factor

    #scale_im = 1/3
    scale_im = 1/args.scale_im              # Initial image downsample for memory reasons

    # Gabor filter constants,
    # If you want the oirignal values for omega0 and sigma0 use default of the args
    omega0 = args.omega0          # Frequency of sinusoid
    sigma0 = args.sigma0          # Sigma of Gaussian
    
    # Network parameters
    hidden_layers = args.layers    # Number of hidden layers in the MLP
    hidden_features = args.width   # Number of hidden units per layer
    device = 'cuda:{}'.format(args.device)


    # Read image
    im = utils.normalize(plt.imread('data/{}.png'.format(args.image)).astype(np.float32), True)

    # This is just an initial downscale operation that we do for memory reasons
    im = cv2.resize(im, None, fx=scale_im, fy=scale_im, interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape
    
    im = im[:scale*(H//scale), :scale*(W//scale), :]
    H, W, _ = im.shape

    # True low resolution image
    im_lr = cv2.resize(im, None, fx=1/scale, fy=1/scale,
                       interpolation=cv2.INTER_AREA)
    H2, W2, _ = im_lr.shape

    # This just serves as a benchmark for our method below.
    im_bi = cv2.resize(im_lr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_LINEAR)
    
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
        sidelength = int(max(H, W))
    else:
        posencode = False
        sidelength = H

    torch.manual_seed(5)
    model = models.get_INR(
                    nonlin=nonlin,
                    in_features=2,
                    out_features=3, 
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=omega0,
                    hidden_omega_0=omega0,
                    scale=sigma0,
                    pos_encode=posencode,
                    sidelength=sidelength,
                    weight_norm=args.weight_norm,
                    linear_layers=args.lin_layers)
    
    activation = {}
    def get_activation_hook(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    if nonlin == 'relu' or nonlin == 'siren' or nonlin=='wire':
        model.net[hidden_layers-1].register_forward_hook(get_activation_hook('last_feat'))
    else:
        model.net[hidden_layers-1].bspline_w.register_forward_hook(get_activation_hook('last_feat'))

    print(model)
        
    # Send model to CUDA
    model.to(device)
    
    # Create an optimizer
    if args.path_norm:
        optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    else:
        print('Adding weight decay param!')
        optim = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=args.lam)

    
    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optim, lambda x: 0.2**min(x/niters, 1))
    
    x = torch.linspace(-1, 1, W2).to(device)
    y = torch.linspace(-1, 1, H2).to(device)
    
    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords_lr = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
    
    x_hr = torch.linspace(-1, 1, W).to(device)
    y_hr = torch.linspace(-1, 1, H).to(device)
    
    X_hr, Y_hr = torch.meshgrid(x_hr, y_hr, indexing='xy')
    coords_hr = torch.hstack((X_hr.reshape(-1, 1), Y_hr.reshape(-1, 1)))[None, ...]
    
    gt = torch.tensor(im).to(device).reshape(H*W, 3)[None, ...]
    gt_lr = torch.tensor(im_lr).to(device).reshape(H2*W2, 3)[None, ...]
    
    im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
    im_bi_ten = torch.tensor(im_bi).to(device).permute(2, 0, 1)[None, ...]
    
    # print(utils.psnr(im, im_bi),
    #       ssim_func(im, im_bi, multichannel=True))
    print(utils.psnr(im, im_bi))
    
    mse_hr_array = torch.zeros(niters, device=device) # high res mse (test error)
    mse_lr_array = torch.zeros(niters, device=device) # low res mse  (train error)
    mse_val_array = torch.zeros(niters, device=device)
    ssim_array = torch.zeros(niters, device=device)
    lpips_array = torch.zeros(niters, device=device)
    
    cond_num_iters = []
    cond_num_array = []
    path_norms_array = []
    
    best_mse = float('inf')
    best_img = None
    
    downsampler = torch.nn.AvgPool2d(scale)

    save_dir = 'results/sisr/{}_SR_img_scale_{}_omega_{}_lr_{}_lam_{}_PN_{}_width_{}_layers_{}_epochs_{}_{}'.format(nonlin,
                                                                                                        scale_im, 
                                                                                                        omega0, 
                                                                                                        learning_rate, 
                                                                                                        args.lam,  
                                                                                                        args.path_norm, 
                                                                                                        args.width, 
                                                                                                        args.layers, 
                                                                                                        args.epochs,         
                                                                                                        datetime.now().strftime('%m%d_%H%M'))
    os.makedirs(save_dir, exist_ok=True)

    tbar = tqdm(range(niters))
    for epoch in tbar:
        #rec = model(coords)
        
        rec_hr = model(coords_hr)
        atoms_dict = activation['last_feat'].detach().squeeze().T
        rec = downsampler(rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...])

        if args.lin_layers:
            path_norm = 0
            for l in range(hidden_layers):
               path_norm += omega0*torch.sum(torch.linalg.norm(model.net[l].block.linear.weight, dim=1) * torch.linalg.norm(model.net[l].block.linear2.weight, dim=0))
            lam = args.lam
            path_norms_array.append(path_norm.item())

        if args.holdout > 0:
            p = args.holdout

            rand_perm = torch.randperm(coords_lr.size(1))
            idx_train = rand_perm[:int(coords_lr.size(1)*p)]
            idx_val = rand_perm[int(coords_lr.size(1)*p):]
            

            loss = ((gt_lr - rec.reshape(1, 3, -1).permute(0, 2, 1))[:, idx_train,:]**2).mean()

            mse_lr_array[epoch] = loss.item()

            with torch.no_grad():                    
                gt_lr_val = gt_lr[:, idx_val,:]
                
                val_loss = ((gt_lr - rec.reshape(1, 3, -1).permute(0, 2, 1))[:, idx_val,:]**2).mean()
                mse_val_array[epoch] = val_loss.item()

            with torch.no_grad():
                rec_hr = model(coords_hr)
                
                im_rec = rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
                
                mse_hr_array[epoch] = ((gt - rec_hr)**2).mean().item()
                # ssim_array[epoch] = ssim(im_gt, im_rec, data_range=1,
                #                          size_average=True)
            
            tbar.set_description('Loss: {:e}, Val Loss: {:e}'.format(loss.item(), val_loss.item()))
            tbar.refresh()
        
        else:
            loss = ((gt_lr - rec.reshape(1, 3, -1).permute(0, 2, 1))**2).mean()
            with torch.no_grad():
                rec_hr = model(coords_hr)
                
                im_rec = rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
                
                mse_hr_array[epoch] = ((gt - rec_hr)**2).mean().item()
                # ssim_array[epoch] = ssim(im_gt, im_rec, data_range=1,
                #                          size_average=True)
        
        
            mse_lr_array[epoch] = loss.item()

            if args.bottleneck:
                tbar.set_description('{:2f}: Loss LR {:e}, Path Norm: {}'.format(-10*torch.log10(mse_hr_array[epoch]),loss.item(), path_norm.item()))
            else:
                tbar.set_description('{:2f}: Loss LR {:e}'.format(-10*torch.log10(mse_hr_array[epoch]),loss.item()))
            
            tbar.refresh()
            
            
        if epoch % 200 == 0:
            cond_num = torch.linalg.cond(atoms_dict)
            cond_num_iters.append(epoch)
            cond_num_array.append(cond_num.item())
        
        if epoch % 10000 == 0:
            np.save(os.path.join(save_dir, 'mse_hr_array_t_{}'.format(epoch)), mse_hr_array.cpu().numpy())
            np.save(os.path.join(save_dir, 'mse_lr_array_t_{}'.format(epoch)), mse_lr_array.cpu().numpy())
            np.save(os.path.join(save_dir, 'path_norm_array_t_{}'.format(epoch)), path_norms_array)    
            # io.savemat(os.path.join(save_dir, 'results.mat'), mdict)
            torch.save(model.state_dict(), os.path.join(save_dir,'model_t_{}.pt'.format(epoch)))
                

            
        imrec = im_rec.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        
        if sys.platform == 'win32':
            cv2.imshow('Reconstruction', imrec[..., ::-1])
            cv2.waitKey(1)
        
        if mse_hr_array[epoch] < best_mse:
            best_mse = mse_hr_array[epoch]
            best_img = imrec

        if args.path_norm:
            loss = loss+lam*path_norm
        
        
        loss.backward()
        optim.step()
        scheduler.step()
        optim.zero_grad()
        

    if posencode:
        nonlin = 'posenc'
        
    mdict = {'rec': best_img,
             'gt': im,
             'rec_bi': im_bi,
             'mse_hr_array': mse_hr_array.cpu().numpy(),
             'mse_lr_array':mse_lr_array.cpu().numpy(),
             'mse_val_array': mse_val_array.cpu().numpy(),
             'cond_num_iters': cond_num_iters,
             'cond_num_array': cond_num_array,
             'path_norm_array':path_norms_array}


    np.save(os.path.join(save_dir, 'mse_hr_array'), mse_hr_array.cpu().numpy())
    np.save(os.path.join(save_dir, 'mse_lr_array'), mse_lr_array.cpu().numpy())
    np.save(os.path.join(save_dir, 'cond_num_arr'), cond_num_array)
    np.save(os.path.join(save_dir, 'conda_num_iters'), cond_num_iters)
    np.save(os.path.join(save_dir, 'path_norm_array'), path_norms_array)    
    # io.savemat(os.path.join(save_dir, 'results.mat'), mdict)
    torch.save(model.state_dict(), os.path.join(save_dir,'model.pt'))
    
    # print(-10*torch.log10(best_mse),
    #       ssim_func(im, best_img, multichannel=True),
    #       lpips_array.min().item())
    print(-10*torch.log10(best_mse))
    
    plt.imsave(os.path.join(save_dir, 'SR_diff_%s.png'%nonlin),
               np.clip(abs(im - best_img), 0, 1),
               vmin=0.0,
               vmax=0.1)

    plt.imsave(os.path.join(save_dir, 'recon.png'), np.clip(imrec, 0, 1))
    plt.imsave(os.path.join(save_dir, 'im_lr.png'), np.clip(im_lr, 0, 1))
    plt.imsave(os.path.join(save_dir, 'im_hr.png'), np.clip(im, 0, 1))

    
    if args.holdout == 0:
        plt.plot(-10*torch.log10(mse_hr_array).detach().cpu().numpy())
        plt.show()
