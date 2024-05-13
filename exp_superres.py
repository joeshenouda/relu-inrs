#!/usr/bin/env python

import os
from tqdm import tqdm
import importlib
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
import skimage
import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import LambdaLR
from modules import models
from modules import utils
from datetime import datetime
import wandb

models = importlib.reload(models)

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='SuperRes')

    # Model arguments
    parser.add_argument('-af', '--act-func', type=str, default='wire',
                        help='Activation function to use (wire, bspline-w, relu)')
    parser.add_argument('--c', type=float, default=1.0, help='Global scaling for BW-ReLU activation')
    parser.add_argument('--omega0', type=float, default=8.0, help='Global scaling for SIREN, WIRE activation')
    parser.add_argument('--sigma0', type=float, default=6.0, help='Global scaling for WIRE activation')
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--width', type=int, default=256, help='Width for layers of MLP')
    parser.add_argument('--lin-layers', action=argparse.BooleanOptionalAction)

    # Image arguments
    parser.add_argument('--scale', type=int, default=4, help='Downsampling scale')
    parser.add_argument('--scale-im', type=int, default=3, help='Initial image downsample for memory reasons')
    parser.add_argument('--image', type=str, default='butterfly', help='Which pic [butterfly, rhino, nemo]')

    # Training arguments
    parser.add_argument('-lr', '--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='LR decay rate')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of iterations')
    parser.add_argument('--lam', type=float, default=0, help='Weight decay/Path Norm param')
    parser.add_argument('--path-norm', action=argparse.BooleanOptionalAction)
    parser.add_argument('--stop-loss', type=float, default=0, help='Stop at this loss')
    parser.add_argument('--device', type=int, default=0, help='GPU to use')
    parser.add_argument('--rand-seed', type=int, default=40)
    parser.add_argument('--wandb', action='store_true', help='Turn on wandb')

    # Parse the arguments
    args = parser.parse_args()
    
    nonlin = args.act_func         # type of nonlinearity, 'wire', 'siren', 'relu', 'posenc'
    niters = args.epochs           # Number of SGD iterations
    learning_rate = args.lr        # Learning rate. 

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
        project_name = 'superresolution'
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

    scale = args.scale                   # Downsampling factor

    #scale_im = 1/3
    scale_im = 1/args.scale_im              # Initial image downsample for memory reasons

    # Gabor filter constants,
    omega0 = args.omega0          # Frequency of sinusoid
    sigma0 = args.sigma0          # Sigma of Gaussian

    # For BW-ReLu
    c = args.c
    
    # Network parameters
    hidden_layers = args.layers    # Number of hidden layers in the MLP
    hidden_features = args.width   # Number of hidden units per layer
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'

    save_dir = 'results/superres/{}_{}_SR_img_scale_{}_c_{}_omega_{}_sigma_{}_lr_{}_lam_{}_PN_{}_width_{}_layers_{}_lin_{}_epochs_{}'\
        .format(datetime.now().strftime('%m%d_%H%M'),
        nonlin,
        scale_im,
        c,
        omega0,
        sigma0,
        learning_rate,
        args.lam,
        args.path_norm,
        args.width,
        args.layers,
        args.epochs,
        args.lin_layers)
    os.makedirs(save_dir, exist_ok=True)

    # Read image
    im = utils.normalize(plt.imread('data/{}.png'.format(args.image)).astype(np.float32), True)
    H, W, _ = im.shape
    # This is just an initial downscale operation that we do for memory reasons
    im = skimage.transform.resize(im, (H//args.scale_im, W//args.scale_im), anti_aliasing=True)
    H, W, _ = im.shape
    
    im = im[:scale*(H//scale), :scale*(W//scale), :]
    H, W, _ = im.shape

    # True low resolution image
    im_lr = skimage.transform.resize(im, (H//scale, W//scale), anti_aliasing=True)
    H2, W2, _ = im_lr.shape

    
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
        sidelength = int(max(H, W))
    else:
        posencode = False
        sidelength = H

    torch.manual_seed(args.rand_seed)
    model = models.get_INR(
                    nonlin=nonlin,
                    in_features=2,
                    out_features=3, 
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=omega0,
                    hidden_omega_0=omega0,
                    c=c,
                    scale=sigma0,
                    pos_encode=posencode,
                    sidelength=sidelength,
                    linear_layers=args.lin_layers)

    print(model)
        
    # Send model to CUDA
    model.to(device)
    
    # Create an optimizer
    if args.path_norm:
        optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    else:
        print('Adding weight decay param lam:{}!'.format(args.lam))
        optim = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=args.lam)

    
    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optim, lambda x: args.lr_decay**min(x/niters, 1))
    
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

    mse_hr_array = [] # high res mse (test error)
    mse_lr_array = [] # low res mse  (train error)
    ssim_array = []

    cond_num_iters = []
    cond_num_array = []
    path_norms_array = []
    learning_rates = []
    
    best_mse = float('inf')
    best_img = None
    
    downsampler = torch.nn.AvgPool2d(scale)

    # Training loop
    tbar = tqdm(range(niters))
    for epoch in tbar:
        rec_hr = model(coords_hr)
        rec = downsampler(rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...])

        path_norm = 0

        if nonlin == 'bspline-w' and args.lin_layers and args.path_norm:
            for l in range(hidden_layers):
                path_norm += c * torch.sum(torch.linalg.norm(model.net[l].block.linear.weight, dim=1) \
                                                * torch.linalg.norm(model.net[l].block.linear2.weight, dim=0))
            lam = args.lam
            path_norms_array.append(path_norm.item())

        elif nonlin == 'bspline-w' and args.path_norm:
            for l in range(hidden_layers):
                if l == 0:
                    path_norm += c * torch.sum(torch.linalg.norm(model.net[l].block.linear.weight, dim=1) \
                                                    * torch.linalg.norm(model.net[l + 1].block.linear.weight, dim=0))
                elif l > 1:
                    path_norm += c * torch.sum(torch.linalg.norm(model.net[l].block.linear.weight, dim=0))

            path_norms_array.append(path_norm.item())

        loss = ((gt_lr - rec.reshape(1, 3, -1).permute(0, 2, 1))**2).mean()

        with torch.no_grad():
            rec_hr = model(coords_hr)

            im_rec = rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...]

            mse_hr_array.append(((gt - rec_hr)**2).mean().item())

            # sssim_func only accepts numpy arrays
            im_gt_np = im_gt.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
            im_rec_np = im_rec.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
            ssim_array.append(ssim_func(im_gt_np, im_rec_np, channel_axis=2))

        mse_lr_array.append(loss.item())

        if nonlin == 'bspline-w' and args.path_norm:
            tbar.set_description('{:2f}: Loss LR {:e}, Path Norm: {}'.format(-10*np.log10(mse_hr_array[epoch]), loss.item(), path_norm.item()))
        else:
            tbar.set_description('{:2f}: Loss LR {:e}'.format(-10*np.log10(mse_hr_array[epoch]), loss.item()))

        tbar.refresh()

        if epoch % 10000 == 0:
            np.save(os.path.join(save_dir, 'mse_hr_array_t_{}'.format(epoch)), mse_hr_array)
            np.save(os.path.join(save_dir, 'mse_lr_array_t_{}'.format(epoch)), mse_lr_array)
            np.save(os.path.join(save_dir, 'path_norm_array_t_{}'.format(epoch)), path_norms_array)    
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_t_{}.pt'.format(epoch)))

        imrec = im_rec.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        
        if mse_hr_array[epoch] < best_mse:
            best_mse = mse_hr_array[epoch]
            best_img = imrec
        # track learning rate

        learning_rates.append(optim.param_groups[0]['lr'])
        loss.backward()
        optim.step()
        scheduler.step()
        optim.zero_grad()

        if loss.item() < args.stop_loss:
            break

    final_mse = mse_hr_array[-1]
    print('\nFinal PSNR: {}, Final SSIM:{}'.format(-10*np.log10(final_mse), ssim_array[-1]))
    if args.path_norm and nonlin == 'bspline-w':
        final_path_norm = path_norms_array[-1]
        print('Final Path Norm: {}'.format(final_path_norm))
        

    # Save everything
    if posencode:
        nonlin = 'posenc'

    np.save(os.path.join(save_dir, 'ssim_array'), ssim_array)
    np.save(os.path.join(save_dir, 'mse_hr_array'), mse_hr_array)
    np.save(os.path.join(save_dir, 'mse_lr_array'), mse_lr_array)
    np.save(os.path.join(save_dir, 'cond_num_arr'), cond_num_array)
    np.save(os.path.join(save_dir, 'cond_num_iters'), cond_num_iters)
    np.save(os.path.join(save_dir, 'path_norm_array'), path_norms_array)    
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
    
    plt.imsave(os.path.join(save_dir, 'SR_diff_%s.pdf'%nonlin),
               np.clip(abs(im - best_img), 0, 1),
               vmin=0.0,
               vmax=0.1)

    plt.imsave(os.path.join(save_dir, 'recon.pdf'), np.clip(imrec, 0, 1), dpi=500)
    plt.imsave(os.path.join(save_dir, 'im_lr.pdf'), np.clip(im_lr, 0, 1), dpi=500)
    plt.imsave(os.path.join(save_dir, 'im_hr.pdf'), np.clip(im, 0, 1), dpi=500)

