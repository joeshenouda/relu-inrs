#!/usr/bin/env python
from torch.optim.lr_scheduler import LambdaLR
from modules import models
from modules import utils
from modules import lin_inverse
from skimage.metrics import structural_similarity as ssim_func
import os
import sys
import tqdm
from scipy import io
import argparse
import time
import numpy as np
import wandb
from datetime import datetime
import skimage
import matplotlib.pyplot as plt
import torch
plt.gray()

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='CT Reconstruction')

    # Model Arguments
    parser.add_argument('-af', '--act-func', type=str, default='bspline-w', help='Activation function to use (bspline-w, wire, siren, posenc)')
    parser.add_argument('--c', type=float, default=1.0, help='Global scaling for BW-ReLU activation')
    parser.add_argument('--omega0', type=float, default=1, help='number of omega_0')
    parser.add_argument('--sigma0', type=float, default=10, help='number of sigma_0')
    parser.add_argument('--layers', type=int, default=3, help='Layers for the MLP')
    parser.add_argument('--lin-layers', action = argparse.BooleanOptionalAction)
    parser.add_argument('--width', type=int, default=300, help='Width for MLP')

    # Image arguments
    parser.add_argument('--meas', type=int, default=100, help='Number of projection measurements')

    # Training arguments
    parser.add_argument('-lr', '--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--no-lr-scheduler', action='store_true', help='No LR scheduler')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='LR decay rate')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--lam', type=float, default=0, help='Weight decay/Path Norm param')
    parser.add_argument('--path-norm', action=argparse.BooleanOptionalAction)
    parser.add_argument('--device', type=int, default=0, help='GPU to use')
    parser.add_argument('--rand-seed', type=int, default=40)
    parser.add_argument('--stop-loss', type=float, default=0, help='Stop at this loss')
    parser.add_argument('--wandb', action='store_true', help='Turn on wandb')
    parser.add_argument('--learnable_c', action = argparse.BooleanOptionalAction, help = 'change c into a param or not')
    
    # Parse the arguments
    args = parser.parse_args()

    nonlin = args.act_func      # type of nonlinearity, 'bspline-w', 'wire', 'siren', 'relu', 'posenc'
    niters = args.epochs            # Number of SGD iterations
    learning_rate = args.lr        # Learning rate.
    lam = args.lam
    no_lr_scheduler = args.no_lr_scheduler

    
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
        project_name = 'ct-reconstruction'
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

    # Gabor filter constants.
    omega0 = args.omega0    # Frequency of sinusoid or gloabal scaling B-spline wavelet
    sigma0= args.sigma0     # Sigma of Gaussian

    # For BW-ReLU
    c = args.c

    # Network parameters
    hidden_layers = args.layers  # Number of hidden layers in the MLP
    hidden_features = args.width  # Number of hidden units per layer
    device = 'cuda:{}'.format(args.device)

    save_dir = 'results/ct/{}_{}_c_{}_omega_{}_sigma_{}_lr_{}_lam_{}_PN_{}_width_{}_layers_{}_lin_{}_epochs_{}_seed_{}'\
        .format(datetime.now().strftime('%m%d_%H%M'),
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
                args.epochs,
                args.rand_seed)
    # Create model
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
    else:
        posencode = False

    os.makedirs(save_dir, exist_ok=True)
    
    # Random seed
    torch.manual_seed(args.rand_seed)
    
    # Generate sampling angles
    nmeas = args.meas                # Number of CT measurement
    thetas = torch.tensor(np.linspace(0, 180, nmeas, dtype=np.float32)).to(device)

    # Create phantom
    img = plt.imread('data/chest.png').astype(np.float32)
    img = utils.normalize(img, True)
    [H, W] = img.shape
    imten = torch.tensor(img)[None, None, ...].to(device)

    with torch.no_grad():
        sinogram_ten = lin_inverse.radon(imten, thetas)

    imten = torch.tensor(img)[None, None, ...].to(device)
    
    print('Image size is {}x{}'.format(H,W))

    model = models.get_INR(
                    nonlin=nonlin,
                    in_features=2,
                    out_features=1, 
                    hidden_features=hidden_features,
                    hidden_layers=hidden_layers,
                    first_omega_0=omega0,
                    hidden_omega_0=omega0,
                    c=c,
                    scale=sigma0,
                    pos_encode=posencode,
                    sidelength=nmeas,
                    linear_layers=args.lin_layers,
                    learn_c=args.learnable_c)

    print(model)
    model = model.to(device)
        
    x = torch.linspace(-1, 1, W).to(device)
    y = torch.linspace(-1, 1, H).to(device)
    
    X, Y = torch.meshgrid(x, y, indexing='xy')
        
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
        
    if args.path_norm:
        optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    else:
        optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters(), weight_decay=lam)
    
    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optimizer, lambda x:args.lr_decay**min(x/5000, 1))

    best_loss = float('inf')
    loss_array = np.zeros(niters)
    lr_array = np.zeros(niters)
    loss_sinogram_array = np.zeros(niters)
    best_im = None
    path_norms_array = []
    if args.learnable_c:
        c_array = np.zeros(niters)
    increased_lr = False
    
    t0 = time.time()
    tbar = tqdm.tqdm(range(niters))
    for idx in tbar:
        # Estimate image       
        img_estim = model(coords).reshape(-1, H, W)[None, ...]

        # Compute sinogram
        sinogram_estim = lin_inverse.radon(img_estim, thetas)
        
        loss_sino = ((sinogram_ten - sinogram_estim)**2).mean()
        loss_sinogram_array[idx] = loss_sino.item()
        path_norm = 0

        if nonlin == 'bspline-w' and args.lin_layers:
            for l in range(hidden_layers):
                path_norm += c * torch.sum(torch.linalg.norm(model.net[l].block.linear.weight, dim=1) \
                                         * torch.linalg.norm(model.net[l].block.linear2.weight, dim=0))

            lam = args.lam
            path_norms_array.append(path_norm.item())

        elif nonlin == 'bspline-w' and args.path_norm:
            with torch.no_grad():
                for l in range(hidden_layers):
                    if l == 0:
                        path_norm += c * torch.sum(torch.linalg.norm(model.net[l].block.linear.weight, dim=1) \
                                                 * torch.linalg.norm(model.net[l+1].block.linear.weight, dim=1))
                    elif l > 1:
                        path_norm += c * torch.sum(torch.linalg.norm(model.net[l].block.linear.weight, dim=0))

        if args.path_norm:
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
    t1 = time.time()
    total_time = t1-t0
    print('Total Train Time: {}'.format(total_time))
    img_estim_cpu = img_estim.detach().cpu().squeeze().numpy()
    
    psnr2 = utils.psnr(img, img_estim_cpu)
    ssim2 = ssim_func(img, img_estim_cpu, data_range=1)
    
    np.save(os.path.join(save_dir, 'loss_array'), loss_array)
    np.save(os.path.join(save_dir, 'loss_sino_array'), loss_sinogram_array)
    np.save(os.path.join(save_dir, 'path_norm_array'), path_norms_array)
    np.save(os.path.join(save_dir, 'recon_img'), img_estim_cpu)
    np.save(os.path.join(save_dir, 'orig_img'), imten.detach().cpu().numpy().squeeze())

    mdict = {'rec': img_estim_cpu,
             'gt': imten.detach().cpu().numpy().squeeze(),
             'mse_sinogram_array': loss_sinogram_array, 
             'mse_array': loss_array}
    
    io.savemat(os.path.join(save_dir, 'results.mat'), mdict)
    torch.save(model.state_dict(), os.path.join(save_dir,'model.pt'))

    # Contrast normalization
    best_im_np = best_im.detach().cpu().numpy().squeeze()
    best_im_cn = (best_im_np - np.min(best_im_np))/ (np.max(best_im_np) - np.min(best_im_np))

    plt.imsave(os.path.join(save_dir, 'recon.pdf'), img_estim_cpu, cmap='gray', vmin=0, vmax=1, dpi=300)
    plt.imsave(os.path.join(save_dir, 'orig.pdf'), imten.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1, dpi=300)
    

    print('PSNR: {:.1f} dB, SSIM: {:.3f}'.format(psnr2, ssim2))
    