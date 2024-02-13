import os
import sys
import tqdm
from scipy import io
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.gray()

from skimage.metrics import structural_similarity as ssim_func
import torch
from torch.optim.lr_scheduler import LambdaLR
from modules import models
from modules import utils
from modules import lin_inverse

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from modules import utils
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

from datetime import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Signal Representation Task')

    # Data setup
    parser.add_argument('--scale-im', type=int, default=1)
    
    # Model settings
    parser.add_argument('--nonlin', type=str, default='bspline',
                    help= 'type of nonlinearity, [wire, siren, mfn, relu, posenc, gauss, bspline]')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--lr-scheduler', type=str, default='LambdaLR', help='Learning rate scheduler to use [none or reduce_plateau, LambdaLR, stepLR] (default: none)')
    parser.add_argument('--width', type=int, default=300, help='Width for MLP')
    parser.add_argument('--lin-layers', action=argparse.BooleanOptionalAction)
    parser.add_argument('--skip-conn', action = argparse.BooleanOptionalAction, help='Add skip connection')

    parser.add_argument('--image', type=str, default='camera')
    parser.add_argument('--lam', type=float, default=0, help='Weight decay/Path Norm param')
    parser.add_argument('--path-norm', action=argparse.BooleanOptionalAction)

    parser.add_argument('--layers', type=int, default=3, help='Layers for the MLP')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs to train for')
    parser.add_argument('--omega0', type=float, default=1, help='number of omega_0')
    parser.add_argument('--sigma', type=float, default=10, help='number of sigma_0')
    parser.add_argument('--rand-seed', type=int, default=40)
    parser.add_argument('--device', type=int, default=1, help='GPU to use')
    args = parser.parse_args()
    
    torch.cuda.set_device('cuda:{}'.format(args.device))
    
    scale_im = 1/args.scale_im              # Initial image downsample for memory reasons
    nonlin = args.nonlin            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    print(nonlin)
    niters = args.epochs               # Number of SGD iterations
    learning_rate = args.lr        # Learning rat.
    
    torch.manual_seed(args.rand_seed)
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3

    # Gabor filter constants.
    omega0 = args.omega0          # Frequency of sinusoid
    sigma0 = args.sigma           # Sigma of Gaussian
    
    # Network parameters
    hidden_layers = args.layers       # Number of hidden layers in the MLP
    hidden_features = args.width   # Number of hidden units per layer
    
    lr_scheduler = args.lr_scheduler

    save_dir = 'results/signal_representation/{}_omega_{}_lr_{}_lam_{}_PN_{}_width_{}_layers_{}_lin_{}_skip_{}_epochs_{}_seed_{}_{}'.format(nonlin,
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
    
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
    else:
        posencode = False


    os.makedirs(save_dir, exist_ok=True)

    out_feats = 3
    if args.image == 'camera':
        out_feats=1
    
    model = models.get_INR(
                nonlin=nonlin,
                in_features=2,
                out_features=out_feats, 
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                first_omega_0=omega0,
                hidden_omega_0=omega0,
                scale=sigma0,
                pos_encode=posencode,
                linear_layers=args.lin_layers)
        

    print(model)
    model = model.cuda()
    
    # Image SetUp
    if args.image == 'camera':
        image = utils.normalize(plt.imread('data/cameraman.tif').astype(np.float32), True)
        H, W = image.shape
    else:
        image = utils.normalize(plt.imread('data/lighthouse.png').astype(np.float32), True)
        image = cv2.resize(image, None, fx=scale_im, fy=scale_im, interpolation=cv2.INTER_AREA)
        H, W, _ = image.shape

    # This is the ground truth image tensor
    if args.image == 'camera':
        img_tensor = torch.tensor(image).cuda().reshape(H*W)[None, ...]
    else:
        img_tensor = torch.tensor(image).cuda().reshape(H*W, 3)[None, ...]
    
    x = torch.linspace(-1, 1, W).cuda()
    y = torch.linspace(-1, 1, H).cuda()
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    # use this as the input
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]
    # coords = torch.stack((coord,coord,coord))
    
    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    
    if lr_scheduler == 'reduce_plateau':
        scheduler = ReduceLROnPlateau(optim, 'min', factor=0.8, patience=20, verbose=True, min_lr=1e-8,
                                      threshold=1e-4)
    elif lr_scheduler == 'stepLR':
        scheduler = StepLR(optim, step_size=2, gamma=0.1)
    
    elif lr_scheduler == 'LambdaLR':
        scheduler = LambdaLR(optim, lambda x: 0.1**min(x/niters, 1))
    
    PSNR_array = np.zeros(niters)
    mse_array = torch.zeros(niters, device='cuda')
    loss_array = torch.zeros(niters, device='cuda')
    
    tbar = tqdm.tqdm(range(niters))
    steps_til_summary = 500
    best_mse = float('inf')
    
    for idx in tbar:
        model_output = model(coords)
        
        if args.image == 'camera':
            estimate_img = model_output.reshape(H, W)
            loss = ((img_tensor - model_output.squeeze(dim=2)) ** 2).mean()
        else:
            estimate_img = model_output.reshape(H, W, 3)
            loss = ((img_tensor - model_output) ** 2).mean()

        # if idx % 100:
        #     plt.imshow(estimate_img.detach().cpu().numpy())
        #     plt.show()
        
        loss_array[idx] = loss.item()
        
        if not idx % steps_til_summary:
            imrec = estimate_img.detach().cpu().numpy()
        
        if mse_array[idx] < best_mse:
            imrec = estimate_img
            best_mse = mse_array[idx]
            best_im = imrec
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if lr_scheduler == 'reduce_plateau':
            scheduler.step(loss)
        elif lr_scheduler =='stepLR' or lr_scheduler =='LambdaLR':
            scheduler.step()
        tbar.set_description('PSNR: {:.3f}, Loss: {:.3e}'.format(-10*np.log10(loss.item()),loss.item()))
        tbar.refresh()

    np.save(os.path.join(save_dir, 'loss_array'), loss_array.cpu().numpy())
    torch.save(model.state_dict(), os.path.join(save_dir,'model.pt'))

    plt.imsave(os.path.join(save_dir, 'recon.pdf'), estimate_img.detach().cpu().numpy(), dpi=300)
    plt.imsave(os.path.join(save_dir, 'orig.pdf'), image, dpi=300)
