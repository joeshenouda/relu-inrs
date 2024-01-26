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
    parser.add_argument('--bottleneck', action=argparse.BooleanOptionalAction)

    parser.add_argument('--layers', type=int, default=2, help='Layers for the MLP')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs to train for')
    parser.add_argument('--omega', type=float, default=1, help='number of omega_0')
    parser.add_argument('--sigma', type=float, default=10, help='number of sigma_0')
    parser.add_argument('--device', type=int, default=1, help='GPU to use')
    args = parser.parse_args()
    
    torch.cuda.set_device('cuda:{}'.format(args.device))
    
    scale_im = 1/args.scale_im              # Initial image downsample for memory reasons
    nonlin = args.nonlin            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    print(nonlin)
    niters = args.epochs               # Number of SGD iterations
    learning_rate = args.lr        # Learning rat.
    torch.manual_seed(47)
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3
    # Noise is not used in this script, but you can do so by modifying line 82 below
    tau = 3e1                   # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2               # Readout noise (dB)
    # Gabor filter constants.
    omega0 = args.omega          # Frequency of sinusoid
    sigma0 = args.sigma           # Sigma of Gaussian
    # Network parameters
    hidden_layers = args.layers       # Number of hidden layers in the MLP
    hidden_features = args.width   # Number of hidden units per layer
    lr_scheduler = args.lr_scheduler
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
    else:
        posencode = False

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
                bottleneck=args.bottleneck)

    print(model)
    model = model.cuda()
    
    # Image SetUp
    image = utils.normalize(plt.imread('data/lighthouse.png').astype(np.float32), True)
    image = cv2.resize(image, None, fx=scale_im, fy=scale_im, interpolation=cv2.INTER_AREA)
    H, W, _ = image.shape
    plt.imshow(image)
    plt.show()
    print("downsampled image shape " + str(image.shape))
    # This is the ground truth image tensor
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
        scheduler = LambdaLR(optim, lambda x: 0.1**min(x/5000, niters/5000))
    PSNR_array = np.zeros(niters)
    mse_array = torch.zeros(niters, device='cuda')
    loss_array = torch.zeros(niters, device='cuda')
    
    tbar = tqdm.tqdm(range(niters))
    steps_til_summary = 500
    best_mse = float('inf')
    
    for idx in tbar:
        model_output = model(coords)
        estimate_img = model_output.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
        loss = ((img_tensor - estimate_img.reshape(1, 3, -1).permute(0, 2, 1)) ** 2).mean()
        
        with torch.no_grad():
            im_rec = model_output.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
            mse_array[idx] =  ((model_output - img_tensor) ** 2).mean().item()
            loss_array[idx] = loss.item()
            PSNR = -10*torch.log10(mse_array[idx])
            PSNR_array[idx] = PSNR
        
        if not idx % steps_til_summary:
            imrec = im_rec.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            # plt.imshow(imrec)
            # plt.show()
        
        if mse_array[idx] < best_mse:
            imrec = im_rec.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            best_mse = mse_array[idx]
            best_img = imrec
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if lr_scheduler == 'reduce_plateau':
            scheduler.step(loss)
        elif lr_scheduler =='stepLR' or lr_scheduler =='LambdaLR':
            scheduler.step()
        tbar.set_description('PSNR: {:.3f}, Loss{:.7f}'.format(-10*torch.log10(mse_array[idx]),loss.item()))
        tbar.refresh()
    mdict = {'rec': best_img,
             'mse_array': mse_array.detach().cpu().numpy(),
             'psnr_array': PSNR_array,
             'loss_array':loss_array.detach().cpu().numpy(),
             'gt': image
             }
    os.makedirs('results/signal_representation', exist_ok=True)
    io.savemat('results/signal_representation/%d_%d_%d_%s_%5f_%d_%d_%d_%d.mat'%(hidden_layers, hidden_features, niters, nonlin, learning_rate, omega0, sigma0, H, W), mdict)