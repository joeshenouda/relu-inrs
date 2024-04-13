from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau
from modules import models
from modules import utils
from datetime import datetime
import tqdm
import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
plt.gray()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Signal Representation Task')

    # Model Arguments
    parser.add_argument('-af','--act-func', type=str, default='bspline-w',
                    help= 'Activation function to use (bspline-w, wire, siren)')
    parser.add_argument('--c', type=float, default=1.0, help='Global scaling for BW-ReLU activation')
    parser.add_argument('--omega0', type=float, default=1, help='number of omega_0')
    parser.add_argument('--sigma0', type=float, default=10, help='number of sigma_0')
    parser.add_argument('--layers', type=int, default=3, help='Layers for the MLP')
    parser.add_argument('--width', type=int, default=300, help='Width for MLP')
    parser.add_argument('--lin-layers', action=argparse.BooleanOptionalAction)

    # Image arguments
    parser.add_argument('--image', type=str, default='camera')
    parser.add_argument('--scale-im', type=int, default=1)

    # Training Arguments
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='LambdaLR',
                        help='Learning rate scheduler to use [none or reduce_plateau, LambdaLR, stepLR] (default: none)')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs to train for')
    parser.add_argument('--lam', type=float, default=0, help='Weight decay/Path Norm param')
    parser.add_argument('--path-norm', action=argparse.BooleanOptionalAction)
    parser.add_argument('--device', type=int, default=0, help='GPU to use')
    parser.add_argument('--rand-seed', type=int, default=40)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    scale_im = 1/args.scale_im              # Initial image downsample for memory reasons
    nonlin = args.act_func            # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    print(nonlin)
    niters = args.epochs               # Number of SGD iterations
    learning_rate = args.lr        # Learning rat.
    
    torch.manual_seed(args.rand_seed)

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
        project_name = 'signal-representation'
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
    omega0 = args.omega0          # Frequency of sinusoid
    sigma0 = args.sigma0           # Sigma of Gaussian

    # For BW-ReLU
    c = args.c

    # Network parameters
    hidden_layers = args.layers       # Number of hidden layers in the MLP
    hidden_features = args.width   # Number of hidden units per layer
    device = 'cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu'
    lr_scheduler = args.lr_scheduler

    save_dir = 'results/signal_representation/{}_c_{}_omega_{}_sigma_{}_lr_{}_lam_{}_PN_{}_width_{}_layers_{}_lin_{}_epochs_{}_seed_{}_{}'.format(nonlin,
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
                                                                                            args.rand_seed,      
                                                                                            datetime.now().strftime('%m%d_%H%M'))
    
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
    else:
        posencode = False

    os.makedirs(save_dir, exist_ok=True)

    # Image SetUp
    image = utils.normalize(plt.imread('data/cameraman.tif').astype(np.float32), True)
    H, W = image.shape

    # This is the ground truth image tensor
    img_tensor = torch.tensor(image).to(device).reshape(H * W)[None, ...]

    x = torch.linspace(-1, 1, W).to(device)
    y = torch.linspace(-1, 1, H).to(device)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # use this as the input
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

    out_feats = 1
    
    model = models.get_INR(
                nonlin=nonlin,
                in_features=2,
                out_features=out_feats, 
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                first_omega_0=omega0,
                hidden_omega_0=omega0,
                c = c,
                scale=sigma0,
                pos_encode=posencode,
                linear_layers=args.lin_layers)
        

    print(model)
    model = model.to(device)
    
    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    
    if lr_scheduler == 'reduce_plateau':
        scheduler = ReduceLROnPlateau(optim, 'min', factor=0.8, patience=20, verbose=True, min_lr=1e-8,
                                      threshold=1e-4)
    elif lr_scheduler == 'stepLR':
        scheduler = StepLR(optim, step_size=2, gamma=0.1)
    
    elif lr_scheduler == 'LambdaLR':
        scheduler = LambdaLR(optim, lambda x: 0.1**min(x/niters, 1))
    
    PSNR_array = np.zeros(niters)
    mse_array = torch.zeros(niters, device=device)
    loss_array = torch.zeros(niters, device=device)
    
    tbar = tqdm.tqdm(range(niters))
    steps_til_summary = 500
    best_mse = float('inf')

    # Training loop
    for idx in tbar:
        model_output = model(coords)
        
        estimate_img = model_output.reshape(H, W)
        loss = ((img_tensor - model_output.squeeze(dim=2)) ** 2).mean()
        
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
        imrec = estimate_img.detach().cpu().numpy()
        imrec_normed = (imrec - np.min(imrec))/np.ptp(imrec)

        loss_normed = ((image - imrec_normed)**2).mean()
        tbar.set_description('PSNR: {:.3f}, Loss DNN: {:.3e}'.format(-10*np.log10(loss_normed), loss.item()))
        tbar.refresh()
        if args.wandb:
            wandb.log({'loss': loss.item(),
                       'PSNR': -10*np.log10(loss_normed),
                      'learning_rate': scheduler.get_last_lr()[0]})

    np.save(os.path.join(save_dir, 'loss_array'), loss_array.cpu().numpy())
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    plt.imsave(os.path.join(save_dir, 'recon.pdf'), estimate_img.detach().cpu().numpy(), dpi=300)
    plt.imsave(os.path.join(save_dir, 'orig.pdf'), image, dpi=30)
