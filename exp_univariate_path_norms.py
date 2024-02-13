import torch
import numpy as np
from torch.fft import rfft, fft
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.nn.utils import clip_grad_value_, clip_grad_norm_


import os
import argparse
from scipy import io

import matplotlib.pyplot as plt
from datetime import datetime

from modules import models
from tqdm import tqdm


def train(net, x_samps, y_samps, iters=50000, lr=1e-3, c=1, optim='adam', nonlin='bspline-w', lrscheduler=False):

    data_loss_arr = []
    path_norm_arr = []
    cond_num_arr = []

    if optim=='sgd':
        optim_func = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        optim_func = torch.optim.Adam(net.parameters(), lr=lr)
        
    if lrscheduler:
        scheduler = ReduceLROnPlateau(optim_func, mode='min',factor=0.5, patience=100, \
                                      threshold=0.00001, verbose=True, min_lr=1e-7)

    criterion = torch.nn.MSELoss()

    tbar = tqdm(range(iters))
    for epoch in tbar:
        y_hat = net(x_samps)
        loss = criterion(y_hat, y_samps)
        
        if nonlin == 'bspline-w':
            path_norm = c*torch.sum(torch.linalg.norm(net.net[0].block.linear.weight, dim=1) \
                                             * torch.linalg.norm(net.net[1].weight, dim=0))
            path_norm_arr.append(path_norm.item())

        #loss_tot = loss + lam * path_norm

        # if t % 10 == 0:
        #     print('Iter: {}, Data Loss: {}, Path Norm: {}, Tot Loss: {}\n'.format(t, loss.item(), path_norm.item(), loss_tot.item()))
        atoms_dict = activation['last_feat'].detach().squeeze().T


        cond_num_arr.append(torch.linalg.cond(atoms_dict))

        data_loss_arr.append(loss.item())
        
        if nonlin == 'bspline-w':
            tbar.set_description('Loss: {:e}, PN:{:2f}'.format(loss.item(), path_norm.item()))
        else:
            tbar.set_description('Loss: {:e}'.format(loss.item()))
        tbar.refresh()

        optim_func.zero_grad()
        loss.backward()
        optim_func.step()
        
        if lrscheduler:
            scheduler.step(loss)

    return data_loss_arr, path_norm_arr, cond_num_arr


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Univariate B-spline W. Path Norm Exp')

    # Add an argument
    parser.add_argument('-lr', '--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of iterations')
    parser.add_argument('--c', type=float, default=10, help='Global scaling for activation')
    parser.add_argument('--af', type=str, default='bspline-w', help='Activation function [bspline-w, relu]')
    parser.add_argument('--f-gt', type=str, default='thresh', help='Ground truth function either CPWL or PWC (relu or thresh)')
        
    torch.manual_seed(40)
    # Parse the arguments
    args = parser.parse_args()

    if args.f_gt == 'relu':
        # Make ReLU NN that will interpolate the data above
        net = models.get_INR(
                            nonlin='relu',
                            in_features=1,
                            out_features=1, 
                            hidden_features=10,
                            hidden_layers=1,
                            first_omega_0=1,
                            hidden_omega_0=1)
        
        net.load_state_dict(torch.load('simple_uni_relu_nn.pt'))
    
    elif args.f_gt == 'thresh':
        # Defining the univariate threshold neural network
        class UnivariateThresholdNet(torch.nn.Module):
            def __init__(self):
                super(UnivariateThresholdNet, self).__init__()
                self.threshold = torch.nn.Threshold(1, 0)
                self.fc1 = torch.nn.Linear(1, 20)  # One input to one output
                self.fc2 = torch.nn.Linear(20, 1, bias=False)  # One input to one output
        
        
                self.fc1.weight.data = torch.ones_like(self.fc1.weight.data)
                #self.fc1.bias.data = torch.linspace(-2, 2, 20)
                self.fc1.bias.data = self.fc1.bias.data * 2
                
                self.fc2.weight.data = 5 * self.fc2.weight.data
        
            def forward(self, x):
                x = self.fc1(x)
                x_out = self.threshold(x)
                x_out[x_out > 0] = 1
                return self.fc2(x_out)

        net = UnivariateThresholdNet()
        net.load_state_dict(torch.load('simple_uni_thresh_nn.pt'))


    
    ## Sample data from this ground truth function
    n_samps = 20
    
    x_samps = torch.linspace(-1,1, n_samps).unsqueeze(dim=1)
    y_samps = net(x_samps).detach()
    
    
    c = args.c
    nonlin = args.af
    

    if args.af == 'bspline-w':       
        # Make B-spline NN that will interpolate the data above
        new_net = models.get_INR(
                            nonlin='bspline-w',
                            in_features=1,
                            out_features=1, 
                            hidden_features=100,
                            hidden_layers=1,
                            first_omega_0=c,
                            hidden_omega_0=c,
                            linear_layers=True,
                            skip_conn=True)
        print(new_net)
    elif args.af == 'relu':
        new_net = models.get_INR(
            nonlin='relu',
            in_features=1,
            out_features=1,
            hidden_features=100,
            hidden_layers=1)

    activation = {}
    def get_activation_hook(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    if nonlin == 'relu' or nonlin == 'siren' or nonlin=='wire':
        new_net.net[0].register_forward_hook(get_activation_hook('last_feat'))
    else:
        new_net.net[0].block.bspline_w.register_forward_hook(get_activation_hook('last_feat'))

    print('Training with c: {}'.format(args.c))
    
    if c < 1:
        lrscheduler=True
    else:
        lrscheduler=False

    data_loss_arr, pn_arr, cond_num_arr = train(new_net, x_samps, y_samps, iters=args.epochs,\
                                                lr=args.lr, c=c, nonlin=args.af, lrscheduler=lrscheduler)

    dir_path = './results/{}_af_{}_univ_path_norms/omega_{}_lr_{}_epochs_{}_loss_{}_{}'.format(args.f_gt,
                                                                                        args.af,
                                                                                         args.c, 
                                                                                         args.lr, 
                                                                                         args.epochs, 
                                                                                         data_loss_arr[-1],
                                                                                         datetime.now().strftime('%m%d_%H%M'))
    os.makedirs(dir_path, exist_ok=True)
    
    torch.save(new_net.state_dict(), os.path.join(dir_path, 'trained_model.pt'))
    np.save(os.path.join(dir_path, 'data_loss_arr'), data_loss_arr)
    np.save(os.path.join(dir_path, 'pn_arr'), pn_arr)

    dom = torch.linspace(-1,1, 10000).unsqueeze(dim=1)
    
    fig = plt.figure(figsize=(10, 8))

    plt.rcParams.update({
        "text.usetex": True
#         "font.family": "sans-serif",
#         "font.sans-serif": "Helvetica",
    })
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    
    plt.plot(dom, new_net(dom).detach().numpy(), linewidth=5, label=r'$g_{\boldsymbol{\theta}}$' +' ($c={:.1f}$)'.format(args.c))
    plt.plot(dom, net(dom).detach(), color='green', linestyle='dashed', linewidth=5, label='$f^{*}$')
    plt.scatter(x_samps, y_samps, color='red', zorder=3, s=100)
    plt.legend(fontsize=40)
#     axs.set_title(r'$f ={:.2f}$'.format(pn_arr[-1]), fontsize=30)
    plt.title(r'$\|g_{\boldsymbol{\theta}}\|$' +'$= {:.2f}$'.format(pn_arr[-1]), fontsize=50)
    
    plt.ylim(-1.5,0.5)

    # getting current axes
    a = plt.gca()
    
    # set visibility of x-axis as False
    xax = a.axes.get_xaxis()
    xax = xax.set_visible(False)

    # set visibility of y-axis as False
    yax = a.axes.get_yaxis()
    yax = yax.set_visible(False)

    
    plt.savefig(os.path.join(dir_path, 'theshold_univ_path_norm_c_{}.pdf'.format(args.c)),bbox_inches='tight', dpi=500)



    
