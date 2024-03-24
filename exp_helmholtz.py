import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.ndimage
import scipy.special
import argparse

import diff_operators

parser = argparse.ArgumentParser(description='Helmholtz')
parser.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

parser.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
parser.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')
parser.add_argument('--num_epochs', type=int, default=50000,
               help='Number of epochs to train for.')

parser.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
parser.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
parser.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
parser.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
parser.add_argument('--velocity', type=str, default='uniform', required=False, choices=['uniform', 'square', 'circle'],
               help='Whether to use uniform velocity parameter')
parser.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
parser.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')

parser.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = parser.parse_args()

########################################################################################################################
############################### Setting up Helmholz dataset ############################################################
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)

def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


class SingleHelmholtzSource(Dataset):
    def __init__(self, sidelength, velocity='uniform', source_coords=[0., 0.]):
        super().__init__()
        torch.manual_seed(0)

        self.sidelength = sidelength
        self.mgrid = get_mgrid(self.sidelength).detach()
        self.velocity = velocity
        self.wavenumber = 20.

        self.N_src_samples = 100
        self.sigma = 1e-4
        self.source = torch.Tensor([1.0, 1.0]).view(-1, 2)
        self.source_coords = torch.tensor(source_coords).view(-1, 2)

        # For reference: this derives the closed-form solution for the inhomogenous Helmholtz equation.
        square_meshgrid = lin2img(self.mgrid[None, ...]).numpy()
        x = square_meshgrid[0, 0, ...]
        y = square_meshgrid[0, 1, ...]

        # Specify the source.
        source_np = self.source.numpy()
        hx = hy = 2 / self.sidelength
        field = np.zeros((sidelength, sidelength)).astype(np.complex64)
        for i in range(source_np.shape[0]):
            x0 = self.source_coords[i, 0].numpy()
            y0 = self.source_coords[i, 1].numpy()
            s = source_np[i, 0] + 1j * source_np[i, 1]

            hankel = scipy.special.hankel2(0, self.wavenumber * np.sqrt((x - x0) ** 2 + (y - y0) ** 2) + 1e-6)
            field += 0.25j * hankel * s * hx * hy

        field_r = torch.from_numpy(np.real(field).reshape(-1, 1))
        field_i = torch.from_numpy(np.imag(field).reshape(-1, 1))
        self.field = torch.cat((field_r, field_i), dim=1)

    def __len__(self):
        return 1

    def get_squared_slowness(self, coords):
        if self.velocity == 'square':
            squared_slowness = torch.zeros_like(coords)
            perturbation = 2.
            mask = (torch.abs(coords[..., 0]) < 0.3) & (torch.abs(coords[..., 1]) < 0.3)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))
        elif self.velocity == 'circle':
            squared_slowness = torch.zeros_like(coords)
            perturbation = 2.
            mask = (torch.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2) < 0.1)
            squared_slowness[..., 0] = torch.where(mask, 1. / perturbation ** 2 * torch.ones_like(mask.float()),
                                                   torch.ones_like(mask.float()))

        else:
            squared_slowness = torch.ones_like(coords)
            squared_slowness[..., 1] = 0.

        return squared_slowness

    def __getitem__(self, idx):
        # indicate where border values are
        coords = torch.zeros(self.sidelength ** 2, 2).uniform_(-1., 1.)
        source_coords_r = 5e2 * self.sigma * torch.rand(self.N_src_samples, 1).sqrt()
        source_coords_theta = 2 * np.pi * torch.rand(self.N_src_samples, 1)
        source_coords_x = source_coords_r * torch.cos(source_coords_theta) + self.source_coords[0, 0]
        source_coords_y = source_coords_r * torch.sin(source_coords_theta) + self.source_coords[0, 1]
        source_coords = torch.cat((source_coords_x, source_coords_y), dim=1)

        # Always include coordinates where source is nonzero
        coords[-self.N_src_samples:, :] = source_coords

        # We use the value "zero" to encode "no boundary constraint at this coordinate"
        boundary_values = self.source * gaussian(coords, mu=self.source_coords, sigma=self.sigma)[:, None]
        boundary_values[boundary_values < 1e-5] = 0.

        # specify squared slowness
        squared_slowness = self.get_squared_slowness(coords)
        squared_slowness_grid = self.get_squared_slowness(self.mgrid)[:, 0, None]

        return {'coords': coords}, {'source_boundary_values': boundary_values, 'gt': self.field,
                                    'squared_slowness': squared_slowness,
                                    'squared_slowness_grid': squared_slowness_grid,
                                    'wavenumber': self.wavenumber}
########################################################################################################################

class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        if self.mode == 'rbf':
            self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
            in_features = kwargs.get('rbf_centers', 1024)
        elif self.mode == 'nerf':
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                       sidelength=kwargs.get('sidelength', None),
                                                       fn_samples=kwargs.get('fn_samples', None),
                                                       use_nyquist=kwargs.get('use_nyquist', True))
            in_features = self.positional_encoding.out_dim

        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        # various input processing methods for different applications
        if self.image_downsampling.downsample:
            coords = self.image_downsampling(coords)
        if self.mode == 'rbf':
            coords = self.rbf_layer(coords)
        elif self.mode == 'nerf':
            coords = self.positional_encoding(coords)

        output = self.net(coords, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}

## To handle complex numbers
def compl_div(x, y):
    ''' x / y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = (a * c + b * d) / (c ** 2 + d ** 2)
    outi = (b * c - a * d) / (c ** 2 + d ** 2)
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


def compl_mul(x, y):
    '''  x * y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = a * c - b * d
    outi = (a + b) * (c + d) - a * c - b * d
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out

## Loss function for this task
def helmholtz_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']

    if 'rec_boundary_values' in gt:
        rec_boundary_values = gt['rec_boundary_values']

    wavenumber = gt['wavenumber'].float()
    x = model_output['model_in']  # (meta_batch_size, num_points, 2)
    y = model_output['model_out']  # (meta_batch_size, num_points, 2)
    squared_slowness = gt['squared_slowness'].repeat(1, 1, y.shape[-1] // 2)
    batch_size = x.shape[1]

    full_waveform_inversion = False
    if 'pretrain' in gt:
        pred_squared_slowness = y[:, :, -1] + 1.
        if torch.all(gt['pretrain'] == -1):
            full_waveform_inversion = True
            pred_squared_slowness = torch.clamp(y[:, :, -1], min=-0.999) + 1.
            squared_slowness_init = torch.stack((torch.ones_like(pred_squared_slowness),
                                                 torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.stack((pred_squared_slowness, torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.where((torch.abs(x[..., 0, None]) > 0.75) | (torch.abs(x[..., 1, None]) > 0.75),
                                           squared_slowness_init, squared_slowness)
        y = y[:, :, :-1]

    du, status = diff_operators.jacobian(y, x)
    dudx1 = du[..., 0]
    dudx2 = du[..., 1]

    a0 = 5.0

    # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
    Lpml = 0.5
    dist_west = -torch.clamp(x[..., 0] + (1.0 - Lpml), max=0)
    dist_east = torch.clamp(x[..., 0] - (1.0 - Lpml), min=0)
    dist_south = -torch.clamp(x[..., 1] + (1.0 - Lpml), max=0)
    dist_north = torch.clamp(x[..., 1] - (1.0 - Lpml), min=0)

    sx = wavenumber * a0 * ((dist_west / Lpml) ** 2 + (dist_east / Lpml) ** 2)[..., None]
    sy = wavenumber * a0 * ((dist_north / Lpml) ** 2 + (dist_south / Lpml) ** 2)[..., None]

    ex = torch.cat((torch.ones_like(sx), -sx / wavenumber), dim=-1)
    ey = torch.cat((torch.ones_like(sy), -sy / wavenumber), dim=-1)

    A = compl_div(ey, ex).repeat(1, 1, dudx1.shape[-1] // 2)
    B = compl_div(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)
    C = compl_mul(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)

    a, _ = diff_operators.jacobian(compl_mul(A, dudx1), x)
    b, _ = diff_operators.jacobian(compl_mul(B, dudx2), x)

    a = a[..., 0]
    b = b[..., 1]
    c = compl_mul(compl_mul(C, squared_slowness), wavenumber ** 2 * y)

    diff_constraint_hom = a + b + c
    diff_constraint_on = torch.where(source_boundary_values != 0.,
                                     diff_constraint_hom - source_boundary_values,
                                     torch.zeros_like(diff_constraint_hom))
    diff_constraint_off = torch.where(source_boundary_values == 0.,
                                      diff_constraint_hom,
                                      torch.zeros_like(diff_constraint_hom))
    if full_waveform_inversion:
        data_term = torch.where(rec_boundary_values != 0, y - rec_boundary_values, torch.Tensor([0.]).cuda())
    else:
        data_term = torch.Tensor([0.])

        if 'pretrain' in gt:  # we are not trying to solve for velocity
            data_term = pred_squared_slowness - squared_slowness[..., 0]

    return {'diff_constraint_on': torch.abs(diff_constraint_on).sum() * batch_size / 1e3,
            'diff_constraint_off': torch.abs(diff_constraint_off).sum(),
            'data_term': torch.abs(data_term).sum() * batch_size / 1}

def write_helmholtz_summary(model, model_input, gt, writer, total_steps, prefix='train_'):
    sl = 256
    coords = get_mgrid(sl)[None,...].cuda()

    def scale_percentile(pred, min_perc=1, max_perc=99):
        min = np.percentile(pred.cpu().numpy(),1)
        max = np.percentile(pred.cpu().numpy(),99)
        pred = torch.clamp(pred, min, max)
        return (pred - min) / (max-min)

    with torch.no_grad():
        if 'coords_sub' in model_input:
            summary_model_input = {'coords':coords.repeat(min(2, model_input['coords_sub'].shape[0]),1,1)}
            summary_model_input['coords_sub'] = model_input['coords_sub'][:2,...]
            summary_model_input['img_sub'] = model_input['img_sub'][:2,...]
            pred = model(summary_model_input)['model_out']
        else:
            pred = model({'coords': coords})['model_out']

        if 'pretrain' in gt:
            gt['squared_slowness_grid'] = pred[...,-1, None].clone() + 1.
            if torch.all(gt['pretrain'] == -1):
                gt['squared_slowness_grid'] = torch.clamp(pred[...,-1, None].clone(), min=-0.999) + 1.
                gt['squared_slowness_grid'] = torch.where((torch.abs(coords[...,0,None]) > 0.75) | (torch.abs(coords[...,1,None]) > 0.75),
                                            torch.ones_like(gt['squared_slowness_grid']),
                                            gt['squared_slowness_grid'])
            pred = pred[...,:-1]

        pred = lin2img(pred)

        pred_cmpl = pred[...,0::2,:,:].cpu().numpy() + 1j * pred[...,1::2,:,:].cpu().numpy()
        pred_angle = torch.from_numpy(np.angle(pred_cmpl))
        pred_mag = torch.from_numpy(np.abs(pred_cmpl))

        min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
        min_max_summary(prefix + 'pred_real', pred[..., 0::2, :, :], writer, total_steps)
        min_max_summary(prefix + 'pred_abs', torch.sqrt(pred[..., 0::2, :, :]**2 + pred[..., 1::2, :, :]**2), writer, total_steps)
        min_max_summary(prefix + 'squared_slowness', gt['squared_slowness_grid'], writer, total_steps)

        pred = scale_percentile(pred)
        pred_angle = scale_percentile(pred_angle)
        pred_mag = scale_percentile(pred_mag)

        pred = pred.permute(1, 0, 2, 3)
        pred_mag = pred_mag.permute(1, 0, 2, 3)
        pred_angle = pred_angle.permute(1, 0, 2, 3)

    writer.add_image(prefix + 'pred_real', make_grid(pred[0::2, :, :, :], scale_each=False, normalize=True),
                     global_step=total_steps)
    writer.add_image(prefix + 'pred_imaginary', make_grid(pred[1::2, :, :, :], scale_each=False, normalize=True),
                     global_step=total_steps)
    writer.add_image(prefix + 'pred_angle', make_grid(pred_angle, scale_each=False, normalize=True),
                     global_step=total_steps)
    writer.add_image(prefix + 'pred_mag', make_grid(pred_mag, scale_each=False, normalize=True),
                     global_step=total_steps)

    if 'gt' in gt:
        gt_field = lin2img(gt['gt'])
        gt_field_cmpl = gt_field[...,0,:,:].cpu().numpy() + 1j * gt_field[...,1,:,:].cpu().numpy()
        gt_angle = torch.from_numpy(np.angle(gt_field_cmpl))
        gt_mag = torch.from_numpy(np.abs(gt_field_cmpl))

        gt_field = scale_percentile(gt_field)
        gt_angle = scale_percentile(gt_angle)
        gt_mag = scale_percentile(gt_mag)

        writer.add_image(prefix + 'gt_real', make_grid(gt_field[...,0,:,:], scale_each=False, normalize=True),
                         global_step=total_steps)
        writer.add_image(prefix + 'gt_imaginary', make_grid(gt_field[...,1,:,:], scale_each=False, normalize=True),
                         global_step=total_steps)
        writer.add_image(prefix + 'gt_angle', make_grid(gt_angle, scale_each=False, normalize=True),
                         global_step=total_steps)
        writer.add_image(prefix + 'gt_mag', make_grid(gt_mag, scale_each=False, normalize=True),
                         global_step=total_steps)
        min_max_summary(prefix + 'gt_real', gt_field[..., 0, :, :], writer, total_steps)

    velocity = torch.sqrt(1/lin2img(gt['squared_slowness_grid']))[:1]
    min_max_summary(prefix + 'velocity', velocity[..., 0, :, :], writer, total_steps)
    velocity = scale_percentile(velocity)
    writer.add_image(prefix + 'velocity', make_grid(velocity[...,0,:,:], scale_each=False, normalize=True),
                     global_step=total_steps)

    if 'squared_slowness_grid' in gt:
        writer.add_image(prefix + 'squared_slowness', make_grid(lin2img(gt['squared_slowness_grid'])[:2,:1],
                                                                scale_each=False, normalize=True),
                         global_step=total_steps)

    if 'img_sub' in model_input:
        writer.add_image(prefix + 'img', make_grid(lin2img(model_input['img_sub'])[:2,:1],
                                                                scale_each=False, normalize=True),
                         global_step=total_steps)



# if we have a velocity perturbation, offset the source
if opt.velocity != 'uniform':
    source_coords = [-0.35, 0.]
else:
    source_coords = [0., 0.]

dataset = SingleHelmholtzSource(sidelength=230, velocity=opt.velocity, source_coords=source_coords)

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

# Define the model.
if opt.mode == 'pinn':
    model = modules.PINNet(out_features=2, type='tanh', mode=opt.mode)
    opt.use_lbfgs = True
else:
    model = modules.SingleBVPNet(out_features=2, type=opt.model, mode=opt.mode, final_layer_factor=1.)

model.cuda()