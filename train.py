import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms

import imageio

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

                
def to_var(tensor, cuda=True):
    if cuda:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)

    
def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def create_dir(directory):
    """Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def gan_checkpoint(iteration, G, D, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G.pkl')
    D_path = os.path.join(opts.checkpoint_dir, 'D.pkl')
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)

def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints.
    """
    G_path = os.path.join(opts.load, 'G.pkl')
    D_path = os.path.join(opts.load, 'D_.pkl')

    G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.g_conv_dim, spectral_norm=opts.spectral_norm, is_col=opts.is_col)
    D = DCDiscriminator(conv_dim=opts.d_conv_dim, is_col=opts.is_col)

    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')

    return G, D


def merge_images(sources, targets, opts):
    _, _, h, w = sources.shape
    row = int(np.sqrt(opts.batch_size))
    merged = np.zeros([3, row * h, row * w * 2])
    for (idx, s, t) in (zip(range(row ** 2), sources, targets, )):
        i = idx // row
        j = idx % row
        merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
        merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
    return merged.transpose(1, 2, 0)


def generate_gif(directory_path, keyword=None):
    images = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".png") and (keyword is None or keyword in filename):
            img_path = os.path.join(directory_path, filename)
            print("adding image {}".format(img_path))
            images.append(imageio.imread(img_path))

    if keyword:
        imageio.mimsave(
            os.path.join(directory_path, 'anim_{}.gif'.format(keyword)), images)
    else:
        imageio.mimsave(os.path.join(directory_path, 'anim.gif'), images)


def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape
    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h * nrows, cell_w * ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w, :] = array[i * ncols + j].transpose(1, 2,
                                                                                                                 0)

    if channels == 1:
        result = result.squeeze()
    return result


def gan_save_samples(G, fixed_noise, iteration, opts):
    generated_images = G(fixed_noise)
    generated_images = to_data(generated_images)

    grid = create_image_grid(generated_images)

    # merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}.png'.format(iteration))
    imageio.imwrite(path, ((grid + 1) * 255 / 2).astype(np.uint8))
    print('Saved {}'.format(path))


def get_fashion_MNIST_loader(opts):
    transform = transforms.Compose([
                    transforms.Resize(opts.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)
                ])
    
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    return trainloader


def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    # print("                 G                     ")
    # print("---------------------------------------")
    # print(G_XtoY)
    # print("---------------------------------------")

    # print("                  D                    ")
    # print("---------------------------------------")
    # print(D_X)
    # print("---------------------------------------")
    return


def create_model(opts):
    """Builds the generators and discriminators.
    """
    ### GAN
    G = DCGenerator(noise_size=opts.noise_size, conv_dim=opts.g_conv_dim, spectral_norm=opts.spectral_norm, is_col=args.is_col)
    D = DCDiscriminator(conv_dim=opts.d_conv_dim, spectral_norm=opts.spectral_norm, is_col=args.is_col)

    print_models(G, None, D, None)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')
    return G, D

def train(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create dataloader for images
    dataloader_X = get_fashion_MNIST_loader(opts=opts)
    
    # Create checkpoint and sample directories
    create_dir(opts.checkpoint_dir)
    create_dir(opts.sample_dir)

    # Start training
    G, D = gan_training_loop(dataloader_X, opts)
    return G, D

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def sample_noise(batch_size, dim):
    return to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)
  

def upconv(in_channels, out_channels, kernel_size, stride=2, padding=2, batch_norm=True, spectral_norm=False):
    """Creates a upsample-and-convolution layer, with optional batch normalization.
    """
    layers = []
    if stride>1:
        layers.append(nn.Upsample(scale_factor=stride))
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
    if spectral_norm:
        layers.append(SpectralNorm(conv_layer))
    else:
        layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=2, batch_norm=True, init_zero_weights=False, spectral_norm=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
            
    if spectral_norm:
        layers.append(SpectralNorm(conv_layer))
    else:
        layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
  

class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim, spectral_norm=False, is_col=False):
        print(f"creting DCGenerator with conv_dim {conv_dim}.")
        super(DCGenerator, self).__init__()

        self.conv_dim = conv_dim
        self.noise_size = noise_size
        self.is_col = is_col

        if self.is_col:
          self.final_output = 3
        else:
          self.final_output = 1

        self.linear_bn = upconv(in_channels=self.noise_size, out_channels=self.conv_dim*4*4*4, 
                                batch_norm=True, kernel_size = 1, stride=1, padding=0, spectral_norm=spectral_norm)
        
        self.upconv1 = upconv(in_channels=self.conv_dim*4,out_channels=self.conv_dim*2,
                              kernel_size=5, stride=2, spectral_norm=spectral_norm)
        
        self.upconv2 = upconv(in_channels=self.conv_dim*2, out_channels=self.conv_dim,
                             kernel_size=5, stride=2, spectral_norm=spectral_norm)
        
        self.upconv3 = upconv(in_channels=self.conv_dim, out_channels=1,
                             kernel_size=5, stride=2, spectral_norm=spectral_norm)

    def forward(self, z):

        batch_size = z.size(0)
        
        out = F.relu(self.linear_bn(z)).view(-1, self.conv_dim*4, 4, 4)    # BS x 128 x 4 x 4
        out = F.relu(self.upconv1(out))  # BS x 64 x 8 x 8
        out = F.relu(self.upconv2(out))  # BS x 32 x 16 x 16
        out = F.tanh(self.upconv3(out))  # BS x 3 x 32 x 32
        
        out_size = out.size()
        if out_size != torch.Size([batch_size, self.final_output, 32, 32]):
            raise ValueError("expect {} x {} x 32 x 32, but get {}".format(batch_size, self.final_output, out_size))
        return out

class DCDiscriminator(nn.Module):
    def __init__(self, conv_dim=64, spectral_norm=False, is_col=False):
        super(DCDiscriminator, self).__init__()

        self.is_col = is_col

        if self.is_col:
          self.input_channel = 3
        else:
          self.input_channel = 1

        self.conv1 = conv(in_channels=self.input_channel, out_channels=conv_dim, kernel_size=5, stride=2, spectral_norm=spectral_norm)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim*2, kernel_size=5, stride=2, spectral_norm=spectral_norm)
        self.conv3 = conv(in_channels=conv_dim*2, out_channels=conv_dim*4, kernel_size=5, stride=2, spectral_norm=spectral_norm)
        self.conv4 = conv(in_channels=conv_dim*4, out_channels=1, kernel_size=5, stride=2, padding=1, batch_norm=False, spectral_norm=spectral_norm)

    def forward(self, x):
        batch_size = x.size(0)

        out = F.relu(self.conv1(x))    # BS x 64 x 16 x 16
        out = F.relu(self.conv2(out))    # BS x 64 x 8 x 8
        out = F.relu(self.conv3(out))    # BS x 64 x 4 x 4

        out = self.conv4(out).squeeze()
        out_size = out.size()
        if out_size != torch.Size([batch_size,]):
            raise ValueError("expect {} x 1, but get {}".format(batch_size, out_size))
        return out

def gan_training_loop(dataloader, opts):

    G, D = create_model(opts)

    g_params = G.parameters()  
    d_params = D.parameters()  


    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr * 2., [opts.beta1, opts.beta2])

    train_iter = iter(dataloader)

    fixed_noise = sample_noise(100, opts.noise_size)  # # 100 x noise_size x 1 x 1

    print('Dataset size:', len(dataloader.dataset))

    iter_per_epoch = len(dataloader.dataset) // len(train_iter)
    total_train_iters = opts.train_iters

    losses = {"iteration": [], "D_fake_loss": [], "D_real_loss": [], "G_loss": []}

    # adversarial_loss = torch.nn.BCEWithLogitsLoss()
    gp_weight = 1

    try:
        for iteration in range(1, opts.train_iters + 1):

            # Reset data_iter for each epoch
            if iteration % iter_per_epoch == 0:
                train_iter = iter(dataloader)

            real_images, real_labels = train_iter.next()
            real_images, real_labels = to_var(real_images), to_var(real_labels).long().squeeze()
            m = real_images.shape[0]
                            

            ones = Variable(torch.Tensor(real_images.shape[0]).float().cuda().fill_(1.0), requires_grad=False)

            for d_i in range(opts.d_train_iters):
                d_optimizer.zero_grad()

                D_real_loss = torch.sum(torch.pow(D(real_images)-ones, 2)) / (2*m)
                noise = sample_noise(m,opts.noise_size)
                fake_images = G(noise)
                D_fake_loss = torch.sum(torch.pow(D(fake_images), 2))/(2*m)

                if opts.gradient_penalty:
                    alpha = torch.rand(real_images.shape[0], 1, 1, 1)
                    alpha = alpha.expand_as(real_images).cuda()
                    interp_images = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True).cuda()
                    D_interp_output = D(interp_images)

                    gradients = torch.autograd.grad(outputs=D_interp_output, inputs=interp_images,
                                                    grad_outputs=torch.ones(D_interp_output.size()).cuda(),
                                                    create_graph=True, retain_graph=True)[0]
                    gradients = gradients.view(real_images.shape[0], -1)
                    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

                    gp = gp_weight * gradients_norm.mean()
                else:
                    gp = 0.0

                D_total_loss = D_real_loss + D_fake_loss + gp

                D_total_loss.backward()
                d_optimizer.step()


            g_optimizer.zero_grad()
            noise = sample_noise(m,100)
            fake_images = G(noise)
            G_loss = torch.sum(torch.pow(D(fake_images)-ones, 2))/m

            G_loss.backward()
            g_optimizer.step()

            if iteration % opts.log_step == 0:
                losses['iteration'].append(iteration)
                losses['D_real_loss'].append(D_real_loss.item())
                losses['D_fake_loss'].append(D_fake_loss.item())
                losses['G_loss'].append(G_loss.item())
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                    iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss.item()))

            if iteration % opts.sample_every == 0:
                gan_save_samples(G, fixed_noise, iteration, opts)

            if iteration % opts.checkpoint_every == 0:
                gan_checkpoint(iteration, G, D, opts)
    except KeyboardInterrupt:
        return G,D

    plt.figure()
    plt.plot(losses['iteration'], losses['D_real_loss'], label='D_real')
    plt.plot(losses['iteration'], losses['D_fake_loss'], label='D_fake')
    plt.plot(losses['iteration'], losses['G_loss'], label='G')
    plt.legend()
    plt.savefig(os.path.join(opts.sample_dir, 'losses.png'))
    plt.close()
    return G, D

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",type=float,default=0.00003)
    parser.add_argument("--gp", type=int, default=0)
    parser.add_argument("--spectral", type=int, default=0)
    parser.add_argument("--d_train_iters", type=int, default=1)

    SEED = 11
    handlers = [logging.StreamHandler()]
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)

    # Set the random seed manually for reproducibility.
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    com_args = parser.parse_args()
    args = AttrDict()
    args_dict = {'is_col': False,
                'image_size':32, 
                'g_conv_dim':32, 
                'd_conv_dim':64,
                'noise_size':100,
                'num_workers': 0,
                'train_iters':20000,
                'X':'Windows', 
                'Y': None,
                'lr':0.00003,
                'beta1':0.5,
                'beta2':0.999,
                'batch_size':32, 
                'checkpoint_dir': 'results/checkpoints_gan_gp1_lr3e-5',
                'sample_dir': 'results/samples_gan_gp1_lr3e-5',
                'load': None,
                'log_step':200,
                'sample_every':200,
                'checkpoint_every':1000,
                'spectral_norm': False,
                'gradient_penalty': False,
                'd_train_iters': 1
    }
    args_dict['lr'] = com_args.lr 
    args_dict["gradient_penalty"] = True if (com_args.gp==1 ) else False 
    args_dict["spectral_norm"] = True if (com_args.spectral == 1) else False
    args_dict["d_train_iters"] = com_args.d_train_iters

    lr = com_args.lr 
    gp = "use_gp" if args_dict["gradient_penalty"] else "no_gp"
    spectral = "use_spectral" if args_dict["spectral_norm"] else "no_spectral"
    d_train_iters = com_args.d_train_iters
    sample_dir = f"slurm-results/part1-{lr}-{d_train_iters}-{gp}-{spectral}"
    args_dict["sample_dir"] = sample_dir
    logger.info(f"saving to folder {sample_dir}")

    args.update(args_dict)

    logger.info(args)

    print_opts(args)
    G, D = train(args)
    

    generate_gif(args_dict["sample_dir"])
