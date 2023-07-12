import torch, torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

# Assign tensor to device, invoke grad
def to_device_grad(x):
    return to_device(x).requires_grad_()

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mod_name', dest='mod_name', type=str, default='W_GP_GAN')
	parser.add_argument('--x_size', dest='x_size', type=int, default='784')
	parser.add_argument('--h_dim', dest='h_dim', type=int, default='400')
	parser.add_argument('--latent_dim', dest='latent_dim', type=int, default='20')
	parser.add_argument('--out_dim', dest='out_dim', type=int, default='1')
	parser.add_argument('--user_epochs', dest='user_epochs', type=int, default='25')
	parser.add_argument('--gen_lr', dest='gen_lr', type=float, default='1e-4')
	parser.add_argument('--disc_lr', dest='disc_lr', type=float, default='1e-4')
	parser.add_argument('--batch_size', dest='batch_size', type=int, default='100')
	parser.add_argument('--weight_decay', dest='weight_decay', type=float, default='1e-5')
	parser.add_argument('--lambda_v', dest='lambda_v', type=int, default='10')
	parser.add_argument('--clip', dest='clip', type=float, default='0.01')
	parser.add_argument('--disc_step', dest='disc_step', type=int, default='5')
	parser.add_argument('--init_gen', dest='init_gen', type=int, default='')
	parser.add_argument('--val_output', dest='val_output', action='store_true', default=True)
	parser.add_argument('--grey_scale', dest='grey_scale', action='store_true')
	parser.add_argument('--data_folder', dest='data_folder', type=str, default='/.data')
	parser.add_argument('--seed', dest='seed', type=int, default='3435')
	parser.add_argument('--val_subset', dest='val_subset', type=int, default='10000')
	parser.add_argument('--val_imgs', dest='val_imgs', type=int, default='3435')
	args = parser.parse_args()

	return args

def to_device(tens):
    if torch.cuda.is_available():
        tens = tens.cuda()
    return tens

# This implements either no grayscale or the standard grayscale transform
# Resize if needed - transforms.Resize(64),
def process_data_grayscale_v1(args):
    # Download and tranform the data
    if (args.grey_scale == True):
        transform_set = transforms.Compose(
                          [transforms.Grayscale(),
                           transforms.ToTensor()])
    else:
        transform_set = transforms.ToTensor()
	
    train_dataset = datasets.MNIST(root=args.data_folder,
                                    train=True,
                                    transform=transform_set,
                                    download=True)

    val_dataset = datasets.MNIST(root=args.data_folder,
                                   train=False,
                                   transform=transform_set)

    print (train_dataset.train_data.shape, val_dataset.test_data.shape)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_iter, val_iter

# Generate and save sample images from generator for model validation
def generate_images(model, args, epoch):
    # Disable batch norm, dropout layers, etc
    model.eval()

    # Generate sample noise tensor
    sample = gen_rand_norm(args.batch_size, args.latent_dim)

    # Generate representation from generator model and reshape
    gen_out = model.gen(sample)
    gen_out = gen_out.view(gen_out.shape[0],
                         model.shape,
                         model.shape,
                         -1).squeeze()

    # Determine grid size and save
    grid_size, k = int(args.val_imgs**0.5), 0
    if args.val_output:
        outname = 'valid/' + args.mod_name + '/'
        if not os.path.exists(outname):
           os.makedirs(outname)
        torchvision.utils.save_image(gen_out.unsqueeze(1).data,
                                     outname + 'reconst_%d.png'
                                     %(epoch), nrow=grid_size)

def save_model(model, savepath):
    torch.save(model.state_dict(), savepath)

def load_model(self, loadpath):
    state = torch.load(loadpath)
    self.model.load_state_dict(state)

# Generate tensor with noise from Normal dist
def gen_rand_norm(batch, latent_dim):
    return to_device(torch.randn(batch, latent_dim))
