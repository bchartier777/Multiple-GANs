""" 
This is an implementation of the following three GANs:

Wasserstein GAN with Gradient Penalties - 'Improved Training of Wasserstein GANs'
 -  https://arxiv.org/abs/1704.00028

'Wasserstein GAN'
 - https://arxiv.org/abs/1701.07875

DC GAN from Ian Goodfellow, et al - 'Generative Adversarial Networks'
 - https://arxiv.org/pdf/1406.2661
 
The primary reference repo was Shayne Obrien,https://github.com/shayneobrien/generative-models
"""

import torch
import torch.optim as optim
from utils import parse_args, process_data_grayscale_v1
from train_v1 import *
from model import *

def main(args):
    torch.manual_seed(args.seed)
    print ("Training GAN:", args.mod_name)

    # Download and extract MNIST data
    train_iter, val_iter = process_data_grayscale_v1(args)

    model = GAN(mod_name=args.mod_name, image_size=args.x_size, hidden_dim=args.h_dim,
                   latent_dim=args.latent_dim, output_dim=args.out_dim)
    model = to_device(model)
    model.dim = int(args.x_size ** 0.5)
    opt_gen = optim.AdamW(params = model.gen.parameters(), lr=args.gen_lr,
                       weight_decay=args.weight_decay)
    opt_disc = optim.AdamW(params = model.disc.parameters(), lr=args.disc_lr,
                       weight_decay=args.weight_decay)

    # Train model and save validation images - not using the test dataset with this version
    train_model(model, args, train_iter, val_iter, opt_gen, opt_disc)

if __name__ == '__main__':
	args = parse_args() # Parse arguments from command line
	main(args)
