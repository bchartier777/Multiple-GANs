import torch
import torch.nn as nn

from utils import *
from network import *

# Class for all three GANs, W-GP-GAN, W-GAN and DC GAN
class GAN(nn.Module):
    def __init__(self, mod_name, image_size, hidden_dim, latent_dim, output_dim=1):
        super().__init__()

        self.__dict__.update(locals())
        self.gen = Generator_v1(image_size, hidden_dim, latent_dim)
        self.disc = Discriminator_v1(image_size, hidden_dim, output_dim, mod_name)

        self.shape = int(image_size ** 0.5)

