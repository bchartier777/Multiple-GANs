import torch

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import to_device, to_var, gen_rand_norm, generate_images
from network import *


def train_model(model, args, train_data, val_data, opt_gen, opt_disc):
    agg_gen_loss = [] # Aggregated generator and discriminator losses
    agg_disc_loss = []

    # Actual number of epochs using the number of training examplars and 
    # user-defined discriminator steps
    if args.disc_step > 0:
        ep_actual = int(np.ceil(len(train_data) / (args.disc_step)))
    else:
        ep_actual = args.user_epochs

    model.train()
    if (args.mod_name == "MM_GAN") and (args.init_gen > 0):
        model, opt_gen, gen_loss, train_data = init_generator(model, opt_gen, train_data, args)

    # Train generator and discriminator for user-defined epochs
    for epoch in tqdm(range(1+args.init_gen, args.user_epochs+1)):

        gen_loss_ep, disc_loss_ep = [], []  # Epoch losses for the generator and discriminator

        for _ in range(ep_actual):

            disc_loss_st = []  # Discriminator loss for inner loop

            for _ in range(args.disc_step):

                # Extract batch and train one step
                bat_img = extract_imgs(train_data)
                opt_disc.zero_grad()
                disc_loss = disc_train_onestep(model, bat_img, args)

                # Update model parameters and log results
                disc_loss.backward()
                opt_disc.step()
                disc_loss_st.append(disc_loss.item())

                # Clip weights to enforce Lipschitz condition
                if (args.mod_name == "W_GAN"):
                    clip_params(model, clip=args.clip)

            disc_loss_ep.append(np.mean(disc_loss_st))

            # Train the generator, minimize the appropriate distance function
            opt_gen.zero_grad()
            gen_loss = train_gen_onestep(args, model, bat_img)

            # Update model parameters and log results
            gen_loss_ep.append(gen_loss.item())
            gen_loss.backward()
            opt_gen.step()

        # Aggregate losses and output progress
        agg_gen_loss.extend(gen_loss_ep)
        agg_disc_loss.extend(disc_loss_ep)

        print ("Epoch [%d/%d], generator loss: %.4f, discriminator loss: %.4f"
               %(epoch, args.user_epochs, np.mean(gen_loss_ep), np.mean(disc_loss_ep)))
        args.user_epochs += 1

        # Generate sample images to validate progress
        if args.val_output:
            generate_images(model, args, epoch)

# Single discriminator training step
def disc_train_onestep(model, bat_img, args):
    sample = gen_rand_norm(bat_img.shape[0], model.latent_dim)
    gen_out = model.gen(sample)

    # Process the real and generated images with the discriminator
    disc_out = model.disc(bat_img) # D(z)
    disc_gen_out = model.disc(gen_out) # D(G(z))

    if (args.mod_name == "W_GP_GAN"):
        # Gradient penalty - Sample batch entries with Uniform distribution
        unif_samp = to_var(torch.rand(bat_img.shape[0], 1).expand(bat_img.size()))

        # Process the noise with the generator - Refer to Algorithm 1 of Original paper - https://arxiv.org/abs/1701.07875
        gen_inter = unif_samp * bat_img + (1-unif_samp)*gen_out
        disc_inter = model.disc(gen_inter)

        # Calculate the gradients with respect to the output of the discriminator
        params = to_device(torch.ones(disc_inter.size()))

        grad = torch.autograd.grad(outputs=disc_inter,
                                    inputs=gen_inter,
                                    grad_outputs=params,
                                    only_inputs=True,
                                    create_graph=True,
                                    retain_graph=True)[0]

        # Gradient calculation
        grad_penalty = (torch.mean((grad.norm(2, dim=1) - 1)**2)) * args.lambda_v

    # Calculate the loss for the user-configured model WGAN-GP, Wasserstein or MiniMax
    if (args.mod_name == "W_GP_GAN"):
        disc_loss = torch.mean(disc_gen_out) - torch.mean(disc_out) + grad_penalty
    elif (args.mod_name == "W_GAN"):
        disc_loss = -1 * (torch.mean(disc_out)) + torch.mean(disc_gen_out)
    else: # DC GAN
        disc_loss = torch.sum(-torch.mean(torch.log(disc_out + 1e-8)
                            + torch.log(1 - disc_gen_out + 1e-8)))

    return disc_loss

# Single generator training step
def train_gen_onestep(args, model, images):
    sample = gen_rand_norm(images.shape[0], model.latent_dim) # z
    gen_out = model.gen(sample) # G(z)
    disc_gen_out = model.disc(gen_out) # D(G(z))

    # Calculate the loss for the user-configured model - it is the same for WGAN-GP and Wasserstein
    if (args.mod_name == "MM_GAN"):
        gen_loss = torch.mean(torch.log((1-disc_gen_out) + 1e-8))
    else:
        gen_loss = -1 * (torch.mean(disc_gen_out))

    return gen_loss

# Extract a batch of images for training
def extract_imgs(dataset):
    bat, _ = next(iter(dataset))
    bat = to_device(bat.view(bat.shape[0], -1))
    return bat

def clip_params(model, clip):
    for parameters in model.disc.parameters():
        parameters.data.clamp_(-clip, clip)

# Train generator for user-defined number of steps 
def init_generator(model, opt_gen, train_data, args):
    if args.init_gen > 0:
        for _ in range(args.init_gen):
            # Process a batch of images
            bat_img = extract_imgs(train_data)

            # Zero gradients for the generator, calculate loss and backpropagate
            opt_gen.zero_grad()

            gen_loss = train_gen_onestep(args, model, bat_img)
            gen_loss.backward()
            opt_gen.step()

        print('Pre-trained gen for {0} steps.'.format(args.init_gen))
    else:
        print('Generator not pre-trained')

    return model, opt_gen, gen_loss, train_data
