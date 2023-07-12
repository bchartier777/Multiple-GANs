import torch
from torch import nn
import torch.nn.functional as F

# Two-layer generator and discriminator from the source repo
class Generator_v1(nn.Module):
    def __init__(self, image_size, hidden_dim, latent_dim):
        super().__init__()

        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, image_size)

    def forward(self, x):
        act = F.relu(self.lin1(x))

        output = torch.sigmoid(self.lin2(act))
        return output


class Discriminator_v1(nn.Module):
    def __init__(self, image_size, hidden_dim, output_dim, model_name):
        super().__init__()

        self.lin1 = nn.Linear(image_size, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.model_name = model_name

    def forward(self, x):
        activated = F.relu(self.lin1(x))
        if (self.model_name == "W_GP_GAN"):
            act = F.relu(self.lin2(activated))
        else:
            act = torch.sigmoid(self.lin2(activated))

        return act

# Six-layer generator and discriminator
class Generator_v2(nn.Module):
    def __init__(self, image_size, hidden_dim, latent_dim):
        super().__init__()

        self.main = nn.Sequential (
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, image_size),
            nn.Sigmoid()
            # nn.Tanh()
             )

    def forward(self, x):
        return self.main(x)

class Discriminator_v2(nn.Module):
    def __init__(self, image_size, hidden_dim, output_dim, model_name):
        super().__init__()

        self.model_name = model_name
        self.main = nn.Sequential (
            nn.Linear(image_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
            #nn.ReLU(nn.Linear(hidden_dim, output_dim))
             )

    def forward(self, x):
        return self.main(x)
        # return self.main(x).view(-1, 1, 28, 28)
