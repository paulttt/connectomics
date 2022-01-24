import torch
import torch.nn as nn
from torch.autograd import Variable

import warnings


def build_network(layers, activation='relu', dropout=0):
    network = []
    for i in range(len(layers)):
        network.append(nn.Linear(layers[i-1], layers[i]))
        if activation == 'relu':
            network.append(nn.ReLU())
        elif activation == 'sigmoid':
            network.append(nn.Sigmoid())
        elif activation == 'tanh':
            network.append(nn.Tanh())
        else:
            warnings.warn('Activation type unknown. Please input either relu, sigmoid or tanh. '
                          'ReLU will be used instead.')
            network.append(nn.ReLU())
        if dropout > 0:
            network.append(nn.Dropout(dropout))
    return nn.Sequential(network)


class VAE(nn.Module):
    def __init__(self,
                 input_dim: int = 784,
                 latent_dim: int = 10,
                 hidden_dims: list = None,
                 dropout: int = 0):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [500, 200]

        modules = []
        input_channels = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_channels, h_dim))
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout())
            input_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims, latent_dim)
        self.decoder_in = nn.Linear(latent_dim, hidden_dims)

        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout())

        self.decoder = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], input_dim),
            nn.Sigmoid()
        )

    def encode(self, input):
        out = self.encoder(input)
        mu = self.fc_mu(out)
        var = self.fc_var(out)
        return mu, var

    def decode(self, z):
        out = self.decoder_in(z)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out

    def reparameterize(self, mu, var):
        std = var.mul(0.5).exp()
        epsilon = Variable(torch.FloatTensor(var.size()).normal_())
        return epsilon.mul(std).add_(mu)

    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)
        x_hat = self.decode(z)
        return x_hat, mu, var
