"""
AgriSense AI - Module 5: Generative AI for Agricultural Class Imbalance
- DCGAN Generator for synthesizing rare plant disease lesions
- Variational Autoencoder (VAE) for latent space interpolation
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class DCGANGenerator(nn.Module):
        """
        Deep Convolutional GAN Generator.
        Projects latent vector z ~ N(0, 1) into synthetic RGB leaf image (64x64).
        """
        def __init__(self, latent_dim=100, feature_maps=64, channels=3):
            super(DCGANGenerator, self).__init__()
            self.network = nn.Sequential(
                # Input is latent_dim, goes into (feature_maps * 8) x 4 x 4
                nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(feature_maps * 8),
                nn.ReLU(True),
                # state size: (feature_maps*8) x 4 x 4 -> (feature_maps*4) x 8 x 8
                nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 4),
                nn.ReLU(True),
                # state size: (feature_maps*4) x 8 x 8 -> (feature_maps*2) x 16 x 16
                nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps * 2),
                nn.ReLU(True),
                # state size: (feature_maps*2) x 16 x 16 -> (feature_maps) x 32 x 32
                nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
                nn.BatchNorm2d(feature_maps),
                nn.ReLU(True),
                # state size: (feature_maps) x 32 x 32 -> (channels) x 64 x 64
                nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, z):
            return self.network(z)


    class LeafVAE(nn.Module):
        """
        Convolutional Variational Autoencoder for continuous leaf feature generation.
        """
        def __init__(self, latent_dim=64):
            super(LeafVAE, self).__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
            self.fc_logvar = nn.Linear(64 * 16 * 16, latent_dim)

            # Decoder
            self.decoder_input = nn.Linear(latent_dim, 64 * 16 * 16)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
                nn.Sigmoid()
            )

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            h = self.encoder(x)
            mu, logvar = self.fc_mu(h), self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            dec_in = self.decoder_input(z).view(-1, 64, 16, 16)
            return self.decoder(dec_in), mu, logvar
else:
    class DCGANGenerator:
        pass
    class LeafVAE:
        pass

