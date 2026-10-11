# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleCellAutoencoder(nn.Module):
    """
    Deep Autoencoder for scRNA-seq non-linear representation learning.
    Total parameters: ~530,000 (~2 MB memory footprint).
    """
    def __init__(self, input_dim=2000, latent_dim=32):
        super().__init__()

        # Encoder: 2000 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, latent_dim)
        )

        # Decoder: 32 -> 64 -> 128 -> 2000
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class SingleCellClusteringAE(nn.Module):
    """
    Deep Soft-Clustering Autoencoder.
    Combines Autoencoder compression with trainable cluster centroids.
    """
    def __init__(self, input_dim=2000, latent_dim=32, num_clusters=8, alpha=1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha

        # Encoder: 2000 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, latent_dim)
        )

        # Decoder: 32 -> 64 -> 128 -> 2000
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, input_dim)
        )

        # Trainable Cluster Centroids: (K x 32)
        self.cluster_layer = nn.Parameter(torch.Tensor(num_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_layer)

    def encode(self, x):
        return self.encoder(x)

    def get_soft_assignments(self, z):
        """
        Student's t-distribution kernel for soft clustering.
        q_ik = (1 + ||z_i - mu_k||^2 / alpha)^(-(alpha+1)/2)
        """
        dist = torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        q = self.get_soft_assignments(z)
        return x_hat, z, q


class RobustBioAE(nn.Module):
    """
    RobustBio: Deep Soft-Clustering with Latent Perturbation Consistency.
    """
    def __init__(self, input_dim=2000, latent_dim=32, num_clusters=8, alpha=1.0):
        super().__init__()
        self.num_clusters = num_clusters
        self.alpha = alpha

        # Encoder: 2000 -> 128 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, latent_dim)
        )

        # Decoder: 32 -> 64 -> 128 -> 2000
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, input_dim)
        )

        # Trainable Cluster Centroids
        self.cluster_layer = nn.Parameter(torch.Tensor(num_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_layer)

    def encode(self, x):
        return self.encoder(x)

    def get_soft_assignments(self, z):
        dist = torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def forward(self, x, noise_std=0.05):
        z = self.encoder(x)

        # Create dual latent views with controlled Gaussian noise
        if self.training and noise_std > 0:
            noise_a = torch.randn_like(z) * noise_std
            noise_b = torch.randn_like(z) * noise_std
            z_a = z + noise_a
            z_b = z + noise_b
        else:
            z_a = z
            z_b = z

        x_hat = self.decoder(z)
        q_a = self.get_soft_assignments(z_a)
        q_b = self.get_soft_assignments(z_b)

        return x_hat, z, q_a, q_b
