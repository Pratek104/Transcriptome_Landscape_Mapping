# train_autoencoder.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

from src.data import load_and_preprocess_pbmc3k
from src.dataset import get_dataloader
from src.models import SingleCellAutoencoder
from src.utils import evaluate_clustering

def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    use_amp = device.type == "cuda"

    for batch_x in dataloader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()

        if use_amp:
            # Mixed precision forward pass for 2GB VRAM optimization
            with torch.amp.autocast(device_type="cuda"):
                x_hat, _ = model(batch_x)
                loss = criterion(x_hat, batch_x)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            x_hat, _ = model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(batch_x)

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def extract_latent_space(model, dataloader, device):
    model.eval()
    latent_vectors = []
    for batch_x in dataloader:
        batch_x = batch_x.to(device)
        _, z = model(batch_x)
        latent_vectors.append(z.cpu().numpy())
    return np.vstack(latent_vectors)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    adata = load_and_preprocess_pbmc3k()
    dataloader = get_dataloader(adata, batch_size=128, shuffle=True)
    eval_dataloader = get_dataloader(adata, batch_size=256, shuffle=False)

    ground_truth = adata.obs["cell_type"].values
    num_classes = len(set(ground_truth))

    model = SingleCellAutoencoder(input_dim=adata.n_vars, latent_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    epochs = 50
    print(f"\n--- Training Deep Autoencoder ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, criterion, scaler, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Reconstruction Loss (MSE): {loss:.6f}")

    print("\nExtracting latent representations (z) and clustering...")
    z_latent = extract_latent_space(model, eval_dataloader, device)

    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    ae_labels = kmeans.fit_predict(z_latent)

    metrics = evaluate_clustering(z_latent, ground_truth, ae_labels)

    print("\n================ BENCHMARK COMPARISON ================")
    print("PCA + K-Means Baseline -> ARI: 0.6008 | NMI: 0.7563 | Silhouette: 0.1540")
    print(f"Autoencoder + K-Means  -> ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f} | Silhouette: {metrics['Silhouette']:.4f}")
    print("=======================================================")

if __name__ == "__main__":
    main()
