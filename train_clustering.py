# train_clustering.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

from src.data import load_and_preprocess_pbmc3k
from src.dataset import get_dataloader
from src.models import SingleCellClusteringAE
from src.utils import evaluate_clustering

def target_distribution(q):
    """
    Computes sharpened target distribution p_ik = (q_ik^2 / f_k) / sum(q_ik^2 / f_j)
    """
    weight = (q ** 2) / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    adata = load_and_preprocess_pbmc3k()
    dataloader = get_dataloader(adata, batch_size=128, shuffle=True)
    eval_dataloader = get_dataloader(adata, batch_size=256, shuffle=False)

    ground_truth = adata.obs["cell_type"].values
    num_classes = len(set(ground_truth))

    # 1. Initialize Deep Model
    model = SingleCellClusteringAE(input_dim=adata.n_vars, latent_dim=32, num_clusters=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_criterion = nn.MSELoss()

    # 2. Pretrain Encoder (30 Epochs) to initialize latent space
    print("\n--- Phase 1: Pretraining Encoder (30 Epochs) ---")
    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat, _, _ = model(batch_x)
            loss = mse_criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        if epoch % 10 == 0:
            print(f"Pretrain Epoch {epoch:02d}/30 | MSE Loss: {total_loss/len(dataloader.dataset):.6f}")

    # 3. Initialize Cluster Centroids with K-Means on latent space z
    print("\nInitializing Cluster Centroids with K-Means...")
    model.eval()
    with torch.no_grad():
        latent_list = []
        for batch_x in eval_dataloader:
            batch_x = batch_x.to(device)
            _, z, _ = model(batch_x)
            latent_list.append(z.cpu().numpy())
        z_init = np.vstack(latent_list)

    kmeans = KMeans(n_clusters=num_classes, n_init=10, random_state=42)
    kmeans.fit(z_init)
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    # 4. Joint Training: Reconstruction + Soft Clustering + Entropy Balance Loss
    print("\n--- Phase 2: Joint Soft-Clustering Optimization (30 Epochs) ---")
    for epoch in range(1, 31):
        model.train()
        total_loss, total_recon, total_clust = 0, 0, 0

        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            x_hat, z, q = model(batch_x)
            p = target_distribution(q.detach())

            # Loss Components
            loss_recon = mse_criterion(x_hat, batch_x)
            loss_clust = F.kl_div(q.log(), p, reduction="batchmean")

            # Entropy Balance Guardrail: prevents cluster collapse
            q_bar = q.mean(dim=0)
            loss_balance = torch.sum(q_bar * torch.log(q_bar + 1e-8))

            # Combined Loss Function
            loss = loss_recon + 0.1 * loss_clust + 0.01 * loss_balance

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_x)
            total_recon += loss_recon.item() * len(batch_x)
            total_clust += loss_clust.item() * len(batch_x)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Joint Epoch {epoch:02d}/30 | Total Loss: {total_loss/len(dataloader.dataset):.6f} | Clust KL: {total_clust/len(dataloader.dataset):.6f}")

    # 5. Extract Final Predictions
    model.eval()
    with torch.no_grad():
        z_final_list, q_final_list = [], []
        for batch_x in eval_dataloader:
            batch_x = batch_x.to(device)
            _, z, q = model(batch_x)
            z_final_list.append(z.cpu().numpy())
            q_final_list.append(q.cpu().numpy())

        z_final = np.vstack(z_final_list)
        q_final = np.vstack(q_final_list)

        # Final cluster predictions are the argmax of soft assignments
        final_preds = np.argmax(q_final, axis=1)

    metrics = evaluate_clustering(z_final, ground_truth, final_preds)

    print("\n================ BENCHMARK COMPARISON ================")
    print("PCA + K-Means Baseline   -> ARI: 0.6008 | NMI: 0.7563 | Silhouette: 0.1540")
    print("Standard Autoencoder     -> ARI: 0.6083 | NMI: 0.7631 | Silhouette: 0.1430")
    print(f"Deep Soft-Clustering AE -> ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f} | Silhouette: {metrics['Silhouette']:.4f}")
    print("=======================================================")

if __name__ == "__main__":
    main()
