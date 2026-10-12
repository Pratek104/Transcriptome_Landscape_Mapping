# train_robustbio.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

from src.data import load_and_preprocess_pbmc3k
from src.dataset import get_dataloader
from src.models import RobustBioAE
from src.utils import evaluate_clustering

def target_distribution(q):
    weight = (q ** 2) / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def symmetric_kl_divergence(q1, q2):
    """Computes D_KL(q1 || q2) + D_KL(q2 || q1) to enforce perturbation consistency."""
    kl1 = F.kl_div(q1.log(), q2, reduction="batchmean")
    kl2 = F.kl_div(q2.log(), q1, reduction="batchmean")
    return kl1 + kl2

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    adata = load_and_preprocess_pbmc3k()
    dataloader = get_dataloader(adata, batch_size=128, shuffle=True)
    eval_dataloader = get_dataloader(adata, batch_size=256, shuffle=False)

    ground_truth = adata.obs["cell_type"].values
    num_classes = len(set(ground_truth))

    model = RobustBioAE(input_dim=adata.n_vars, latent_dim=32, num_clusters=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_criterion = nn.MSELoss()

    # 1. Pretraining
    print("\n--- Phase 1: Pretraining Encoder (30 Epochs) ---")
    for epoch in range(1, 31):
        model.train()
        total_loss = 0
        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat, _, _, _ = model(batch_x, noise_std=0.0)
            loss = mse_criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Pretrain Epoch {epoch:02d}/30 | MSE Loss: {total_loss/len(dataloader.dataset):.6f}")

    # 2. Centroid Initialization
    model.eval()
    with torch.no_grad():
        latent_list = []
        for batch_x in eval_dataloader:
            batch_x = batch_x.to(device)
            _, z, _, _ = model(batch_x, noise_std=0.0)
            latent_list.append(z.cpu().numpy())
        z_init = np.vstack(latent_list)

    kmeans = KMeans(n_clusters=num_classes, n_init=10, random_state=42)
    kmeans.fit(z_init)
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    # 3. RobustBio Training (Reconstruction + Soft-Clustering + Perturbation Consistency)
    print("\n--- Phase 2: Training RobustBio with Perturbation Consistency (30 Epochs) ---")
    for epoch in range(1, 31):
        model.train()
        total_loss = 0

        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            x_hat, z, q_a, q_b = model(batch_x, noise_std=0.05)
            p_a = target_distribution(q_a.detach())

            # Loss Functions
            loss_recon = mse_criterion(x_hat, batch_x)
            loss_clust = F.kl_div(q_a.log(), p_a, reduction="batchmean")
            loss_consistency = symmetric_kl_divergence(q_a, q_b)

            # Entropy Balance Guardrail
            q_bar = q_a.mean(dim=0)
            loss_balance = torch.sum(q_bar * torch.log(q_bar + 1e-8))

            # Total Loss
            loss = loss_recon + 0.1 * loss_clust + 0.05 * loss_consistency + 0.01 * loss_balance

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        if epoch % 10 == 0 or epoch == 1:
            print(f"RobustBio Epoch {epoch:02d}/30 | Loss: {total_loss/len(dataloader.dataset):.6f}")

    # 4. Save trained model weights and metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "cluster_centers": model.cluster_layer.detach().cpu(),
        "latent_dim": 32,
        "num_clusters": num_classes,
    }
    torch.save(checkpoint, "robustbio_checkpoint.pt")
    print("\n[SUCCESS] Saved trained weights to 'robustbio_checkpoint.pt'!")

    # 5. Extract Soft Assignments & Compute Per-Cell Uncertainty
    model.eval()
    with torch.no_grad():
        z_final_list, q_final_list = [], []
        for batch_x in eval_dataloader:
            batch_x = batch_x.to(device)
            _, z, q_a, _ = model(batch_x, noise_std=0.0)
            z_final_list.append(z.cpu().numpy())
            q_final_list.append(q_a.cpu().numpy())

        z_final = np.vstack(z_final_list)
        q_final = np.vstack(q_final_list)

        final_preds = np.argmax(q_final, axis=1)

        # Calculate Per-Cell Entropy (Uncertainty)
        # H_i = - sum(q_ik * log(q_ik))
        cell_entropy = -np.sum(q_final * np.log(q_final + 1e-8), axis=1)

    metrics_all = evaluate_clustering(z_final, ground_truth, final_preds)

    # 6. Risk-Coverage Evaluation: Remove Top 15% Most Uncertain Cells
    cutoff_threshold = np.percentile(cell_entropy, 85)
    certain_mask = cell_entropy <= cutoff_threshold

    metrics_certain = evaluate_clustering(
        z_final[certain_mask],
        ground_truth[certain_mask],
        final_preds[certain_mask]
    )

    print("\n================ ROBUSTBIO BENCHMARK RESULTS ================")
    print(f"All Cells (100% Coverage)     -> ARI: {metrics_all['ARI']:.4f} | NMI: {metrics_all['NMI']:.4f} | Silhouette: {metrics_all['Silhouette']:.4f}")
    print(f"Filtered Cells (85% Coverage)  -> ARI: {metrics_certain['ARI']:.4f} | NMI: {metrics_certain['NMI']:.4f} | Silhouette: {metrics_certain['Silhouette']:.4f}")
    print("==============================================================")

if __name__ == "__main__":
    main()
