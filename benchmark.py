# benchmark.py
"""RobustBio one-command automated benchmark suite.

Runs:
  1. Classical PCA (32D) + K-Means baseline
  2. RobustBio (100% cell coverage) from saved checkpoint
  3. RobustBio (85% coverage, entropy-filtered selective classification)

Usage:
    python benchmark.py
"""
import torch
import numpy as np
from sklearn.cluster import KMeans
import scanpy as sc

from src.data import load_and_preprocess_pbmc3k
from src.dataset import get_dataloader
from src.models import SingleCellAutoencoder, SingleCellClusteringAE, RobustBioAE
from src.utils import evaluate_clustering


def run_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running RobustBio Automated Benchmark Suite on {device}...\n")

    adata = load_and_preprocess_pbmc3k()
    ground_truth = adata.obs["cell_type"].values
    num_classes = len(set(ground_truth))

    # 1. Classical PCA + K-Means Baseline
    sc.tl.pca(adata, svd_solver="arpack", n_comps=32)
    X_pca = adata.obsm["X_pca"]
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    pca_preds = kmeans.fit_predict(X_pca)
    m_pca = evaluate_clustering(X_pca, ground_truth, pca_preds)

    # 2. RobustBio Model Evaluation from Saved Checkpoint
    eval_dataloader = get_dataloader(adata, batch_size=256, shuffle=False)
    model = RobustBioAE(input_dim=adata.n_vars, latent_dim=32, num_clusters=num_classes).to(device)

    try:
        checkpoint = torch.load("robustbio_checkpoint.pt", map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        if isinstance(checkpoint, dict) and "cluster_centers" in checkpoint:
            model.cluster_layer.data = torch.as_tensor(
                checkpoint["cluster_centers"], dtype=torch.float32, device=device
            )
        print("Loaded trained weights from 'robustbio_checkpoint.pt'.")
    except FileNotFoundError:
        print("[ERROR] 'robustbio_checkpoint.pt' not found! Train model first via 'python train_robustbio.py'.")
        return

    model.eval()
    z_list, q_list = [], []
    with torch.no_grad():
        for batch_x in eval_dataloader:
            batch_x = batch_x.to(device)
            _, z, q_a, _ = model(batch_x, noise_std=0.0)
            z_list.append(z.cpu().numpy())
            q_list.append(q_a.cpu().numpy())

    z_final = np.vstack(z_list)
    q_final = np.vstack(q_list)
    preds = np.argmax(q_final, axis=1)
    entropy = -np.sum(q_final * np.log(q_final + 1e-8), axis=1)

    m_robustbio_100 = evaluate_clustering(z_final, ground_truth, preds)

    # Filtered 85% coverage (remove top 15% highest-entropy cells)
    cutoff = np.percentile(entropy, 85)
    mask = entropy <= cutoff
    m_robustbio_85 = evaluate_clustering(z_final[mask], ground_truth[mask], preds[mask])

    print("\n" + "=" * 70)
    print("                ROBUSTBIO COMPREHENSIVE BENCHMARK TABLE")
    print("=" * 70)
    print(f"{'Method / Experiment':<35} | {'ARI':<8} | {'NMI':<8} | {'Silhouette':<10}")
    print("-" * 70)
    print(f"{'Baseline: PCA (32D) + K-Means':<35} | {m_pca['ARI']:<8.4f} | {m_pca['NMI']:<8.4f} | {m_pca['Silhouette']:<10.4f}")
    print(f"{'RobustBio (100% Cell Coverage)':<35} | {m_robustbio_100['ARI']:<8.4f} | {m_robustbio_100['NMI']:<8.4f} | {m_robustbio_100['Silhouette']:<10.4f}")
    print(f"{'RobustBio (85% Coverage - Filtered)':<35} | {m_robustbio_85['ARI']:<8.4f} | {m_robustbio_85['NMI']:<8.4f} | {m_robustbio_85['Silhouette']:<10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
