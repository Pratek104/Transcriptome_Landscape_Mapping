# visualize_robustbio.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from src.data import load_and_preprocess_pbmc3k
from src.dataset import get_dataloader
from src.models import RobustBioAE
from src.utils import evaluate_clustering

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    adata = load_and_preprocess_pbmc3k()
    eval_dataloader = get_dataloader(adata, batch_size=256, shuffle=False)
    ground_truth = adata.obs["cell_type"].values
    num_classes = len(set(ground_truth))

    # 2. Load Saved Model Checkpoint (strict: fail fast on random weights)
    model = RobustBioAE(input_dim=adata.n_vars, latent_dim=32, num_clusters=num_classes).to(device)
    try:
        checkpoint = torch.load("robustbio_checkpoint.pt", map_location=device)
        # Preferred key from train_robustbio.py; accept legacy variants.
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
        print("Successfully loaded trained weights from 'robustbio_checkpoint.pt'!")
    except FileNotFoundError:
        print("\n[ERROR] 'robustbio_checkpoint.pt' not found!")
        print("Run 'python train_robustbio.py' first to train the model and save weights.")
        return

    # Quick inference pass (assuming model is trained/evaluated)
    # We will compute UMAP embeddings directly on adata
    print("\n[1/3] Extracting Latent Space & Soft Assignments...")
    model.eval()

    # Generate model outputs
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

    # Attach outputs to AnnData object for UMAP plotting
    adata.obsm["X_robustbio"] = z_final
    adata.obs["robustbio_pred"] = preds.astype(str)
    adata.obs["entropy"] = entropy

    # 3. Compute 2D UMAP on RobustBio Latent Space
    print("[2/3] Computing 2D UMAP Projection on RobustBio Latent Space...")
    sc.pp.neighbors(adata, use_rep="X_robustbio", n_neighbors=15)
    sc.tl.umap(adata)

    # 4. Generate 3-Panel Figure
    print("[3/3] Generating UMAP Visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Ground Truth
    sc.pl.umap(adata, color="cell_type", ax=axes[0], show=False, title="Ground Truth Annotations")

    # Panel B: RobustBio Predicted Clusters
    sc.pl.umap(adata, color="robustbio_pred", ax=axes[1], show=False, title="RobustBio Clusters")

    # Panel C: Per-Cell Assignment Entropy (Uncertainty Heatmap)
    sc.pl.umap(
        adata,
        color="entropy",
        ax=axes[2],
        show=False,
        title="Per-Cell Uncertainty (Entropy H_i)",
        color_map="viridis"
    )

    plt.tight_layout()
    plt.savefig("robustbio_umap_uncertainty.png", dpi=300)
    print("Saved UMAP visualization to 'robustbio_umap_uncertainty.png'")
    plt.close()

    # 5. Generate Risk-Coverage Curve Plot
    print("Computing Risk-Coverage Curves (Coverage: 100% -> 50%)...")
    coverages = np.linspace(1.0, 0.5, 11)
    ari_scores, sil_scores = [], []

    for cov in coverages:
        cutoff = np.percentile(entropy, cov * 100)
        mask = entropy <= cutoff

        m = evaluate_clustering(z_final[mask], ground_truth[mask], preds[mask])
        ari_scores.append(m["ARI"])
        sil_scores.append(m["Silhouette"])

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Cell Coverage Ratio (Filtering High-Entropy Cells)")
    ax1.set_ylabel("Adjusted Rand Index (ARI)", color=color)
    ax1.plot(coverages * 100, ari_scores, marker="o", color=color, linewidth=2, label="ARI")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.invert_xaxis()  # 100% -> 50%

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Silhouette Score", color=color)
    ax2.plot(coverages * 100, sil_scores, marker="s", color=color, linewidth=2, linestyle="--", label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("RobustBio Risk-Coverage Curve (Selective Classification)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("robustbio_risk_coverage_curve.jpg", dpi=300)
    print("Saved Risk-Coverage curve to 'robustbio_risk_coverage_curve.jpg'")

if __name__ == "__main__":
    main()
