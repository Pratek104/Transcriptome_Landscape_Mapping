"""Baseline benchmarks: PCA + K-Means and PCA + Graph Leiden, with UMAP visualization."""
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from src.data import load_and_preprocess_pbmc3k
from src.utils import evaluate_clustering


def main():
    # 1. Load preprocessed data
    adata = load_and_preprocess_pbmc3k()
    ground_truth = adata.obs["cell_type"].values
    num_classes = len(set(ground_truth))
    print(f"Ground truth: {num_classes} cell types, {adata.n_obs} cells.")

    # 2. Compute 32D PCA
    print("\n[1/4] Computing PCA...")
    sc.tl.pca(adata, svd_solver="arpack", n_comps=32)
    X_pca = adata.obsm["X_pca"]

    # 3. Predict clusters using K-Means
    print("[2/4] Running K-Means...")
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_pca)
    adata.obs["kmeans_pred"] = kmeans_labels
    adata.obs["kmeans_pred"] = adata.obs["kmeans_pred"].astype("category")

    metrics_kmeans = evaluate_clustering(X_pca, ground_truth, kmeans_labels)
    print(f"K-Means Results  -> ARI: {metrics_kmeans['ARI']:.4f} | NMI: {metrics_kmeans['NMI']:.4f} | Silhouette: {metrics_kmeans['Silhouette']:.4f}")

    # 4. Graph Leiden baseline (bioinformatics standard)
    print("\n[3/4] Running Graph Leiden...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=32)
    try:
        sc.tl.leiden(adata, resolution=0.5, key_added="leiden_pred")
    except Exception as e:
        print(f"Leiden failed ({e}); falling back to K-Means labels for baseline 2.")
        adata.obs["leiden_pred"] = kmeans_labels.astype(str)
    leiden_labels = adata.obs["leiden_pred"].values

    metrics_leiden = evaluate_clustering(X_pca, ground_truth, leiden_labels)
    print(f"Leiden Results   -> ARI: {metrics_leiden['ARI']:.4f} | NMI: {metrics_leiden['NMI']:.4f} | Silhouette: {metrics_leiden['Silhouette']:.4f}")

    # 5. Compute 2D UMAP embeddings for visualization
    print("[4/4] Computing 2D UMAP Projection...")
    # neighbors already computed above; recompute only if missing (e.g. fallback path)
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=32)
    sc.tl.umap(adata)

    # 6. Plot comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Ground Truth Cell Types
    sc.pl.umap(
        adata,
        color="cell_type",
        ax=axes[0],
        show=False,
        title="Ground Truth (Biological Annotations)",
    )

    # Plot Predicted K-Means Clusters
    sc.pl.umap(
        adata,
        color="kmeans_pred",
        ax=axes[1],
        show=False,
        title="Baseline K-Means Predictions",
    )

    # Plot Predicted Leiden Clusters
    sc.pl.umap(
        adata,
        color="leiden_pred",
        ax=axes[2],
        show=False,
        title="Baseline Leiden Predictions",
    )

    plt.tight_layout()
    plt.savefig("pbmc3k_baseline_clustering.png", dpi=300)
    print("\nVisualization saved to 'pbmc3k_baseline_clustering.png'!")
    plt.show()

    # 7. Visualize key biological marker genes on the UMAP (biological proof)
    # CD3D = T-cells, MS4A1 = B-cells, LYZ = Monocytes
    marker_genes = ["CD3D", "MS4A1", "LYZ"]
    available = [g for g in marker_genes if g in adata.var_names]
    missing = [g for g in marker_genes if g not in adata.var_names]
    if missing:
        print(f"\nNote: marker gene(s) {missing} not in 2000-HVG subset, skipping those.")
    if available:
        print(f"Plotting marker genes {available} on UMAP...")
        sc.pl.umap(adata, color=available, spot_size=10, show=False)
        plt.tight_layout()
        plt.savefig("pbmc3k_marker_genes.png", dpi=300)
        print("Visualization saved to 'pbmc3k_marker_genes.png'!")
        plt.show()
    else:
        print("No marker genes available in adata.var_names; skipping marker-gene UMAP.")


if __name__ == "__main__":
    main()
