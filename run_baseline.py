"""Baseline benchmarks: PCA + K-Means and PCA + Graph Leiden."""
import scanpy as sc
from sklearn.cluster import KMeans
from src.data import load_and_preprocess_pbmc3k
from src.utils import evaluate_clustering


def main():
    adata = load_and_preprocess_pbmc3k()
    ground_truth = adata.obs["cell_type"].values
    num_classes = len(set(ground_truth))
    print(f"Ground truth: {num_classes} cell types, {adata.n_obs} cells.")

    print("\n--- Running Baseline 1: PCA + K-Means ---")
    sc.tl.pca(adata, svd_solver="arpack", n_comps=32)
    X_pca = adata.obsm["X_pca"]

    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_pca)

    metrics_kmeans = evaluate_clustering(X_pca, ground_truth, kmeans_labels)
    print(f"K-Means Results  -> ARI: {metrics_kmeans['ARI']:.4f} | NMI: {metrics_kmeans['NMI']:.4f} | Silhouette: {metrics_kmeans['Silhouette']:.4f}")

    print("\n--- Running Baseline 2: PCA + Graph Leiden (Bioinformatics Standard) ---")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=32)
    try:
        sc.tl.leiden(adata, resolution=0.5, key_added="leiden_pred")
    except Exception as e:
        print(f"Leiden failed ({e}); falling back to K-Means labels for baseline 2.")
        adata.obs["leiden_pred"] = kmeans_labels.astype(str)
    leiden_labels = adata.obs["leiden_pred"].values

    metrics_leiden = evaluate_clustering(X_pca, ground_truth, leiden_labels)
    print(f"Leiden Results   -> ARI: {metrics_leiden['ARI']:.4f} | NMI: {metrics_leiden['NMI']:.4f} | Silhouette: {metrics_leiden['Silhouette']:.4f}")


if __name__ == "__main__":
    main()
