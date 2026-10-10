"""Evaluation utilities for single-cell clustering benchmarks."""
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def evaluate_clustering(X_latent: np.ndarray, labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    """
    Computes core clustering benchmarks:
    - ARI: Biological annotation agreement (-1 to 1, higher is better)
    - NMI: Mutual information between clusters and truth (0 to 1)
    - Silhouette: Latent separation quality (-1 to 1)

    Args:
        X_latent: Latent/embedding space (n_cells x n_dims). Used for silhouette.
        labels_true: Ground-truth cell annotations (may contain NaN / "nan" / None).
        labels_pred: Predicted cluster labels.

    Returns:
        dict with keys ARI, NMI, Silhouette, Num_Clusters.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # Filter out unannotated cells (NaN numeric, "nan"/"NaN"/"" strings, None).
    if np.issubdtype(labels_true.dtype, np.number):
        valid_mask = ~np.isnan(labels_true.astype(float))
    else:
        # Object/string array: exclude None, NaN float, and common nan strings.
        valid_mask = np.array([
            not (v is None
                 or (isinstance(v, float) and np.isnan(v))
                 or str(v).strip().lower() in ("nan", "none", ""))
            for v in labels_true
        ])

    y_true = labels_true[valid_mask]
    y_pred = labels_pred[valid_mask]
    X_sub = np.asarray(X_latent[valid_mask].toarray() if hasattr(X_latent[valid_mask], "toarray") else X_latent[valid_mask])

    if len(y_true) == 0:
        raise ValueError("No valid labeled cells after filtering NaN annotations.")

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    # Silhouette is undefined for a single cluster; return NaN in that case.
    # Subsample >10k cells to save CPU time.
    if len(np.unique(y_pred)) < 2:
        sil = float("nan")
    else:
        if len(X_sub) > 10000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_sub), 10000, replace=False)
            sil = silhouette_score(X_sub[idx], y_pred[idx])
        else:
            sil = silhouette_score(X_sub, y_pred)

    return {
        "ARI": float(ari),
        "NMI": float(nmi),
        "Silhouette": float(sil),
        "Num_Clusters": int(len(np.unique(y_pred))),
    }
