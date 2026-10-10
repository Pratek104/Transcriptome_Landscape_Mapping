"""Data pipeline & preprocessing for PBMC 3k (2,000 HVGs, memory-optimized)."""
import os
import scanpy as sc
import anndata as ad


def load_and_preprocess_pbmc3k(data_dir: str = "./data") -> ad.AnnData:
    """Loads and precomputes log1p normalized expression matrix for PBMC 3k.

    Steps: QC filter (min_genes=200, min_cells=3) -> save raw counts layer ->
    total-count normalize (1e4) + log1p -> top 2,000 HVGs -> attach
    ground-truth annotations from pbmc3k_processed (louvain).

    Args:
        data_dir: Directory used by scanpy to cache downloads.

    Returns:
        AnnData with n_obs cells x 2000 HVGs and obs['cell_type'].
    """
    os.makedirs(data_dir, exist_ok=True)
    # Point scanpy's cache at data_dir without changing global state permanently.
    old_datasetdir = sc.settings.datasetdir
    sc.settings.datasetdir = os.path.abspath(data_dir)
    try:
        print("[1/3] Downloading/Loading PBMC 3k dataset...")
        adata = sc.datasets.pbmc3k()
    finally:
        sc.settings.datasetdir = old_datasetdir

    print("[2/3] Performing Quality Control & Normalization...")
    # Basic QC filtering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Save raw counts in a layer for count-based loss functions.
    # .X may be sparse; copy preserves type.
    adata.layers["raw_counts"] = adata.X.copy()

    # Target sum normalization and log1p transformation
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("[3/3] Selecting top 2,000 Highly Variable Genes (HVGs)...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

    # Load standard cell type ground-truth annotations from Scanpy
    old_datasetdir = sc.settings.datasetdir
    sc.settings.datasetdir = os.path.abspath(data_dir)
    try:
        pbmc3k_annotated = sc.datasets.pbmc3k_processed()
    finally:
        sc.settings.datasetdir = old_datasetdir

    # pbmc3k_processed stores labels in 'louvain'; fall back gracefully.
    label_col = "louvain" if "louvain" in pbmc3k_annotated.obs.columns else pbmc3k_annotated.obs.columns[0]

    # Align annotations with our preprocessed object
    common_cells = adata.obs_names.intersection(pbmc3k_annotated.obs_names)
    if len(common_cells) == 0:
        raise RuntimeError("No overlapping cells between pbmc3k and pbmc3k_processed; cannot attach labels.")
    adata = adata[common_cells].copy()
    adata.obs["cell_type"] = pbmc3k_annotated.obs.loc[common_cells, label_col].astype(str).values

    print(f"Dataset successfully ready: {adata.n_obs} cells x {adata.n_vars} HVGs.")
    return adata
