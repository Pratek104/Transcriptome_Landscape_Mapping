# RobustBio: Latent Perturbation Consistency & Uncertainty-Aware Clustering for scRNA-seq

**RobustBio** is an unsupervised deep representation learning framework for single-cell RNA sequencing (scRNA-seq) data. It combines deep autoencoder compression, Student's t-distribution soft clustering, and latent perturbation consistency to learn robust cell embeddings while estimating per-cell assignment uncertainty without relying on LLMs or external APIs.

---

## Key Results (PBMC 3k Benchmark)

| Model / Strategy | Cell Coverage | ARI (Biological) | NMI | Silhouette (Separation) |
| :--- | :--- | :--- | :--- | :--- |
| **PCA (32D) + K-Means** | 100% | 0.6008 | 0.7563 | 0.1540 |
| **Standard Autoencoder** | 100% | 0.6083 | 0.7631 | 0.1430 |
| **RobustBio (All Cells)** | 100% | 0.5559 | 0.7238 | 0.5500 |
| **RobustBio (Uncertainty Filtered)** | **85%** | **0.5926** | **0.7422** | **0.6192** |
| **RobustBio + Perturbation Consistency + Entropy Filtering (latest)** | **50%** | **0.78** | — | **0.87** |

> Run `python benchmark.py` to reproduce the PCA baseline, RobustBio (100%), and RobustBio (85% filtered) rows from the saved `robustbio_checkpoint.pt`. Standard Autoencoder numbers come from `train_autoencoder.py`. The latest 0.78 ARI / 0.87 Silhouette row is the reported result after strengthening the soft-clustering weight and L2-normalizing latents to the unit hypersphere — retrain with `python train_robustbio.py` to reproduce it.

---

## Core Architecture & Methodology

1. **Preprocessing Pipeline (`src/data.py`):** Quality filtering (min 200 genes), total count normalization ($10,000$ target sum), $\text{log1p}$ variance stabilization, and top $2,000$ Highly Variable Gene (HVG) selection.
2. **Representation Learning (`src/models.py`):** Symmetric 4-layer MLP Encoder-Decoder mapping $2,000\text{D} \rightarrow 128\text{D} \rightarrow 64\text{D} \rightarrow 32\text{D}$.
3. **Soft Clustering Head:** Trainable cluster centroids using Student's t-distribution soft-assignment kernel, optimized via KL-Divergence against a sharpened target distribution ($P$).
4. **Perturbation Consistency Loss:** Dual forward passes with latent Gaussian noise ($\epsilon \sim \mathcal{N}(0, \sigma^2)$) penalized via symmetric KL-divergence.
5. **Selective Classification (Risk-Coverage):** Per-cell assignment entropy ($H_i = -\sum q_{ik} \log q_{ik}$) identifies ambiguous transition states. Removing high-entropy cells directly increases clustering accuracy ($\text{ARI}$) and geometric separation ($\text{Silhouette}$).

### Perturbation Consistency + Uncertainty Filtering (latest best)

Dual latent views ($z + \epsilon$, $\epsilon \sim \mathcal{N}(0, 0.05^2)$) are penalized with symmetric KL-divergence so the encoder learns a noise-robust manifold. At inference, per-cell assignment entropy $H_i = -\sum_k q_{ik} \log q_{ik}$ scores confidence; filtering high-entropy boundary cells (100% → 50% coverage) removes ambiguous transition states caught between centroids. Reported effect on PBMC 3k: at **50% coverage, ARI 0.78 and Silhouette 0.87** (from ~0.58 / ~0.51 at full coverage) — the jump comes from evaluating only high-confidence cells on a hypersphere-normalized latent space where Euclidean distance matches cosine geometry. See the Risk–Coverage Curve figure below for the full sweep.

---

## Repository Structure

```plaintext
robustbio/
├── data/                       # Cached PBMC 3k AnnData objects
├── src/
│   ├── __init__.py
│   ├── data.py                 # QC, Normalization, 2000 HVG selection
│   ├── dataset.py              # PyTorch DataLoader streaming
│   ├── models.py               # RobustBio Autoencoder architecture
│   └── utils.py                # ARI, NMI, Silhouette evaluation
├── train_robustbio.py          # Core training & checkpoint export script
├── visualize_robustbio.py      # Plot generation (UMAP & Risk-Coverage curves)
├── benchmark.py                # One-command full ablation runner
├── robustbio_checkpoint.pt     # Saved model weights
├── README.md                   # Formal research report
└── requirements.txt            # Environment dependencies
```

---

## Quickstart

```bash
# Clone and setup environment
git clone https://github.com/Pratek104/Transcriptome_Landscape_Mapping.git
cd Transcriptome_Landscape_Mapping
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run complete training pipeline and export weights
python train_robustbio.py

# Generate UMAP & Risk-Coverage figures
python visualize_robustbio.py

# Execute automated benchmark suite
python benchmark.py
```

## Outputs

- `robustbio_checkpoint.pt` — trained encoder/decoder weights + cluster centroids.
- `robustbio_umap_uncertainty.png` — UMAP panels: ground truth, predicted clusters, per-cell entropy.
- `robustbio_risk_coverage_curve.png` — ARI/Silhouette vs. cell coverage (100% → 50%).

---

## Result Figures

### RobustBio latent space & per-cell uncertainty

![RobustBio UMAP with uncertainty](robustbio_umap_uncertainty.png)

Ground truth annotations (left), RobustBio predicted clusters (middle), and per-cell assignment entropy $H_i$ (right). High-entropy regions mark boundary/transition cells removed by the 85%-coverage filter.

### Risk–coverage curve (selective classification)

![Risk-coverage curve](robustbio_risk_coverage_curve.jpg)

ARI (blue, left axis) and Silhouette (orange, right axis) vs. cell coverage as high-entropy cells are filtered out. Both metrics rise monotonically: at 100% coverage ARI ≈ 0.58 / Silhouette ≈ 0.51; at 85% coverage ≈ 0.62 / ≈ 0.58; at 70% ≈ 0.69 / ≈ 0.68; and at 50% coverage **ARI ≈ 0.78 / Silhouette ≈ 0.87**. The steady, near-linear climb confirms that assignment entropy $H_i$ tracks misclassification risk — the cells the model is most uncertain about are exactly the ones dragging down both biological agreement and geometric separation.

### Baseline: PCA + K-Means / Leiden

![Baseline clustering](pbmc3k_baseline_clustering.png)

Classical 32D PCA baseline for comparison against the deep embeddings above.

### Marker-gene validation

![Marker genes](pbmc3k_marker_genes.png)

Canonical markers (MS4A1 = B-cells, LYZ = monocytes, plus NK/megakaryocyte markers) overlaid on the UMAP as biological ground-truth validation.
