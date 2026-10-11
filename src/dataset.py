# src/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp

class SingleCellDataset(Dataset):
    """Memory-efficient PyTorch Dataset for scRNA-seq expression matrices."""
    def __init__(self, adata):
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def get_dataloader(adata, batch_size=128, shuffle=True):
    dataset = SingleCellDataset(adata)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False
    )
