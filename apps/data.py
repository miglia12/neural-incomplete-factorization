from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from torch_geometric.data import Data

from torch_geometric.loader import DataLoader

from apps.symbolic import load_symb


def matrix_to_graph_sparse(A, b):
    edge_index = torch.tensor(list(map(lambda x: [x[0], x[1]], zip(A.row, A.col))), dtype=torch.long)
    edge_features = torch.tensor(list(map(lambda x: [x], A.data)), dtype=torch.float)
    node_features = torch.tensor(list(map(lambda x: [x], b)), dtype=torch.float)

    # diag_elements = edge_index[:, 0] == edge_index[:, 1]
    # node_features = edge_features[diag_elements]
    # node_features = torch.cat((node_features, torch.tensor(list(map(lambda x: [x], b)), dtype=torch.float)), dim=1)
    
    # Embed the information into data object
    data = Data(x=node_features, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
    return data


def matrix_to_graph(A, b):
    return matrix_to_graph_sparse(coo_matrix(A), b)


def graph_to_matrix(data, normalize=False):
    A = torch.sparse_coo_tensor(data.edge_index, data.edge_attr[:, 0].squeeze(), requires_grad=False)
    b = data.x[:, 0].squeeze()
    
    if normalize:
        b = b / torch.linalg.norm(b)
    
    return A, b


def get_dataloader(dataset, n=0, batch_size=1, spd=True, mode="train", size=None, graph=True,
                   root=Path("./data")):
    # Setup datasets

    root = Path(root)

    if dataset == "random":
        data = FolderDataset(root / "Random" / mode, n, size=size, graph=graph)

    elif dataset == "poisson":
        data = FolderDataset(root / "Poisson" / mode, n, size=size, graph=graph)

    else:
        raise NotImplementedError("Dataset not implemented, Available: random, poisson")

    # Data Loaders
    if mode == "train":
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=1, shuffle=False)

    return dataloader


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, n, graph=True, size=None) -> None:
        super().__init__()

        self.graph = True
        assert self.graph, "Graph keyword is depracated, only graph=True is supported."

        folder = Path(folder)
        ext = "pt" if self.graph else "npz"
        pattern = f"{n}_*.{ext}" if n != 0 else f"*.{ext}"
        self.files: list[Path] = list(folder.glob(pattern))

        if size is not None:
            assert len(self.files) >= size, f"Only {len(self.files)} files found in {folder} with n={n}"
            self.files = self.files[:size]

        if len(self.files) == 0:
            raise FileNotFoundError(f"No files found in {folder} with n={n}")

        # PARDISO symbolic file (.symb.npz) next to each .pt, or None when absent
        # (e.g. synthetic dataset has no symbolic data) — __getitem__ falls back to legacy behaviour.
        self.symb_files: list[Path | None] = [
            (pt_path.with_suffix(".symb.npz") if pt_path.with_suffix(".symb.npz").exists() else None)
            for pt_path in self.files
        ]
        self._first_relabel_logged: bool = False

        # One-shot dataset-level summary (zero hot-path cost).
        n_pt: int = len(self.files)
        n_symb: int = sum(1 for symb_path in self.symb_files if symb_path is not None)
        if n_symb > 0:
            print(f"[FolderDataset] {folder}  pt={n_pt}  symbolic={n_symb}  -> METIS relabel ENABLED")
        else:
            print(f"[FolderDataset] {folder}  pt={n_pt}  symbolic=0  -> natural ordering")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.graph:
            g = torch.load(self.files[idx], weights_only=False)

        else:
            # deprecated...
            d = np.load(self.files[idx], allow_pickle=True)
            g = matrix_to_graph(d["A"], d["b"])

        # If a symbolic file is paired with this matrix, relabel into PARDISO/METIS order
        # so the model trains on perm(A) instead of A. Phase 1 of the experiment rollout.
        symb_path: Path | None = self.symb_files[idx]
        if symb_path is not None:
            symb = load_symb(symb_path)
            perm = torch.from_numpy(symb["perm"])
            invp = torch.from_numpy(symb["invp"])
            g.edge_index = invp[g.edge_index]   # fancy indexing: relabel both endpoints
            g.x = g.x[perm]                     # gather node features into METIS order

            # One-shot validation: confirms METIS chose a non-trivial reordering on the first item.
            # Skipped on every subsequent call; zero cost in the steady-state loader hot path.
            if not self._first_relabel_logged:
                n: int = symb["n"]
                perm_np: NDArray[np.int64] = symb["perm"]
                is_identity: bool = bool(np.array_equal(perm_np, np.arange(n)))
                n_fixed: int = int((perm_np == np.arange(n)).sum())
                print(
                    f"[FolderDataset] first matrix: n={n}  perm_is_identity={is_identity}  "
                    f"fixed_points={n_fixed}/{n}"
                )
                self._first_relabel_logged = True

        return g
