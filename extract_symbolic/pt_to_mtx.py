"""Convert PyG .pt matrix to Matrix Market (symmetric) for extract_symbolic.

Usage: python pt_to_mtx.py <in.pt> <out.mtx>

"""
import sys

import numpy as np
import torch
from scipy.io import mmwrite
from scipy.sparse import coo_matrix


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: pt_to_mtx.py <in.pt> <out.mtx>")
    in_pt, out_mtx = sys.argv[1], sys.argv[2]

    data = torch.load(in_pt, weights_only=False)
    ei = data.edge_index.numpy()
    val = data.edge_attr[:, 0].numpy().astype(np.float64)
    n = int(data.x.shape[0])
    A = coo_matrix((val, (ei[0], ei[1])), shape=(n, n)).tocsr()

    asym = abs(A - A.T)
    scale = max(float(abs(A).max()) if A.nnz else 0.0, 1.0)
    if asym.nnz and float(asym.max()) > 1e-10 * scale:
        sys.exit(f"ERROR: {in_pt}: matrix not symmetric (max|A-A^T|={asym.max():.3e})")

    mmwrite(out_mtx, A, field="real", symmetry="symmetric")


if __name__ == "__main__":
    main()
