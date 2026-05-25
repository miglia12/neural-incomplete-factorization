from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from numpy.typing import NDArray


class Symb(TypedDict):
    perm: NDArray[np.int64]
    invp: NDArray[np.int64]
    snode: NDArray[np.int64]
    nmod: NDArray[np.int64]
    ddist: NDArray[np.int64]
    xsuper: NDArray[np.int64]
    xlindx: NDArray[np.int64]
    lindx: NDArray[np.int64]
    n: int
    nsuper: int
    nparts: int
    nproc: int
    reorder_method: int
    recon_res: float
    verified: bool


def load_symb(path: str | Path) -> Symb:
    z = np.load(Path(path))
    arrays = ("perm", "invp", "snode", "nmod", "ddist", "xsuper", "xlindx", "lindx")
    out: Symb = {k: z[k].astype(np.int64, copy=False) for k in arrays}  # type: ignore[typeddict-item]
    out["n"] = int(z["n"])
    out["nsuper"] = int(z["nsuper"])
    out["nparts"] = int(z["nparts"])
    out["nproc"] = int(z["nproc"])
    out["reorder_method"] = int(z["reorder_method"])
    out["recon_res"] = float(z["recon_res"])
    out["verified"] = bool(z["verified"])
    return out


def expand_l_pattern(symb: Symb) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    xsuper: NDArray[np.int64] = symb["xsuper"]
    xlindx: NDArray[np.int64] = symb["xlindx"]
    lindx: NDArray[np.int64] = symb["lindx"]
    nsuper: int = symb["nsuper"]

    # Per-supernode panel height and width via paired-slice subtraction (vectorised "diff").
    panel_height_per_supernode: NDArray[np.int64] = xlindx[1:] - xlindx[:-1]
    width_per_supernode: NDArray[np.int64] = xsuper[1:] - xsuper[:-1]

    # nnz(L) = sum_s (panel_height*width  -  width*(width-1)/2). np.sum reduces over s.
    total: int = int(
        np.sum(
            panel_height_per_supernode * width_per_supernode
            - width_per_supernode * (width_per_supernode - 1) // 2
        )
    )

    rows = np.empty(total, dtype=np.int64)
    cols = np.empty(total, dtype=np.int64)

    # Fill per (supernode, column) with slice copies — bulk row-write replaces the per-row Python loop.
    cursor: int = 0
    for supernode_idx in range(nsuper):
        panel_start: int = int(xlindx[supernode_idx])
        panel_end: int = int(xlindx[supernode_idx + 1])
        first_col: int = int(xsuper[supernode_idx])
        last_col: int = int(xsuper[supernode_idx + 1])
        panel_height: int = panel_end - panel_start
        width: int = last_col - first_col
        panel: NDArray[np.int64] = lindx[panel_start:panel_end]

        for column_offset in range(width):
            num_emitted: int = panel_height - column_offset
            rows[cursor : cursor + num_emitted] = panel[column_offset:]              # slice copy: k row labels at once
            cols[cursor : cursor + num_emitted] = first_col + column_offset          # scalar broadcast into slice
            cursor += num_emitted

    assert cursor == total, f"emitted {cursor} entries, expected {total}"
    return rows, cols


def compute_fill_ins(
    l_rows: NDArray[np.int64],
    l_cols: NDArray[np.int64],
    a_edge_index_metis: NDArray[np.int64],
    n: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    # Boolean masks → strict-lower entries of L and A in one pass each.
    l_strict_mask: NDArray[np.bool_] = l_rows > l_cols
    a_strict_mask: NDArray[np.bool_] = a_edge_index_metis[0] > a_edge_index_metis[1]

    # Encode each (row, col) pair as a single int: row*n + col. Unique since row, col < n.
    l_strict_keys: NDArray[np.int64] = l_rows[l_strict_mask] * n + l_cols[l_strict_mask]
    a_strict_keys: NDArray[np.int64] = (
        a_edge_index_metis[0, a_strict_mask] * n + a_edge_index_metis[1, a_strict_mask]
    )

    # np.setdiff1d: vectorised set-difference. assume_unique skips an internal dedup pass.
    fillin_keys: NDArray[np.int64] = np.setdiff1d(l_strict_keys, a_strict_keys, assume_unique=True)

    # Decode back to (row, col) via integer divmod.
    fillin_rows: NDArray[np.int64] = fillin_keys // n
    fillin_cols: NDArray[np.int64] = fillin_keys % n
    return fillin_rows, fillin_cols


class DataExtras(TypedDict):
    perm: torch.Tensor
    invp: torch.Tensor
    fillin_edge_index: torch.Tensor
    n: int
    nnz_a: int
    nnz_l: int
    mp_growth_factor: float


def build_data_extras(
    symb: Symb,
    a_edge_index_natural: NDArray[np.int64],
) -> DataExtras:
    n: int = symb["n"]

    # L's lower-tri pattern in METIS order.
    l_rows, l_cols = expand_l_pattern(symb)

    # Relabel A's endpoints natural → METIS via fancy indexing: out[i,j] = invp[in[i,j]] elementwise.
    a_edge_index_metis: NDArray[np.int64] = symb["invp"][a_edge_index_natural]

    # Structural fill-ins (strict-lower L minus strict-lower A).
    fillin_rows, fillin_cols = compute_fill_ins(l_rows, l_cols, a_edge_index_metis, n)

    # Strict-lower counts via boolean-mask reduction.
    nnz_a: int = int((a_edge_index_metis[0] > a_edge_index_metis[1]).sum())
    nnz_l: int = int((l_rows > l_cols).sum())
    mp_growth_factor: float = float(nnz_l) / float(nnz_a) if nnz_a > 0 else 0.0

    return DataExtras(
        perm=torch.from_numpy(symb["perm"]),                                            # zero-copy
        invp=torch.from_numpy(symb["invp"]),
        fillin_edge_index=torch.stack(
            [torch.from_numpy(fillin_rows), torch.from_numpy(fillin_cols)], dim=0
        ),
        n=n,
        nnz_a=nnz_a,
        nnz_l=nnz_l,
        mp_growth_factor=mp_growth_factor,
    )


if __name__ == "__main__":
    import sys

    path: str = sys.argv[1] if len(sys.argv) > 1 else "extract_symbolic/bcsstk05.symb.npz"
    symb = load_symb(path)
    print(f"loaded {path}  (n={symb['n']}, nsuper={symb['nsuper']})")

    rows, cols = expand_l_pattern(symb)
    print(f"  expand_l_pattern: filled rows={rows.shape}, cols={cols.shape}")
    print(f"    first 6 (row, col): {list(zip(rows[:6].tolist(), cols[:6].tolist()))}")

    # lower-tri invariant: count entries where row < col (should be zero).
    n_violations: int = int((rows < cols).sum())
    print(f"    lower-tri check: {n_violations} violations (want 0)")

    # diagonal-at-column-head: np.unique returns the first-occurrence index for each distinct col,
    # then we check rows at those indices equal that column.
    unique_cols, first_occurrence_idx = np.unique(cols, return_index=True)
    n_diag_violations: int = int((rows[first_occurrence_idx] != unique_cols).sum())
    print(
        f"    diagonal-at-head: {n_diag_violations} violations (want 0); "
        f"{unique_cols.shape[0]} distinct columns"
    )

    # ----- compute_fill_ins (Step 5) -----
    from scipy.io import mmread  # local import: only needed in this driver

    mtx_path: str = path.replace(".symb.npz", ".mtx")
    a_coo = mmread(mtx_path).tocoo()
    a_edge_index_natural: NDArray[np.int64] = np.stack(
        [a_coo.row.astype(np.int64), a_coo.col.astype(np.int64)], axis=0
    )
    invp: NDArray[np.int64] = symb["invp"]
    a_edge_index_metis: NDArray[np.int64] = invp[a_edge_index_natural]                  # fancy indexing

    fillin_rows, fillin_cols = compute_fill_ins(rows, cols, a_edge_index_metis, symb["n"])
    print(f"  compute_fill_ins: {fillin_rows.shape[0]} fill-in edges")

    # A ⊆ L containment: |A_strict_lower| + |fillins| should equal |L_strict_lower|.
    a_strict_lower_count: int = int((a_edge_index_metis[0] > a_edge_index_metis[1]).sum())
    l_strict_lower_count: int = int((rows > cols).sum())
    expected_total: int = a_strict_lower_count + fillin_rows.shape[0]
    print(
        f"    A⊆L check: |A_strict_lower|={a_strict_lower_count} + |fillins|={fillin_rows.shape[0]} "
        f"== |L_strict_lower|={l_strict_lower_count}? {expected_total == l_strict_lower_count}"
    )

    # ----- build_data_extras (Step 6) -----
    extras = build_data_extras(symb, a_edge_index_natural)
    print("  build_data_extras stats:")
    print(f"    n                = {extras['n']}")
    print(f"    nnz_a            = {extras['nnz_a']}    (strict-lower, after relabel)")
    print(f"    nnz_l            = {extras['nnz_l']}    (strict-lower L)")
    print(f"    mp_growth_factor = {extras['mp_growth_factor']:.3f}    (nnz_l / nnz_a)")
    print(f"    fillin_edge_index shape = {tuple(extras['fillin_edge_index'].shape)}")
    print(f"    perm shape = {tuple(extras['perm'].shape)}, dtype = {extras['perm'].dtype}")
