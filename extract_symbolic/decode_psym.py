"""
Usage: python decode_psym.py <in.psym> <out.symb.npz> [--pt <matrix.pt>]
       python decode_psym.py --selftest

.psym binary layout (little-endian), as the C `write_psym` emits it:
    "PSYM" | i32 version=3, n, nsuper, nparts, nproc, reorder_method
           | i64 gssubs | f64 recon_res
    i32[n] perm, invp, snode | i32[nsuper] nmod | i32[2*(nparts+1)] ddist
    i32[nsuper+1] xsuper | i64[nsuper+1] xlindx | i32[gssubs] lindx
All arrays are 1-based (Fortran); shifted to 0-based here.
"""
import hashlib
import os
import struct
import sys

import numpy as np

PSYM_MAGIC = b"PSYM"
PSYM_VERSION = 3
_HEADER = struct.Struct("<4s6iqd")  # magic, 6 i32, i64 gssubs, f64 recon_res


def read_psym(path):
    with open(path, "rb") as f:
        head = f.read(_HEADER.size)
        if len(head) != _HEADER.size:
            raise ValueError(f"{path}: truncated header")
        magic, version, n, nsuper, nparts, nproc, reorder_method, gssubs, recon_res = \
            _HEADER.unpack(head)
        if magic != PSYM_MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        if version != PSYM_VERSION:
            raise ValueError(f"{path}: version {version}, expected {PSYM_VERSION}")

        def rd(dt, count):
            a = np.fromfile(f, dtype=dt, count=count)
            if a.size != count:
                raise ValueError(f"{path}: truncated array (got {a.size}, want {count})")
            return a

        perm = rd("<i4", n)
        invp = rd("<i4", n)
        snode = rd("<i4", n)
        nmod = rd("<i4", nsuper)
        ddist = rd("<i4", 2 * (nparts + 1))
        xsuper = rd("<i4", nsuper + 1)
        xlindx = rd("<i8", nsuper + 1)
        lindx = rd("<i4", gssubs)

    return {
        "n": int(n), "nsuper": int(nsuper), "nparts": int(nparts), "nproc": int(nproc),
        "reorder_method": int(reorder_method), "gssubs": int(gssubs),
        "recon_res": float(recon_res),
        "perm": perm.astype(np.int64) - 1,
        "invp": invp.astype(np.int64) - 1,
        "snode": snode.astype(np.int64) - 1,
        "nmod": nmod.astype(np.int64),
        "ddist": ddist.astype(np.int64),       # 1-based supernode pairs; analysis-only
        "xsuper": xsuper.astype(np.int64) - 1,
        "xlindx": xlindx.astype(np.int64) - 1,
        "lindx": lindx.astype(np.int64) - 1,
    }


def _recompute_nmod(p):
    """Re-derive supernode in-degrees from (lindx, snode) — mirrors C check 2."""
    nsuper, xsuper, xlindx, lindx, snode = (
        p["nsuper"], p["xsuper"], p["xlindx"], p["lindx"], p["snode"])
    indeg = np.zeros(nsuper, dtype=np.int64)
    for s in range(nsuper):
        width = int(xsuper[s + 1] - xsuper[s])
        lo, hi = int(xlindx[s]), int(xlindx[s + 1])
        off_rows = lindx[lo + width:hi]  # skip the diagonal block
        if off_rows.size == 0:
            continue
        tgt = snode[off_rows]            # ascending rows -> non-decreasing supernodes
        distinct = tgt[np.concatenate(([True], tgt[1:] != tgt[:-1]))]
        np.add.at(indeg, distinct, 1)
    return indeg


def verify(p):
    """Integrity checks (raise on failure). Serialization guard, not factorization."""
    n, nsuper = p["n"], p["nsuper"]
    perm, invp = p["perm"], p["invp"]
    if not np.array_equal(np.sort(perm), np.arange(n)):
        raise ValueError("perm is not a permutation of 0..n-1")
    if not np.array_equal(invp[perm], np.arange(n)):
        raise ValueError("invp is not the inverse of perm")

    lindx, xlindx = p["lindx"], p["xlindx"]
    if lindx.size and (lindx.min() < 0 or lindx.max() >= n):
        raise ValueError("lindx out of range")
    for s in range(nsuper):
        seg = lindx[int(xlindx[s]):int(xlindx[s + 1])]
        if seg.size > 1 and not np.all(seg[1:] > seg[:-1]):
            raise ValueError(f"lindx not strictly ascending in supernode {s}")

    if not np.array_equal(_recompute_nmod(p), p["nmod"]):
        raise ValueError("re-derived nmod disagrees with stored nmod")


def pattern_hash(edge_index):
    """Order-independent hash of A's sparsity pattern (pairs it to its .pt)."""
    ei = np.asarray(edge_index, dtype=np.int64)
    pairs = np.ascontiguousarray(ei.T[np.lexsort((ei[1], ei[0]))])
    return hashlib.sha256(pairs.tobytes()).hexdigest()


def decode(psym_path, npz_path, pt_path=None):
    p = read_psym(psym_path)
    verify(p)

    a_hash = ""
    if pt_path is not None:
        import torch
        data = torch.load(pt_path, weights_only=False)
        a_hash = pattern_hash(data.edge_index.numpy())

    # Atomic write: a worker killed mid-write (OOM/timeout) must not leave a
    # truncated .npz that skip-existing would later trust. Write a temp, then rename.
    tmp = npz_path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            # symbolic core (0-based, PARDISO/permuted order)
            perm=p["perm"], invp=p["invp"], snode=p["snode"], nmod=p["nmod"],
            ddist=p["ddist"], xsuper=p["xsuper"], xlindx=p["xlindx"], lindx=p["lindx"],
            # scalars / provenance
            n=p["n"], nsuper=p["nsuper"], nparts=p["nparts"], nproc=p["nproc"],
            reorder_method=p["reorder_method"], recon_res=p["recon_res"],
            a_pattern_hash=a_hash, verified=True,
        )
    os.replace(tmp, npz_path)


def _selftest():
    import tempfile, os
    # 5x5, 2 supernodes (cols {0,1} and {2,3,4}); arrays 1-based as PARDISO stores them.
    n, nsuper, nparts = 5, 2, 1
    arrs = dict(perm=[3, 1, 5, 2, 4], invp=[2, 4, 1, 5, 3], snode=[1, 1, 2, 2, 2],
                nmod=[0, 1], ddist=[1, 2, 2, 3], xsuper=[1, 3, 6], xlindx=[1, 5, 8],
                lindx=[1, 2, 4, 5, 3, 4, 5])
    d = tempfile.mkdtemp()
    psym = os.path.join(d, "ex.psym")
    with open(psym, "wb") as f:
        f.write(_HEADER.pack(PSYM_MAGIC, PSYM_VERSION, n, nsuper, nparts, 1, 2,
                             len(arrs["lindx"]), 1e-13))
        for name, dt in [("perm", "<i4"), ("invp", "<i4"), ("snode", "<i4"),
                         ("nmod", "<i4"), ("ddist", "<i4"), ("xsuper", "<i4"),
                         ("xlindx", "<i8"), ("lindx", "<i4")]:
            np.asarray(arrs[name], dt).tofile(f)

    p = read_psym(psym)
    assert p["n"] == 5 and p["nsuper"] == 2 and p["reorder_method"] == 2
    assert np.array_equal(p["perm"], [2, 0, 4, 1, 3])
    assert np.array_equal(p["invp"][p["perm"]], np.arange(5))
    verify(p)  # raises if integrity is off

    npz = os.path.join(d, "ex.symb.npz")
    decode(psym, npz)
    z = np.load(npz)
    assert np.array_equal(z["lindx"], [0, 1, 3, 4, 2, 3, 4])  # 0-based
    assert np.array_equal(z["nmod"], [0, 1]) and bool(z["verified"])

    os.remove(psym); os.remove(npz); os.rmdir(d)
    print("decode_psym self-test: OK (read, verify, store all match known answer)")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
        return
    if len(sys.argv) not in (3, 5):
        sys.exit("usage: decode_psym.py <in.psym> <out.symb.npz> [--pt <matrix.pt>]")
    psym_path, npz_path = sys.argv[1], sys.argv[2]
    pt_path = None
    if len(sys.argv) == 5:
        if sys.argv[3] != "--pt":
            sys.exit("usage: decode_psym.py <in.psym> <out.symb.npz> [--pt <matrix.pt>]")
        pt_path = sys.argv[4]
    decode(psym_path, npz_path, pt_path)


if __name__ == "__main__":
    main()
