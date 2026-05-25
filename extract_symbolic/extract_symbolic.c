// extract_symbolic <matrix.mtx> [out.psym]
//
// SPD only (mtype=2). Runs PARDISO symbolic + factorize, verifies the
// pardiso_get_symbolic export, and on success dumps the symbolic core to out.psym.
// A failed check writes no file and returns nonzero.
//
// exit: 0 ok | 1 read/init/reorder | 2 factor | 3 preconditions
//       4 pattern/nmod | 5 reconstruction | 6 dump i/o

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mmio_csr.h"
#include "pardiso.h"

// Exported by libpardiso with no public header (like the pardiso_get_factor_* family);
// declared here as a ctypes binding would.
extern void pardiso_get_symbolic(void* handle[],
                                 int matrix_number,
                                 int* n,
                                 int* nsuper,
                                 int* nparts,
                                 void** xsuper,
                                 void** xlnz,
                                 void** xlindx,
                                 void** lindx,
                                 void** snode,
                                 void** nmod,
                                 void** ddist,
                                 void** perm,
                                 void** invp,
                                 void** ipiv,
                                 void** mc64perm,
                                 void** mc64uv);

extern void pardiso_get_factor_ldl(void* handle[],
                                   int matrix_number,
                                   int* nsuper,
                                   void** la,
                                   void** xsuper,
                                   void** xlnz,
                                   void** xlindx,
                                   void** lindx,
                                   void** perm,
                                   void** ipiv,
                                   void** mc64uv,
                                   void** ddist);

#define N_PROBES 100
#define RECON_TOL 1e-11

// u = A w, A stored as a single triangle (one entry per pair, diagonal once), 0-based CSR.
static void symmetric_spmv(int n,
                           const int* ia,
                           const int* ja,
                           const double* a,
                           const double* w,
                           double* u)
{
  for (int i = 0; i < n; i++)
    u[i] = 0.0;
  for (int i = 0; i < n; i++) {
    for (int k = ia[i]; k < ia[i + 1]; k++) {
      int j = ja[k];
      u[i] += a[k] * w[j];
      if (j != i)
        u[j] += a[k] * w[i];
    }
  }
}

// y = L L^T v, reading the supernodal factor in place.
// Supernode s is a dense m x width column-major panel, m = xlindx[s+1]-xlindx[s].
// Column c (offset off = c-first_col) is lnz[xlnz[c] .. +m-1], where lnz[xlnz[c]+r] is
// row lindx[xlindx[s]+r]: diagonal at r==off, r<off is the upper part (L^T), r>off is L.
// All arrays and vectors are 1-based.
static void apply_factor(int n,
                         int nsuper,
                         const int32_t* XS,
                         const int64_t* XLI,
                         const int32_t* LI,
                         const int64_t* XLNZ,
                         const double* lnz1,
                         const double* v,
                         double* y,
                         double* t)  // scratch, holds L^T v
{
  for (int c = 1; c <= n; c++)
    t[c] = 0.0;
  for (int s = 1; s <= nsuper; s++) {
    int first_col = XS[s];
    for (int c = first_col; c < XS[s + 1]; c++) {
      int64_t off = (int64_t)(c - first_col);
      int64_t lo = XLI[s];
      int64_t m = XLI[s + 1] - lo;
      int64_t base = XLNZ[c];
      double acc = lnz1[base + off] * v[c];  // diagonal: row LI[lo+off] == c
      for (int64_t r = off + 1; r < m; r++)
        acc += lnz1[base + r] * v[LI[lo + r]];
      t[c] = acc;
    }
  }

  for (int c = 1; c <= n; c++)
    y[c] = 0.0;
  for (int s = 1; s <= nsuper; s++) {
    int first_col = XS[s];
    for (int c = first_col; c < XS[s + 1]; c++) {
      int64_t off = (int64_t)(c - first_col);
      int64_t lo = XLI[s];
      int64_t m = XLI[s + 1] - lo;
      int64_t base = XLNZ[c];
      y[c] += lnz1[base + off] * t[c];
      for (int64_t r = off + 1; r < m; r++)
        y[LI[lo + r]] += lnz1[base + r] * t[c];
    }
  }
}

// max over N_PROBES of ||A_perm v - L L^T v|| / ||A_perm v||, pp the 1-based permutation.
static double recon_residual(int n,
                             int nsuper,
                             const int32_t* XS,
                             const int64_t* XLI,
                             const int32_t* LI,
                             const int64_t* XLNZ,
                             const double* lnz1,
                             const int32_t* pp,
                             const int* ia,
                             const int* ja,
                             const double* a)
{
  double* v = malloc((size_t)(n + 1) * sizeof(*v));
  double* y = malloc((size_t)(n + 1) * sizeof(*y));
  double* t = malloc((size_t)(n + 1) * sizeof(*t));
  double* w = malloc((size_t)n * sizeof(*w));
  double* u = malloc((size_t)n * sizeof(*u));

  srand(12345);
  double worst = 0.0;
  for (int probe = 0; probe < N_PROBES; probe++) {
    for (int i = 1; i <= n; i++)
      v[i] = 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0;

    // A_perm v = scatter v into A's order, A w, gather back by pp.
    for (int i = 1; i <= n; i++)
      w[pp[i] - 1] = v[i];
    symmetric_spmv(n, ia, ja, a, w, u);

    apply_factor(n, nsuper, XS, XLI, LI, XLNZ, lnz1, v, y, t);

    double num = 0.0, den = 0.0;
    for (int i = 1; i <= n; i++) {
      double av = u[pp[i] - 1];
      double diff = av - y[i];
      num += diff * diff;
      den += av * av;
    }
    double rel = (den > 0.0) ? sqrt(num / den) : sqrt(num);
    if (rel > worst)
      worst = rel;
  }

  free(v);
  free(y);
  free(t);
  free(w);
  free(u);
  return worst;
}

// Dump the symbolic core straight from PARDISO's pointers (1-based, as stored).
// nproc is the thread count that produced this ordering (it sets nparts).
// reorder_method is iparm[1] (2=METIS, 3=METIS5.1). recon_res is the check-3
// reconstruction residual. ddist holds 2*(nparts+1) entries: a (first,last)
// supernode pair per domain then the separator pair.
// layout (little-endian):
//   "PSYM" | i32 version=3,n,nsuper,nparts,nproc,reorder_method
//          | i64 gssubs | f64 recon_res
//   i32[n] perm,invp,snode | i32[nsuper] nmod | i32[2*(nparts+1)] ddist
//   i32[nsuper+1] xsuper | i64[nsuper+1] xlindx | i32[gssubs] lindx
static int write_psym(const char* path,
                      int32_t n,
                      int32_t nsuper,
                      int32_t nparts,
                      int32_t nproc,
                      int32_t reorder_method,
                      int64_t gssubs,
                      double recon_res,
                      const int32_t* perm,
                      const int32_t* invp,
                      const int32_t* snode,
                      const int32_t* nmod,
                      const int32_t* ddist,
                      const int32_t* xsuper,
                      const int64_t* xlindx,
                      const int32_t* lindx)
{
  FILE* f = fopen(path, "wb");
  if (!f)
    return -1;
  int32_t version = 3;
  size_t ns1 = (size_t)nsuper + 1;
  size_t ddist_len = (size_t)2 * (nparts + 1);
  int ok = 1;
  ok &= (fwrite("PSYM", 1, 4, f) == 4);
  ok &= (fwrite(&version, sizeof(int32_t), 1, f) == 1);
  ok &= (fwrite(&n, sizeof(int32_t), 1, f) == 1);
  ok &= (fwrite(&nsuper, sizeof(int32_t), 1, f) == 1);
  ok &= (fwrite(&nparts, sizeof(int32_t), 1, f) == 1);
  ok &= (fwrite(&nproc, sizeof(int32_t), 1, f) == 1);
  ok &= (fwrite(&reorder_method, sizeof(int32_t), 1, f) == 1);
  ok &= (fwrite(&gssubs, sizeof(int64_t), 1, f) == 1);
  ok &= (fwrite(&recon_res, sizeof(double), 1, f) == 1);
  ok &= (fwrite(perm, sizeof(int32_t), (size_t)n, f) == (size_t)n);
  ok &= (fwrite(invp, sizeof(int32_t), (size_t)n, f) == (size_t)n);
  ok &= (fwrite(snode, sizeof(int32_t), (size_t)n, f) == (size_t)n);
  ok &= (fwrite(nmod, sizeof(int32_t), (size_t)nsuper, f) == (size_t)nsuper);
  ok &= (fwrite(ddist, sizeof(int32_t), ddist_len, f) == ddist_len);
  ok &= (fwrite(xsuper, sizeof(int32_t), ns1, f) == ns1);
  ok &= (fwrite(xlindx, sizeof(int64_t), ns1, f) == ns1);
  ok &= (fwrite(lindx, sizeof(int32_t), (size_t)gssubs, f) == (size_t)gssubs);
  ok &= (fclose(f) == 0);
  return ok ? 0 : -1;
}

int main(int argc, char* argv[])
{
  if (argc != 2 && argc != 3) {
    fprintf(stderr, "Usage: %s <matrix.mtx> [out.psym]\n", argv[0]);
    return 1;
  }
  const char* mtx_path = argv[1];
  const char* psym_path = (argc == 3) ? argv[2] : NULL;

  int n, ncol, nnz;
  int *ia, *ja;
  double* a;
  if (mm_read_mtx_csr(mtx_path, &n, &ncol, &nnz, &ia, &ja, &a) != 0) {
    fprintf(stderr, "Failed to read %s\n", mtx_path);
    return 1;
  }
  if (n <= 0 || nnz <= 0 || n != ncol) {
    fprintf(stderr, "Bad matrix: n=%d ncol=%d nnz=%d\n", n, ncol, nnz);
    return 1;
  }

  // PARDISO wants 1-based CSR; keep the 0-based originals for the reconstruction.
  int* ia1 = malloc((size_t)(n + 1) * sizeof(*ia1));
  int* ja1 = malloc((size_t)nnz * sizeof(*ja1));
  for (int i = 0; i < n + 1; i++)
    ia1[i] = ia[i] + 1;
  for (int i = 0; i < nnz; i++)
    ja1[i] = ja[i] + 1;

  void* pt[64];
  int iparm[64] = {0};
  double dparm[64] = {0.0};
  int mtype = 2;
  int solver = 0;
  int error = 0;

  pardisoinit(pt, &mtype, &solver, iparm, dparm, &error);
  if (error != 0) {
    fprintf(stderr, "pardisoinit error: %d\n", error);
    return 1;
  }

  char* var = getenv("OMP_NUM_THREADS");
  int num_procs = 1;
  if (var != NULL)
    sscanf(var, "%d", &num_procs);
  iparm[2] = num_procs;

  // method PARDISO will order with (mirrors pardiso_state: METIS default unless overridden)
  int32_t reorder_method = (iparm[0] == 0) ? 2 : (int32_t)iparm[1];

  int maxfct = 1, mnum = 1, msglvl = 1, idum, nrhs = 1;
  double ddum;

  printf("Matrix: %s\n", mtx_path);
  printf("  n=%d  nnz=%d  threads=%d  mtype=%d (SPD)\n", n, nnz, num_procs, mtype);

  int phase = 11;
  pardiso(pt,
          &maxfct,
          &mnum,
          &mtype,
          &phase,
          &n,
          a,
          ia1,
          ja1,
          &idum,
          &nrhs,
          iparm,
          &msglvl,
          &ddum,
          &ddum,
          &error,
          dparm);
  if (error != 0) {
    fprintf(stderr, "  REORDER error: %d\n", error);
    return 1;
  }

  phase = 22;
  pardiso(pt,
          &maxfct,
          &mnum,
          &mtype,
          &phase,
          &n,
          a,
          ia1,
          ja1,
          &idum,
          &nrhs,
          iparm,
          &msglvl,
          &ddum,
          &ddum,
          &error,
          dparm);
  if (error != 0) {
    fprintf(stderr, "  FACTOR error: %d\n", error);
    return 2;
  }

  int sn = 0, snsuper = 0, snparts = 0;
  int32_t *xsuper = NULL, *lindx = NULL, *snode = NULL, *nmod = NULL, *ddist = NULL;
  int32_t *perm = NULL, *invp = NULL, *ipiv = NULL, *mc64perm = NULL;
  int64_t *xlnz = NULL, *xlindx = NULL;
  double* mc64uv = NULL;
  pardiso_get_symbolic(pt,
                       mnum,
                       &sn,
                       &snsuper,
                       &snparts,
                       (void**)&xsuper,
                       (void**)&xlnz,
                       (void**)&xlindx,
                       (void**)&lindx,
                       (void**)&snode,
                       (void**)&nmod,
                       (void**)&ddist,
                       (void**)&perm,
                       (void**)&invp,
                       (void**)&ipiv,
                       (void**)&mc64perm,
                       (void**)&mc64uv);

  // get_factor_ldl is the only value extractor on the new layout; lnz feeds check 3.
  int ldl_nsuper = 0;
  double* lnz = NULL;
  int32_t *l_xsuper = NULL, *l_lindx = NULL, *l_perm = NULL, *l_ipiv = NULL, *l_ddist = NULL;
  int64_t *l_xlnz = NULL, *l_xlindx = NULL;
  double* l_mc64uv = NULL;
  pardiso_get_factor_ldl(pt,
                         mnum,
                         &ldl_nsuper,
                         (void**)&lnz,
                         (void**)&l_xsuper,
                         (void**)&l_xlnz,
                         (void**)&l_xlindx,
                         (void**)&l_lindx,
                         (void**)&l_perm,
                         (void**)&l_ipiv,
                         (void**)&l_mc64uv,
                         (void**)&l_ddist);

  int rc = 0;
  double recon_res = 0.0;  // check-3 residual, stamped into the dump

  // 1-based views, valid until phase -1.
  const int32_t* XS = xsuper - 1;
  const int64_t* XLI = xlindx - 1;
  const int32_t* LI = lindx - 1;
  const int32_t* SN = snode - 1;
  const int32_t* NMOD = nmod - 1;

  // [1] preconditions: dims, both exports point at the same arrays, perm is a bijection
  // with inverse invp, and the SPD case is clean (no pivots, identity matching/scaling).
  printf("\n[1] preconditions (dims, perm bijection, SPD no-pivot)\n");
  if (sn != n || snsuper <= 0 || !xsuper || !xlnz || !xlindx || !lindx || !snode || !nmod ||
      !perm || !invp || !lnz) {
    printf("    FAIL: null pointer or dimension mismatch (n=%d sn=%d nsuper=%d)\n", n, sn, snsuper);
    rc = 3;
    goto done;
  }
  if (l_xsuper != xsuper || l_xlnz != xlnz || l_xlindx != xlindx || l_lindx != lindx ||
      l_perm != perm || ldl_nsuper != snsuper) {
    printf("    FAIL: get_symbolic and get_factor_ldl disagree on internal arrays\n");
    rc = 3;
    goto done;
  }
  if (iparm[13] != 0) {
    printf("    FAIL: %d perturbed pivots (iparm[13]) - not a clean SPD factor\n", iparm[13]);
    rc = 3;
    goto done;
  }
  for (int i = 0; mc64perm && i < n; i++) {
    if (mc64perm[i] != i + 1) {
      printf("    FAIL: matching permutation non-identity at %d (=%d)\n", i, mc64perm[i]);
      rc = 3;
      goto done;
    }
  }
  for (int i = 0; mc64uv && i < 2 * n; i++) {
    if (mc64uv[i] != 1.0) {
      printf("    FAIL: scaling mc64uv non-unit at %d (=%g)\n", i, mc64uv[i]);
      rc = 3;
      goto done;
    }
  }
  {
    char* seen = calloc((size_t)n, sizeof(*seen));
    for (int i = 0; i < n; i++) {
      int p = perm[i];
      if (p < 1 || p > n || seen[p - 1] || invp[p - 1] != i + 1) {
        printf("    FAIL: perm not a bijection / invp not its inverse (i=%d perm=%d)\n", i, p);
        free(seen);
        rc = 3;
        goto done;
      }
      seen[p - 1] = 1;
    }
    free(seen);
  }
  printf("    OK: n=%d nsuper=%d nparts=%d; perm bijection; 0 pivots; identity matching/scaling\n",
         n,
         snsuper,
         snparts);

  // [2] lindx must be strictly ascending and in range (the Python fill-in/cmod walk
  // relies on it), and the in-degree we derive from it must equal PARDISO's nmod[].
  printf("\n[2] pattern structure + nmod cross-check\n");
  {
    int32_t* indeg = calloc((size_t)(snsuper + 1), sizeof(*indeg));
    for (int s = 1; s <= snsuper; s++) {
      int width = XS[s + 1] - XS[s];
      int64_t lo = XLI[s], hi = XLI[s + 1];
      for (int64_t r = lo; r < hi; r++) {
        if (LI[r] < 1 || LI[r] > n) {
          printf("    FAIL: lindx out of range at %lld (=%d)\n", (long long)r, LI[r]);
          free(indeg);
          rc = 4;
          goto done;
        }
        if (r > lo && LI[r] <= LI[r - 1]) {
          printf("    FAIL: lindx not strictly ascending in supernode %d\n", s);
          free(indeg);
          rc = 4;
          goto done;
        }
      }
      // one incoming edge per distinct (consecutive) target supernode among off-diag rows
      int prev = -1;
      for (int64_t r = lo + width; r < hi; r++) {
        int tgt = SN[LI[r]];
        if (tgt != prev) {
          indeg[tgt]++;
          prev = tgt;
        }
      }
    }
    for (int s = 1; s <= snsuper; s++) {
      if (indeg[s] != NMOD[s]) {
        printf("    FAIL: supernode %d in-degree %d != nmod %d\n", s, indeg[s], NMOD[s]);
        free(indeg);
        rc = 4;
        goto done;
      }
    }
    free(indeg);
    printf("    OK: lindx ascending/in-range; derived in-degree matches nmod[] (%d supernodes)\n",
           snsuper);
  }

  // [3] end-to-end: rebuilt L L^T must reproduce A under the forward perm. Proves
  // pattern, perm and values are mutually consistent.
  printf("\n[3] numeric reconstruction  ||A_perm v - L L^T v|| / ||A_perm v||\n");
  {
    recon_res = recon_residual(n, snsuper, XS, XLI, LI, xlnz - 1, lnz - 1, perm - 1, ia, ja, a);
    printf("    residual: %.3e  (forward perm, tol %.0e, %d probes)\n",
           recon_res,
           (double)RECON_TOL,
           N_PROBES);
    if (recon_res >= RECON_TOL) {
      printf("    FAIL: reconstruction does not reproduce A within tolerance\n");
      rc = 5;
      goto done;
    }
    printf("    OK: exported pattern + perm + values reproduce A\n");
  }

  if (psym_path) {
    int64_t gssubs = xlindx[snsuper] - 1;  // lindx length (xlindx is 1-based)
    if (write_psym(psym_path,
                   n,
                   snsuper,
                   snparts,
                   num_procs,
                   reorder_method,
                   gssubs,
                   recon_res,
                   perm,
                   invp,
                   snode,
                   nmod,
                   ddist,
                   xsuper,
                   xlindx,
                   lindx) != 0) {
      printf("\n    FAIL: could not write %s\n", psym_path);
      rc = 6;
      goto done;
    }
    printf("\n  dumped: %s  (perm, invp, snode, nmod, ddist, xsuper, xlindx, lindx; "
           "gssubs=%lld, nproc=%d)\n",
           psym_path,
           (long long)gssubs,
           num_procs);
  }

  printf("\n  verdict: PASS\n");

done:
  if (rc != 0)
    printf("\n  verdict: FAIL (rc=%d)\n", rc);

  phase = -1;
  pardiso(pt,
          &maxfct,
          &mnum,
          &mtype,
          &phase,
          &n,
          &ddum,
          ia1,
          ja1,
          &idum,
          &nrhs,
          iparm,
          &msglvl,
          &ddum,
          &ddum,
          &error,
          dparm);

  free(ia);
  free(ja);
  free(a);
  free(ia1);
  free(ja1);
  return rc;
}
