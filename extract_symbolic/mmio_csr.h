/* Header-Only Matrix Market to CSR I/O Library
 * Fully compatible with the official NIST ANSI C mmio specification.
 * http://math.nist.gov/MatrixMarket/
 */

#ifndef MMIO_FAST_CSR_H
#define MMIO_FAST_CSR_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

typedef char MM_typecode[4];

#define mm_is_matrix(typecode) ((typecode)[0] == 'M')
#define mm_is_sparse(typecode) ((typecode)[1] == 'C')
#define mm_is_coordinate(typecode) ((typecode)[1] == 'C')
#define mm_is_dense(typecode) ((typecode)[1] == 'A')
#define mm_is_array(typecode) ((typecode)[1] == 'A')
#define mm_is_complex(typecode) ((typecode)[2] == 'C')
#define mm_is_real(typecode) ((typecode)[2] == 'R')
#define mm_is_pattern(typecode) ((typecode)[2] == 'P')
#define mm_is_integer(typecode) ((typecode)[2] == 'I')
#define mm_is_symmetric(typecode) ((typecode)[3] == 'S')
#define mm_is_general(typecode) ((typecode)[3] == 'G')
#define mm_is_skew(typecode) ((typecode)[3] == 'K')
#define mm_is_hermitian(typecode) ((typecode)[3] == 'H')

#define mm_clear_typecode(typecode) \
  ((*typecode)[0] = (*typecode)[1] = (*typecode)[2] = ' ', (*typecode)[3] = 'G')
#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)

#define MM_COULD_NOT_READ_FILE 11
#define MM_PREMATURE_EOF 12
#define MM_NOT_MTX 13
#define MM_NO_HEADER 14
#define MM_UNSUPPORTED_TYPE 15
#define MM_LINE_TOO_LONG 16
#define MM_COULD_NOT_WRITE_FILE 17

static inline int mm_read_banner(FILE* f, MM_typecode* matcode)
{
  char line[MM_MAX_LINE_LENGTH];
  char banner[MM_MAX_TOKEN_LENGTH], mtx[MM_MAX_TOKEN_LENGTH];
  char crd[MM_MAX_TOKEN_LENGTH], data_type[MM_MAX_TOKEN_LENGTH];
  char storage_scheme[MM_MAX_TOKEN_LENGTH];
  char* p;

  mm_clear_typecode(matcode);
  if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
    return MM_PREMATURE_EOF;
  if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme) != 5)
    return MM_PREMATURE_EOF;

  for (p = mtx; *p != '\0'; *p = tolower(*p), p++)
    ;
  for (p = crd; *p != '\0'; *p = tolower(*p), p++)
    ;
  for (p = data_type; *p != '\0'; *p = tolower(*p), p++)
    ;
  for (p = storage_scheme; *p != '\0'; *p = tolower(*p), p++)
    ;

  if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
    return MM_NO_HEADER;
  if (strcmp(mtx, "matrix") != 0)
    return MM_UNSUPPORTED_TYPE;

  (*matcode)[0] = 'M';
  (*matcode)[1] = (strcmp(crd, "coordinate") == 0) ? 'C' : 'A';

  if (strcmp(data_type, "real") == 0)
    (*matcode)[2] = 'R';
  else if (strcmp(data_type, "complex") == 0)
    (*matcode)[2] = 'C';
  else if (strcmp(data_type, "pattern") == 0)
    (*matcode)[2] = 'P';
  else if (strcmp(data_type, "integer") == 0)
    (*matcode)[2] = 'I';
  else
    return MM_UNSUPPORTED_TYPE;

  if (strcmp(storage_scheme, "general") == 0)
    (*matcode)[3] = 'G';
  else if (strcmp(storage_scheme, "symmetric") == 0)
    (*matcode)[3] = 'S';
  else if (strcmp(storage_scheme, "hermitian") == 0)
    (*matcode)[3] = 'H';
  else if (strcmp(storage_scheme, "skew-symmetric") == 0)
    (*matcode)[3] = 'K';
  else
    return MM_UNSUPPORTED_TYPE;

  return 0;
}

static inline int mm_read_mtx_crd_size(FILE* f, int* M, int* N, int* nz)
{
  char line[MM_MAX_LINE_LENGTH];
  int num_items_read;
  *M = *N = *nz = 0;
  do {
    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)
      return MM_PREMATURE_EOF;
  } while (line[0] == '%');

  if (sscanf(line, "%d %d %d", M, N, nz) == 3)
    return 0;
  else {
    do {
      num_items_read = fscanf(f, "%d %d %d", M, N, nz);
      if (num_items_read == EOF)
        return MM_PREMATURE_EOF;
    } while (num_items_read != 3);
  }
  return 0;
}

static inline void quicksort_csr(int* cols, double* vals, int low, int high)
{
  if (low < high) {
    int pivot = cols[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
      if (cols[j] < pivot) {
        i++;
        int tc = cols[i];
        cols[i] = cols[j];
        cols[j] = tc;
        double tv = vals[i];
        vals[i] = vals[j];
        vals[j] = tv;
      }
    }
    int tc = cols[i + 1];
    cols[i + 1] = cols[high];
    cols[high] = tc;
    double tv = vals[i + 1];
    vals[i + 1] = vals[high];
    vals[high] = tv;

    int pi = i + 1;
    quicksort_csr(cols, vals, low, pi - 1);
    quicksort_csr(cols, vals, pi + 1, high);
  }
}

static inline int mm_read_mtx_csr(const char* fname,
                                  int* M,
                                  int* N,
                                  int* final_nz,
                                  int** row_ptr,
                                  int** col_ind,
                                  double** val)
{
  FILE* f = fopen(fname, "r");
  if (!f)
    return MM_COULD_NOT_READ_FILE;

  MM_typecode matcode;
  if (mm_read_banner(f, &matcode) != 0) {
    fclose(f);
    return MM_NO_HEADER;
  }
  if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
    fclose(f);
    return MM_UNSUPPORTED_TYPE;
  }

  int nz_coo;
  if (mm_read_mtx_crd_size(f, M, N, &nz_coo) != 0) {
    fclose(f);
    return MM_PREMATURE_EOF;
  }

  int is_sym = mm_is_symmetric(matcode) || mm_is_skew(matcode) || mm_is_hermitian(matcode);
  int is_pattern = mm_is_pattern(matcode);

  int* I_coo = (int*)malloc(nz_coo * sizeof(int));
  int* J_coo = (int*)malloc(nz_coo * sizeof(int));
  double* val_coo = is_pattern ? NULL : (double*)malloc(nz_coo * sizeof(double));

  char line[MM_MAX_LINE_LENGTH];
  *final_nz = 0;

  for (int i = 0; i < nz_coo; i++) {
    if (!fgets(line, sizeof(line), f)) {
      fclose(f);
      return MM_PREMATURE_EOF;
    }

    if (line[0] == '%' || line[0] == '\n') {
      i--;
      continue;
    }

    char* ptr = line;
    I_coo[i] = strtol(ptr, &ptr, 10) - 1;
    J_coo[i] = strtol(ptr, &ptr, 10) - 1;

    if (!is_pattern) {
      val_coo[i] = strtod(ptr, NULL);
    }

    if (I_coo[i] <= J_coo[i] || is_sym)
      (*final_nz)++;
  }
  fclose(f);

  *row_ptr = (int*)calloc((*M + 1), sizeof(int));
  *col_ind = (int*)malloc(*final_nz * sizeof(int));
  *val = (double*)malloc(*final_nz * sizeof(double));

  for (int i = 0; i < nz_coo; i++) {
    int r = I_coo[i];
    int c = J_coo[i];
    int ur = (is_sym && r > c) ? c : r;
    (*row_ptr)[ur + 1]++;
  }

  for (int i = 0; i < *M; i++) {
    (*row_ptr)[i + 1] += (*row_ptr)[i];
  }

  int* offset = (int*)calloc(*M, sizeof(int));
  for (int i = 0; i < nz_coo; i++) {
    int r = I_coo[i];
    int c = J_coo[i];
    double v = is_pattern ? 1.0 : val_coo[i];

    if (is_sym && r > c) {
      int dest = (*row_ptr)[c] + offset[c]++;
      (*col_ind)[dest] = r;
      (*val)[dest] = mm_is_skew(matcode) ? -v : v;
    }
    else {
      int dest = (*row_ptr)[r] + offset[r]++;
      (*col_ind)[dest] = c;
      (*val)[dest] = v;
    }
  }

  for (int i = 0; i < *M; i++) {
    int start = (*row_ptr)[i];
    int end = (*row_ptr)[i + 1];
    if (end - start > 1) {
      quicksort_csr(*col_ind, *val, start, end - 1);
    }
  }

  free(offset);
  free(I_coo);
  free(J_coo);
  if (val_coo)
    free(val_coo);

  return 0;
}

#endif
