#!/bin/bash -l
#
#SBATCH --job-name=cmod-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=1

module load intel mkl python/3.12-conda

TARBALL_PATH="$WORK/venvs/archive/nif-venv.tar.gz"
if [ -f "$TARBALL_PATH" ]; then
  echo "Staging venv from $TARBALL_PATH ..."
  tar --use-compress-program='pigz' -xf "$TARBALL_PATH" -C "$TMPDIR"
  sed -i "s|$WORK/venvs/nif|$TMPDIR/nif|g" "$TMPDIR/nif/bin/activate"
  source "$TMPDIR/nif/bin/activate"
else
  echo "WARNING: tarball not found at $TARBALL_PATH; using NFS venv directly (slow)"
  source "$WORK/venvs/nif/bin/activate"
fi

DATA_DIR="${1:?usage: sbatch run.sh <data-dir> [out-dir]}"
OUT_DIR="${2:-$DATA_DIR}"
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
EXE="${EXE:-$HERE/extract_symbolic}"

[ -x "$EXE" ] || {
  echo "ERROR: extract_symbolic not found/executable at $EXE — run 'make' first"
  exit 1
}
[ -d "$DATA_DIR" ] || {
  echo "ERROR: data dir not found: $DATA_DIR"
  exit 1
}
mkdir -p "$OUT_DIR"

export LD_LIBRARY_PATH="$HERE:$LD_LIBRARY_PATH"

process_one() {
  pt="$1"
  name="$(basename "$pt" .pt)"
  out="$OUT_DIR/$name.symb.npz"
  [ -f "$out" ] && {
    echo "skip  $name"
    return
  } # skip-existing

  mtx="$TMPDIR/$name.mtx"
  psym="$TMPDIR/$name.psym"
  if ! python "$HERE/pt_to_mtx.py" "$pt" "$mtx"; then
    echo "FAIL  $name (pt_to_mtx)"
    rm -f "$mtx"
    return
  fi
  "$EXE" "$mtx" "$psym" > /dev/null 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "FAIL  $name (extract_symbolic rc=$rc)"
    rm -f "$mtx" "$psym"
    return
  fi
  if ! python "$HERE/decode_psym.py" "$psym" "$out"; then
    echo "FAIL  $name (decode_psym)"
    rm -f "$mtx" "$psym"
    return
  fi
  echo "ok    $name"
  rm -f "$mtx" "$psym"
}
export -f process_one
export HERE EXE OUT_DIR

NPROC="${NPROC:-$((${SLURM_CPUS_PER_TASK:-72} / 2))}"
shopt -s nullglob
echo "Processing $(ls "$DATA_DIR"/*.pt 2> /dev/null | wc -l) matrices on $NPROC cores -> $OUT_DIR"

printf '%s\0' "$DATA_DIR"/*.pt | xargs -0 -P "$NPROC" -n 1 bash -c 'process_one "$1"' _

n_pt=$(ls "$DATA_DIR"/*.pt 2> /dev/null | wc -l)
n_npz=$(ls "$OUT_DIR"/*.symb.npz 2> /dev/null | wc -l)
echo "done: $n_npz/$n_pt present in $OUT_DIR"
