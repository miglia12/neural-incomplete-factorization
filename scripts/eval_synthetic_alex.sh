#!/bin/bash -l
#
#SBATCH --job-name=nif-eval-table1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:45:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cuda/12.9.0 gcc/11.2.0 python/3.12-conda

test -d "$SLURM_SUBMIT_DIR/data/Random" || {
  echo "ERROR: $SLURM_SUBMIT_DIR/data/Random missing — did you run gen_synthetic?"
  exit 1
}
test -d "$SLURM_SUBMIT_DIR/results" || {
  echo "ERROR: $SLURM_SUBMIT_DIR/results missing — set up the symlink to \$WORK/.../results"
  exit 1
}

CHECKPOINT_DIR="$SLURM_SUBMIT_DIR/results/table1_synthetic_20260513_171331"
test -f "$CHECKPOINT_DIR/best_model.pt" || {
  echo "ERROR: best_model.pt missing at $CHECKPOINT_DIR"
  exit 1
}
test -f "$CHECKPOINT_DIR/config.json" || {
  echo "ERROR: config.json   missing at $CHECKPOINT_DIR"
  exit 1
}

TARBALL_PATH="$WORK/venvs/archive/nif-venv.tar.gz"
if [ -f "$TARBALL_PATH" ]; then
  echo "Staging venv from $TARBALL_PATH ..."
  tar --use-compress-program='pigz' -xf "$TARBALL_PATH" -C $TMPDIR
  sed -i "s|$WORK/venvs/nif|$TMPDIR/nif|g" $TMPDIR/nif/bin/activate
  source $TMPDIR/nif/bin/activate
else
  echo "WARNING: tarball not found at $TARBALL_PATH; using NFS venv directly (will be slow)"
  source $WORK/venvs/nif/bin/activate
fi

mkdir -p $TMPDIR/data
cp -r "$SLURM_SUBMIT_DIR/data/Random" $TMPDIR/data/
echo "Data staged: $(du -sh $TMPDIR/data | cut -f1)"

echo
echo "Baselines (None, Jacobi, IC(0))"
echo
python "$SLURM_SUBMIT_DIR/test.py" \
  --data-root "$TMPDIR/data" \
  --results-root "$SLURM_SUBMIT_DIR/results" \
  --model none \
  --n 10000 \
  --subset test \
  --solver cg \
  --device 0 \
  --save \
  --name "eval_baselines_$(date +%Y%m%d_%H%M%S)"

echo
echo "NeuralIF"
echo
python "$SLURM_SUBMIT_DIR/test.py" \
  --data-root "$TMPDIR/data" \
  --results-root "$SLURM_SUBMIT_DIR/results" \
  --model neuralif \
  --checkpoint "$CHECKPOINT_DIR" \
  --weights best_model \
  --n 10000 \
  --subset test \
  --solver cg \
  --device 0 \
  --save \
  --name "eval_neuralif_$(date +%Y%m%d_%H%M%S)"
