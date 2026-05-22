#!/bin/bash -l
#
#SBATCH --job-name=nif-eval-poisson
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#
# Evaluate a NeuralIF model on the POISSON test set.
#
#   sbatch scripts/eval_poisson_alex.sh
#       No argument -> original behavior: baselines (None/Jacobi/IC(0)) + the
#       in-distribution Poisson model (DEFAULT_CKPT below).
#
#   sbatch scripts/eval_poisson_alex.sh <checkpoint_dir>
#       Argument -> cross-eval: run ONLY that model on the Poisson test set,
#       skip baselines (already produced by the no-argument run).
#
# Result dirs: results/eval_<trained_on>_on_poisson_<ts>/  (+ eval_baselines_on_poisson_<ts>/)

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-$(nproc)}

module load cuda/12.9.0 gcc/11.2.0 python/3.12-conda

DEFAULT_CKPT="$SLURM_SUBMIT_DIR/results/poisson_20260522_133332"
if [ -z "$1" ]; then
  CHECKPOINT_DIR="$DEFAULT_CKPT"
  RUN_BASELINES=1
else
  CHECKPOINT_DIR="$1"
  RUN_BASELINES=0
fi

test -f "$CHECKPOINT_DIR/best_model.pt" || {
  echo "ERROR: best_model.pt missing at $CHECKPOINT_DIR"
  exit 1
}
test -f "$CHECKPOINT_DIR/config.json" || {
  echo "ERROR: config.json missing at $CHECKPOINT_DIR"
  exit 1
}
test -d "$SLURM_SUBMIT_DIR/data/Poisson" || {
  echo "ERROR: data/Poisson missing — did you run gen_poisson?"
  exit 1
}
test -d "$SLURM_SUBMIT_DIR/results" || {
  echo 'ERROR: results/ missing — set up the symlink to $WORK/.../results'
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

mkdir -p $TMPDIR/data/Poisson
cp -r "$SLURM_SUBMIT_DIR/data/Poisson/test" $TMPDIR/data/Poisson/
echo "Data staged: $(du -sh $TMPDIR/data/Poisson | cut -f1)"

# Derive the model's training dataset from its config.
TRAIN_DS=$(python -c "import json; print(json.load(open('$CHECKPOINT_DIR/config.json')).get('dataset','unknown'))")
TS=$(date +%Y%m%d_%H%M%S)
echo "Model trained on: '$TRAIN_DS'  |  Evaluating on: 'poisson'"
[ "$TRAIN_DS" = "poisson" ] && echo "(in-distribution)" || echo "(CROSS-distribution)"

if [ "$RUN_BASELINES" = "1" ]; then
  echo
  echo "Baselines (None, Jacobi, IC(0)) on poisson test set"
  echo
  python "$SLURM_SUBMIT_DIR/test.py" \
    --data-root "$TMPDIR/data" \
    --results-root "$SLURM_SUBMIT_DIR/results" \
    --dataset poisson \
    --model none \
    --n 0 \
    --subset test \
    --solver cg \
    --device 0 \
    --save \
    --name "eval_baselines_on_poisson_${TS}"
fi

echo
echo "NeuralIF ('${TRAIN_DS}'-trained) on poisson test set"
echo
python "$SLURM_SUBMIT_DIR/test.py" \
  --data-root "$TMPDIR/data" \
  --results-root "$SLURM_SUBMIT_DIR/results" \
  --dataset poisson \
  --model neuralif \
  --checkpoint "$CHECKPOINT_DIR" \
  --weights best_model \
  --n 0 \
  --subset test \
  --solver cg \
  --device 0 \
  --save \
  --name "eval_${TRAIN_DS}_on_poisson_${TS}"

echo "Done. Result: results/eval_${TRAIN_DS}_on_poisson_${TS}/"
