#!/bin/bash -l
#
#SBATCH --job-name=nif-train-random
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1

module load cuda/12.9.0 gcc/11.2.0 python/3.12-conda

test -d "$SLURM_SUBMIT_DIR/data/Random" || {
  echo "ERROR: $SLURM_SUBMIT_DIR/data/Random missing — did you run gen_synthetic?"
  exit 1
}
test -d "$SLURM_SUBMIT_DIR/results" || {
  echo "ERROR: $SLURM_SUBMIT_DIR/results missing — set up the symlink to \$WORK/.../results"
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

RUN_NAME="random_$(date +%Y%m%d_%H%M%S)"
echo "Run name: $RUN_NAME"
python "$SLURM_SUBMIT_DIR/train.py" \
  --data-root "$TMPDIR/data" \
  --results-root "$SLURM_SUBMIT_DIR/results" \
  --model neuralif \
  --n 10000 \
  --num_epochs 50 \
  --batch_size 5 \
  --device 0 \
  --save \
  --augment_nodes \
  --activation tanh \
  --graph_norm \
  --name "$RUN_NAME"

echo "Training complete. Checkpoints at $SLURM_SUBMIT_DIR/results/$RUN_NAME/"
