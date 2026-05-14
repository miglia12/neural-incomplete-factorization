#!/bin/bash -l
#
#SBATCH --job-name=nif-train-table1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:30:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1

module load cuda/12.9.0 gcc/11.2.0 python/3.12-conda

tar --use-compress-program='pigz' -xf $WORK/venvs/archive/nif-venv.tar.gz -C $TMPDIR
sed -i "s|$WORK/venvs/nif|$TMPDIR/nif|g" $TMPDIR/nif/bin/activate
source $TMPDIR/nif/bin/activate

mkdir -p $TMPDIR/data
cp -r "$SLURM_SUBMIT_DIR/data/Random" $TMPDIR/data/
echo "Data staged: $(du -sh $TMPDIR/data | cut -f1)"

cd $TMPDIR
ln -s "$SLURM_SUBMIT_DIR/results" results
export PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH"

RUN_NAME="table1_synthetic_$(date +%Y%m%d_%H%M%S)"
echo "Run name: $RUN_NAME"
python "$SLURM_SUBMIT_DIR/train.py" \
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
