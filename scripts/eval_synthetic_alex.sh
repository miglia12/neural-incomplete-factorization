#!/bin/bash -l
#
#SBATCH --job-name=nif-eval-table1
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:45:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

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

CHECKPOINT_DIR="$SLURM_SUBMIT_DIR/results/table1_synthetic_20260513_171331"

echo ""
echo "Baselines (None, Jacobi, IC(0))"
echo ""
python "$SLURM_SUBMIT_DIR/test.py" \
  --model none \
  --n 10000 \
  --subset test \
  --solver cg \
  --device 0 \
  --save \
  --name "eval_baselines_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "NeuralIF"
echo ""
python "$SLURM_SUBMIT_DIR/test.py" \
  --model neuralif \
  --checkpoint "$CHECKPOINT_DIR" \
  --weights best_model \
  --n 10000 \
  --subset test \
  --solver cg \
  --device 0 \
  --save \
  --name "eval_neuralif_$(date +%Y%m%d_%H%M%S)"
