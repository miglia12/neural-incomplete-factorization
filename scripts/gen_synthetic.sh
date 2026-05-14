#!/bin/bash -l
#
#SBATCH --job-name=nif-datagen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gcc/11.2.0 python/3.12-conda

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

cd "$SLURM_SUBMIT_DIR"
python apps/synthetic.py
