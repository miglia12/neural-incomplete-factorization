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

tar --use-compress-program='pigz' -xf $WORK/venvs/archive/nif-venv.tar.gz -C $TMPDIR
sed -i "s|/home/atuin/j101df/j101df12/venvs/nif|$TMPDIR/nif|g" $TMPDIR/nif/bin/activate
source $TMPDIR/nif/bin/activate

cd "$SLURM_SUBMIT_DIR"
python apps/synthetic.py
