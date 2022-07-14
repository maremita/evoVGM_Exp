#!/bin/bash

#SBATCH --account=ctb-banire
#SBATCH --mail-user=amine.m.remita@gmail.com
#SBATCH --job-name=CoV_BP001AKL01
#SBATCH --cpus-per-task=12
#SBATCH --mem=64000M
#SBATCH --time=10:00:00
#SBATCH --error=../exp_jobs/CoV/%j.err
#SBATCH --output=../exp_jobs/CoV/%j.out

module load StdEnv/2020
module load python/3.8
module load scipy-stack
module load imkl/2022.1.0

echo Running $SLURM_JOB_NAME $SLURM_JOB_ID
echo

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

# Copy packages
# Update here the paths for pyvolve and evovgm
cp -a /home/mremita/project/mremita/Thesis/Software/pyvolve/. $SLURM_TMPDIR/pyvolve/
cp -a /home/mremita/project/mremita/Thesis/Software/evoVGM/. $SLURM_TMPDIR/evoVGM/

pip install --no-index $SLURM_TMPDIR/pyvolve/
pip install --no-index $SLURM_TMPDIR/evoVGM/

evovgm.py ../exp_2022_bcb/CoV_BP001AKL01_config.ini
