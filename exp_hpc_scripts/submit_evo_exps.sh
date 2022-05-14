#!/bin/bash

#SBATCH --account=def-banire
#SBATCH --mail-user=amine.m.remita@gmail.com

#SBATCH --nodelist=blg8598
#SBATCH --partition=c-slarge

#module load StdEnv/2020
#module load arch/avx2

module load python/3.7

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

pip install --no-index /home/mremita/project/mremita/Thesis/Software/pyvolve
pip install --no-index /home/mremita/project/mremita/Thesis/Software/evoSubVGMSA

# Variables $PROGRAM and CONF_file are initialized with export in running script

$PROGRAM $CONF_file
