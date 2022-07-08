#!/bin/bash

#SBATCH --account=ctb-banire
#SBATCH --mail-user=amine.m.remita@gmail.com

module load StdEnv/2020
module load python/3.8
module load imkl/2020.1.217

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

pip install --no-index /home/mremita/project/mremita/Thesis/Software/pyvolve
pip install --no-index /home/mremita/project/mremita/Thesis/Software/evoVGM

# Variables $PROGRAM and CONF_file are initialized with export in running script

$PROGRAM $CONF_file
