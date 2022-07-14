#!/bin/bash

## SBATCH parameters are passed within the command

module load StdEnv/2020
module load python/3.9
module load imkl/2022.1.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

# Update here the paths for pyvolve and evovgm
pip install --no-index /home/mremita/project/mremita/Thesis/Software/pyvolve
pip install --no-index /home/mremita/project/mremita/Thesis/Software/evoVGM

# Variables $PROGRAM and CONF_file are initialized with export in running script

$PROGRAM $CONF_file
