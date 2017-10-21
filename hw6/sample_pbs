#!/bin/bash
#PBS -l nodes=1:ppn=16:xk
#PBS -l walltime=12:00:00
#PBS -N GA3C
#PBS -e $PBS_JOBID.err
#PBS -o $PBS_JOBID.out
#PBS -q normal

cd /path/to/ga3c_files

. /opt/modules/default/init/bash # NEEDED to add module commands to shell
module swap PrgEnv-cray PrgEnv-gnu
module load bwpy
module load bwpy-mpi
module load tensorflow
module load cudatoolkit

alias python='python3.4'
export CPATH="${BWPY_INCLUDE_PATH}"
export LIBRARY_PATH="${BWPY_LIBRARY_PATH}"
export PMI_NO_FORK=1
export PMI_NO_PREINITIALIZE=1

beta=1.0
gae=1.0
aprun -n 1 -d 16 -cc none python3.4 GA3C.py GAE_LAMBDA=$gae BETA_START=$beta BETA_END=$beta >& $PBS_JOBID.log
