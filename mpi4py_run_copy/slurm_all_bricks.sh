#!/bin/bash -l

#SBATCH -p regular
#SBATCH -N 10
#SBATCH -t 05:00:00
#SBATCH --account=desi
###SBATCH --image=driftingpig/obiwan_dr8:step11
#SBATCH -J obiwan
#SBATCH -o ./slurm_output/elg_like_%j.out
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --mail-user=kong.291@osu.edu  
#SBATCH --mail-type=ALL


#Note: in slurm_brick_scheduler, RANDOMS_FROM_FITS needs to be changed everytime you start a new run
#note:only rowstart 0/201 are valid, 101 is not valid
export name_for_run=dr8_10deg2 #elg_ngc_run
export name_for_randoms=ngc_randoms_per_brick
export randoms_db=None #run from a fits file
export dataset=dr8
export rowstart=0
export do_skipids=no
export do_more=no
export minid=1
export object=elg
export nobj=200

export usecores=32
export threads=$usecores
#threads=1
export CSCRATCH_OBIWAN=$CSCRATCH/obiwan_Aug/repos_for_docker
#obiwan paths
export obiwan_data=$CSCRATCH_OBIWAN/obiwan_data 
export obiwan_code=$CSCRATCH_OBIWAN/obiwan_code 
export obiwan_out=$CSCRATCH_OBIWAN/obiwan_out   

# Load production env
#source $CSCRATCH/obiwan_code/obiwan/bin/run_atnersc/bashrc_obiwan

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Protect against astropy configs
export XDG_CONFIG_HOME=/dev/shm
srun -n $SLURM_JOB_NUM_NODES mkdir -p $XDG_CONFIG_HOME/astropy

srun -N 10 -n 20 -c $usecores shifter --image=driftingpig/obiwan_dr8:step11 ./example1.sh
wait
