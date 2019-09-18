# obiwan-dr8

How to run dr8-obiwan:
everything needed is here:
/global/cscratch1/sd/huikong/obiwan_Aug/repos_for_docker/obiwan_code/dr8

top level script is:
/global/cscratch1/sd/huikong/obiwan_Aug/repos_for_docker/obiwan_code/dr8/mpi4py_run/slurm_all_bricks.sh

IN slurm_all_bricks.sh:
1. L32:export CSCRATCH_OBIWAN=$CSCRATCH/obiwan_Aug/repos_for_docker
define a top level obiwan output folder as $CSCRATCH_OBIWAN, change L32 to this folder name
then:
cd $CSCRATCH_OBIWAN
mkdir obiwan_out

2.L18: export name_for_run=dr8_10deg2
change it to the name of this run

3.L19: export name_for_randoms=ngc_randoms_per_brick
this corresponds to L3 in this file:
/global/cscratch1/sd/huikong/obiwan_Aug/repos_for_docker/obiwan_code/dr8/mpi4py_run/slurm_brick_scheduler.sh
L3:RANDOMS_FROM_FITS=/global/cscratch1/sd/huikong/obiwan_Aug/repos_for_docker/obiwan_out/eboss_elg/$name_for_randoms/brick_${1}.fits
this is a fits file that consists of randoms that obiwan draw from. I don't have a reasonable random input for dr8-elgs currently. 
If you remain these unchanged, it will use the randoms in this folder. (However it's not correct ones for desi)

4./global/cscratch1/sd/huikong/obiwan_Aug/repos_for_docker/obiwan_code/dr8/mpi4py_run/example1.py 
L 10:BRICKSTAT_DIR='/global/cscratch1/sd/huikong/obiwan_Aug/repos_for_docker/obiwan_code/py/obiwan/Drones/obiwan_analysis/preprocess/brickstat/%s/'%name
In such folder, it has UnfinishedBricks.txt, FinishedBricks.txt. They records bricks to be finished, and bricks already finished. You can change this folder to your own folder. 
