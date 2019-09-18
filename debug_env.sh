#!/bin/bash -l
#3531m015
export name_for_run=dr8_test
export randoms_db=None
export dataset=dr8
export rowstart=1
export do_skipids=no
export do_more=no
export minid=1
export object=elg
export nobj=200

export usecores=1
export threads=1
export CSCRATCH=/global/cscratch1/sd/huikong
export CSCRATCH_OBIWAN=$CSCRATCH/obiwan_Aug/repos_for_docker

export obiwan_data=$CSCRATCH_OBIWAN/obiwan_data
export obiwan_code=$CSCRATCH_OBIWAN/obiwan_code
export obiwan_out=$CSCRATCH_OBIWAN/obiwan_out

shifter --image=driftingpig/obiwan_dr8:step11 /bin/bash
