#!/bin/bash

#PBS -P jh2
#PBS -q normal
#PBS -l walltime=01:00:00
#PBS -l ncpus=48
#PBS -l mem=100GB
#PBS -l jobfs=100GB
#PBS -l wd
#PBS -N jt_test_slug
#PBS -j oe
#PBS -m bea
#PBS -l storage=scratch/jh2+gdata/jh2
#PBS -M u6735916@anu.edu.au
#PBS -o /home/100/jt4478/slug2/std_io/testmdd.log

##### Change to your working directory #####
cd /home/100/jt4478/slug2
module unload python3/3.9.2
module unload python3/3.8.2
module unload python 
module load python3/3.9.2
module load hdf5/1.10.5
pip install astropy
pip install numexpr 
pip install emcee 
pip install h5py
##### Execute Program #####
python3 analyze_catalog_mid_mdd.py /g/data/jh2/jt4478/cluster_slug/tang /home/100/jt4478/slug2/allcomp /g/data/jh2/jt4478/cluster_slug/lib_mass.pdf /g/data/jh2/jt4478/cluster_slug/lib_time.pdf /g/data/jh2/jt4478/cluster_slug/lib_av.pdf ./cluster_catalogs/hlsp_628c.tab ./cluster_catalogs/hlsp_628e.tab --mdd --verbose --niter 5000 --bwphot 0.05 --bwphys 0.05 --outname ngc628_mdd
