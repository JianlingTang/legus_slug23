#!/bin/bash

#PBS -P jh2
#PBS -q express
#PBS -l walltime=02:00:00
#PBS -l ncpus=48
#PBS -l mem=100GB
#PBS -l jobfs=100GB
#PBS -l wd
#PBS -N jt_color_mag
#PBS -j oe
#PBS -m bea
#PBS -l storage=scratch/jh2+gdata/jh2
#PBS -M u6735916@anu.edu.au
#PBS -o /home/100/jt4478/slug2/std_io/plot_color_mag.log

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
pip install varname
##### Execute Program #####
python3 plot_color_mag.py /g/data/jh2/jt4478/cluster_slug/tang /g/data/jh2/jt4478/cluster_slug/lib_mass.pdf /g/data/jh2/jt4478/cluster_slug/lib_age.pdf /g/data/jh2/jt4478/cluster_slug/lib_av.pdf /home/100/jt4478/slug_legus_pipeline/completeness_data/ 