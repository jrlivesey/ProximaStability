#!/bin/bash

## Job Name
#SBATCH --job-name=REBOUND

## Allocation Definition
## The account and partition options should be the same except in a few cases (e.g. ckpt queue and genpool queue).
#SBATCH --account=stf
#SBATCH --partition=stf

## Resources
## Total number of Nodes
#SBATCH --nodes=1   

## Number of cores per node
#SBATCH --ntasks-per-node=20

## Walltime (3 hours). Do not specify a walltime substantially more than your job needs.
#SBATCH --time=72:00:00

## Memory per node. It is important to specify the memory since the default memory is very small.
## For mox, --mem may be more than 100G depending on the memory of your nodes.
## For ikt, --mem may be 58G or more depending on the memory of your nodes.
## See above section on "Specifying memory" for choices for --mem.
#SBATCH --mem=1G

## Specify the working directory for this job
#SBATCH --chdir=/gscratch/stf/jlivesey/megno_maps

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jlivesey@uw.edu

## export all your environment variables to the batch job session
#SBATCH --export=all

python megno-hyak.py inc
