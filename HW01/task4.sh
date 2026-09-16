#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J FirstSlurm
#SBATCH -c 2
#SBATCH -t 0-00:01:00
#SBATCH -o FirstSlurm.out -e FirstSlurm.err

hostname
