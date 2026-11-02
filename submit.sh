#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 6
#SBATCH --mem=60G
#SBATCH -J spine
#SBATCH -p gpu
#SBATCH --gres gpu:rtx8000:1
#SBATCH --ntasks-per-node=1
#SBATCH --time 10-00
#SBATCH --output slurm_out/%j.out
#SBATCH --error slurm_out/%j.err

python CLI.py fit --config conf/trainer.yaml "$@"
