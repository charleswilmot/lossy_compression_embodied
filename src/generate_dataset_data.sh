#!/usr/bin/env bash
#SBATCH --mincpus 10
#SBATCH --mem 40000
#SBATCH -LXserver
#SBATCH --gres gpu:1


srun -u python generate_dataset_data.py "$@"
