#!/bin/bash
#SBATCH -t 00:02:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

# Load anaconda module so we have access to modern python interpreter
module load anaconda3/2022.5

# Activate the environment we created with setup_python_env.sh
conda activate hf

python pipelines.py

