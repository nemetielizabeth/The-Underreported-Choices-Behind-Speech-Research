#!/bin/bash
#SBATCH --job-name=wavlm
#SBATCH --output=LOGS/%x_%j.out
#SBATCH --error=LOGS/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gpus=1

# Go to the TASKS directory
cd /script/path

# Activate the virtual environment
source /venv/bin/activate

# Run the Python script
python run-wavlm.py

