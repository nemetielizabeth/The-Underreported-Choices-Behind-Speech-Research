#!/bin/bash
#SBATCH --job-name=roberta
#SBATCH --output=LOGS/%x_%j.out
#SBATCH --error=LOGS/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gpus=1

cd /scripts/directory

# Activate the virtual environment
source /venv/bin/activate

# Run the Python script
python run-roberta.py
