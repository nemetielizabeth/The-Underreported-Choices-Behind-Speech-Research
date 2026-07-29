#!/bin/bash
#SBATCH --job-name=opensmile
#SBATCH --output=LOGS/%x_%j.out
#SBATCH --error=LOGS/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --cpus-per-task=1

cd /scripts/directory

# Activate the virtual environment
source /venv/bin/activate

# Run the Python script
python run-opensmile.py

