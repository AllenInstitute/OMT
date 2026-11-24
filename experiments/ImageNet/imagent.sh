#!/bin/bash

#SBATCH -N 1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -p celltypes
#SBATCH -t 12:00:00

# chmod +x imagent.sh
# sbatch imagent.sh

source activate omt

# Embed train split (full ~1.2M images)
python embedding.py --split train

# To embed val split as well, uncomment:
# python embedding.py --split val

# To do a quick test run with 1000 images:
# python embedding.py --split train --max-images 1000