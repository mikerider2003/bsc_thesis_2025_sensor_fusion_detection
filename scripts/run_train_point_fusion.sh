#!/bin/bash
#SBATCH -p GPU              # partition (queue)
#SBATCH -N 1                # number of nodes
#SBATCH -t 0-36:00          # time (D-HH:MM)
#SBATCH -o outputs/Point_Pillars_slurm.%N.%j.out  # STDOUT
#SBATCH -e outputs/Point_Pillars_slurm.%N.%j.err  # STDERR
#SBATCH --gres=gpu:1        # request 1 GPU

# Setup Conda environment
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

# Activate your conda environment
source activate env2

# Navigate to your project directory
cd ~/bsc_thesis_2025_sensor_fusion_detection

# Run your training script
python -m src.training.train_PointFusion