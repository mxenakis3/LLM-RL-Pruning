#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1 -C H100
#SBATCH -t 5:00:00
#SBATCH -J PPO
#SBATCH -o %x.out.txt
#SBATCH -e %x.err.txt

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate DL_ENV

python gpurun.py