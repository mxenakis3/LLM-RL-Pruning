#!/bin/bash
#SBATCH --ntasks 8
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:1 -C H100
#SBATCH -t 1:00:00
#SBATCH -J PPO
#SBATCH -o %x.out.txt
#SBATCH -e %x.err.txt

cd $SLURM_SUBMIT_DIR

module load anaconda3
conda activate DL_ENV

python gpurun.py