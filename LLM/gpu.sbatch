#!/bin/bash
#SBATCH -JHGX_H100_Example               # Job name  
#SBATCH -N1 --ntasks-per-node=1          # Number of nodes and cores per node required 
#SBATCH --gres=gpu:H200:2                # GPU type (H100) and number of GPUs 
#SBATCH --mem-per-gpu=224GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t00:30:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -ores-%j.out
#sSBATCH --mail-type=BEGIN       # Mail preferences 
#sSBATCH --mail-user=slin468@gatech.edu # E-mail address for notifications 
python trial.py
