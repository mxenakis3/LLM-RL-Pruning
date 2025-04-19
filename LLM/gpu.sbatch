#!/bin/bash
#SBATCH -JHGX_H100_Example               # Job name  
#SBATCH -N1 --ntasks-per-node=1          # Number of nodes and cores per node required 
#sSBATCH --gres=gpu:H200:1                # GPU type (H100) and number of GPUs 
#SBATCH --gres=gpu:1 -C HX00
#SBATCH --mem-per-gpu=112GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t00:10:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -ores-%j.out
#sSBATCH --mail-type=BEGIN       # Mail preferences 
#sSBATCH --mail-user=slin468@gatech.edu # E-mail address for notifications 
python server_LLM_pool.py
