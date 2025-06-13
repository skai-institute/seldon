#!/usr/bin/bash
#
#SBATCH --time=08:00:00
#SBATCH --account=jackob-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=60000
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --job-name=train_tae
#SBATCH --gres=gpu:H200:1
#SBATCH --output=/scratch/jackob/logs/tae_train_%A_%a.out
#SBATCH --error=/scratch/jackob/logs/tae_train_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackob@illinois.edu
date;hostname;pwd
source ~/.bashrc
source activate tae

python train.py #--config-name=light_curve_autoencoder_GRU
rm core*
