#!/usr/bin/bash
#
#SBATCH --time=48:00:00
#SBATCH --account=bezb-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=80000
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=train_tae
#SBATCH --gpus=1
#SBATCH --output=logs/tae_train_%A.out
#SBATCH --error=logs/tae_train_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackob@illinois.edu
date;hostname;pwd
source ~/.bashrc
source activate tae

batchsize="512"
hidden_dim="128"
input_dim="128"
data_dir="/projects/bezb/data/cleaned_data_8_previous_nondetects_zeroed_merged.pkl"
config_file="light_curve_autoencoder_GRUODE"

python train.py --config-name=$config_file dataset.config.batch_size=$batchsize model_params.config.encoder.config.input_dim=$input_dim model_params.config.encoder.config.hidden_dim=$input_dim model_params.config.decoder.config.hidden=$hidden_dim dataset.config.lightcurve_path=$data_dir

rm core*
