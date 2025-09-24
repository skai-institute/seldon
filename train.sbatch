#!/usr/bin/bash
#
#SBATCH --time=12:00:00
#SBATCH --account=jackob-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --mem=150g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=train_tae
#SBATCH --gres=gpu:A100:1
#SBATCH --output=/scratch/jackob/logs/tae_train_%A.out
#SBATCH --error=/scratch/jackob/logs/tae_train_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackob@illinois.edu
date;hostname;pwd
source ~/.bashrc
source activate tae

export OMP_NUM_THREADS=24

batchsize="512"
encoder_dim="192"
input_dim="1"
decoder_dim="192"

srun python train.py --config-name=light_curve_autoencoder_GRUODE dataset.config.batch_size=$batchsize model_params.config.encoder.config.input_dim=$input_dim model_params.config.encoder.config.hidden_dim=$encoder_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=BandAwareGRUODEVAE_${batchsize}_${encoder_dim}_${decoder_dim}
#python train.py --config-name=light_curve_autoencoder_GRU dataset.config.batch_size=$batchsize model_params.config.encoder.config.input_dim=$input_dim model_params.config.encoder.config.hidden_dim=$hidden_dim model_params.config.decoder.config.hidden=$hidden_dim logging_params.name=BandAwareGRUVAE_${batchsize}_${input_dim}_${hidden_dim}
#python train.py --config-name=light_curve_autoencoder_PureDeepSets dataset.config.batch_size=$batchsize model_params.config.encoder.config.input_dim=$input_dim model_params.config.encoder.config.hidden_dim=$hidden_dim model_params.config.decoder.config.hidden=$hidden_dim logging_params.name=BandAwareDeepSetVAE_${batchsize}_${input_dim}_${hidden_dim}

#python train.py model_params.config.is_adversarial=True logging_params.name=BandAwareDeepSetVAE_adv #& 
#python train.py model_params.config.is_adversarial=False logging_params.name=BandAwareDeepSetVAE_No_adv #--config-name=light_curve_autoencoder_GRU
rm core*
