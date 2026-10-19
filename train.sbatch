#!/usr/bin/bash
#
#SBATCH --time=48:00:00
#SBATCH --account=caps
#SBATCH --partition=caps-gpu
#SBATCH --nodes=1
#SBATCH --mem=120g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=train_tae
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/jackob/logs/tae_train_%A.out
#SBATCH --error=/scratch/jackob/logs/tae_train_%A.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jackob@illinois.edu
date;hostname;pwd
source ~/.bashrc
#export TORCH_COMPILE_DISABLE=1
#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "$CUDA_VISIBLE_DEVICES"
conda activate tae_refactor

export OMP_NUM_THREADS=16

batchsize="512"
encoder_dim="256"
input_dim="1"
decoder_dim="256"
latent_dim="6"
ode_dim=$encoder_dim
config_name="SELDON_2.0_S_E_UQ_C_test_new_rnn_ode"
config_name="SELDON_2.0_S_E_UQ_C_test_new_rnn_ode_class_conditional"
config_name="SELDON_2.0_S_E_UQ_C_with_transmission_delta"
config_name="SELDON_2.0_S_E_UQ_C_with_transmission_delta_refactored"

HOSTNAME="$(hostname)"
if [[ $HOSTNAME == *"gh"* ]]; then
    sed -i -e 's/projects/taiga/g' "/projects/ncsa/caps/skai/software/Transient-Foundation-Model/TAE/configs/${config_name}.yaml"
else
    sed -i -e 's/taiga/projects/g' "/projects/ncsa/caps/skai/software/Transient-Foundation-Model/TAE/configs/${config_name}.yaml"
fi

#srun python train.py --config-name=light_curve_autoencoder_GRUODE trainer_params.devices=-1 trainer_params.num_nodes=$SLURM_NNODES dataset.config.batch_size=$batchsize model_params.config.encoder.config.input_dim=$input_dim model_params.config.encoder.config.hidden_dim=$encoder_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=BandAwareGRUODEVAE_${batchsize}_${encoder_dim}_${decoder_dim}
#srun python train.py --config-name=${config_name} trainer_params.devices=-1 trainer_params.num_nodes=$SLURM_NNODES dataset.config.batch_size=$batchsize model_params.config.encoder.config.embedder.config.input_dim=$input_dim model_params.config.encoder.config.embedder.config.output_dim=$encoder_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=${config_name}_b${batchsize}_e${encoder_dim}_d${decoder_dim}
#trainer_params.overfit_batches=1

srun python train.py --config-name=${config_name} trainer_params.devices=1 trainer_params.num_nodes=$SLURM_NNODES global_params.latent_dim=$latent_dim dataset.config.batch_size=$batchsize global_params.hidden_size=$encoder_dim global_params.ode_size=$ode_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=${config_name}_l${latent_dim}_b${batchsize}_e${encoder_dim}_o${ode_dim}_d${decoder_dim}_roman_elasticc

rm core*
