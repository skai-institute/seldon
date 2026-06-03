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
latent_dim="8"
ode_dim="256" #$encoder_dim
config_name="SELDON_2.0_S_E_UQ_C_test_new_rnn_ode"
config_name="SELDON_2.0_S_E_UQ_C_test_new_rnn_ode_class_conditional"
config_name="SELDON_2.0_S_E_UQ_C_with_transmission_delta"
config_name="SELDON_2.0_S_E_UQ_C_with_transmission_delta_refactored"

sed -i -e 's/taiga/projects/g' /projects/ncsa/caps/skai/software/Transient-Foundation-Model/TAE/configs/SELDON_2.0_S_E_UQ_C_with_transmission_delta_refactored.yaml


#srun python train.py --config-name=light_curve_autoencoder_GRUODE trainer_params.devices=-1 trainer_params.num_nodes=$SLURM_NNODES dataset.config.batch_size=$batchsize model_params.config.encoder.config.input_dim=$input_dim model_params.config.encoder.config.hidden_dim=$encoder_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=BandAwareGRUODEVAE_${batchsize}_${encoder_dim}_${decoder_dim}
#srun python train.py --config-name=${config_name} trainer_params.devices=-1 trainer_params.num_nodes=$SLURM_NNODES dataset.config.batch_size=$batchsize model_params.config.encoder.config.embedder.config.input_dim=$input_dim model_params.config.encoder.config.embedder.config.output_dim=$encoder_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=${config_name}_b${batchsize}_e${encoder_dim}_d${decoder_dim}
#trainer_params.overfit_batches=1

srun python train.py --config-name=${config_name} trainer_params.devices=1 trainer_params.num_nodes=$SLURM_NNODES global_params.latent_dim=$latent_dim dataset.config.batch_size=$batchsize global_params.hidden_size=$encoder_dim global_params.ode_size=$ode_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=${config_name}_l${latent_dim}_b${batchsize}_e${encoder_dim}_o${ode_dim}_d${decoder_dim}

#latent_dim="8"
#python train.py --config-name=${config_name} trainer_params.devices=-1 trainer_params.num_nodes=$SLURM_NNODES global_params.latent_dim=$latent_dim dataset.config.batch_size=$batchsize model_params.config.encoder.config.embedder.config.input_dim=$input_dim global_params.hidden_size=$encoder_dim global_params.ode_size=$ode_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=${config_name}_l${latent_dim}_b${batchsize}_e${encoder_dim}_o${ode_dim}_d${decoder_dim} &

#latent_dim="12"
#python train.py --config-name=${config_name} trainer_params.devices=-1 trainer_params.num_nodes=$SLURM_NNODES global_params.latent_dim=$latent_dim dataset.config.batch_size=$batchsize model_params.config.encoder.config.embedder.config.input_dim=$input_dim global_params.hidden_size=$encoder_dim global_params.ode_size=$ode_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=${config_name}_l${latent_dim}_b${batchsize}_e${encoder_dim}_o${ode_dim}_d${decoder_dim} &

#latent_dim="16"
#python train.py --config-name=${config_name} trainer_params.devices=-1 trainer_params.num_nodes=$SLURM_NNODES global_params.latent_dim=$latent_dim dataset.config.batch_size=$batchsize model_params.config.encoder.config.embedder.config.input_dim=$input_dim global_params.hidden_size=$encoder_dim global_params.ode_size=$ode_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=${config_name}_l${latent_dim}_b${batchsize}_e${encoder_dim}_o${ode_dim}_d${decoder_dim} &


#latent_dim="24"
#python train.py --config-name=${config_name} trainer_params.devices=-1 trainer_params.num_nodes=$SLURM_NNODES global_params.latent_dim=$latent_dim dataset.config.batch_size=$batchsize model_params.config.encoder.config.embedder.config.input_dim=$input_dim global_params.hidden_size=$encoder_dim global_params.ode_size=$ode_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=${config_name}_l${latent_dim}_b${batchsize}_e${encoder_dim}_o${ode_dim}_d${decoder_dim}

exit
for l in "6" "8"
do 
    latent_dim=$l
    #for e in "128" "384"
    #do 
    encoder_dim="128" #$e
    ode_dim="128" #$e

    #for d in "96"
    #do
    decoder_dim="32" #$d
    python train.py --config-name=${config_name} trainer_params.devices=-1 trainer_params.num_nodes=$SLURM_NNODES global_params.latent_dim=$latent_dim dataset.config.batch_size=$batchsize model_params.config.encoder.config.embedder.config.input_dim=$input_dim global_params.hidden_size=$encoder_dim global_params.ode_size=$ode_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=${config_name}_l${latent_dim}_b${batchsize}_e${encoder_dim}_o${ode_dim}_d${decoder_dim} &
    sleep 60
    #done

    #done

done

latent_dim="8"
encoder_dim="96"
decoder_dim="32"
ode_dim="96"
wait;
#python train.py --config-name=${config_name} trainer_params.devices=-1 trainer_params.num_nodes=$SLURM_NNODES global_params.latent_dim=$latent_dim dataset.config.batch_size=$batchsize model_params.config.encoder.config.embedder.config.input_dim=$input_dim global_params.hidden_size=$encoder_dim global_params.ode_size=$ode_dim model_params.config.decoder.config.hidden=$decoder_dim logging_params.name=${config_name}_l${latent_dim}_b${batchsize}_e${encoder_dim}_o${ode_dim}_d${decoder_dim}


#python train.py --config-name=light_curve_autoencoder_GRU dataset.config.batch_size=$batchsize model_params.config.encoder.config.input_dim=$input_dim model_params.config.encoder.config.hidden_dim=$hidden_dim model_params.config.decoder.config.hidden=$hidden_dim logging_params.name=BandAwareGRUVAE_${batchsize}_${input_dim}_${hidden_dim}
#python train.py --config-name=light_curve_autoencoder_PureDeepSets dataset.config.batch_size=$batchsize model_params.config.encoder.config.input_dim=$input_dim model_params.config.encoder.config.hidden_dim=$hidden_dim model_params.config.decoder.config.hidden=$hidden_dim logging_params.name=BandAwareDeepSetVAE_${batchsize}_${input_dim}_${hidden_dim}

#python train.py model_params.config.is_adversarial=True logging_params.name=BandAwareDeepSetVAE_adv #& 
#python train.py model_params.config.is_adversarial=False logging_params.name=BandAwareDeepSetVAE_No_adv #--config-name=light_curve_autoencoder_GRU
rm core*
