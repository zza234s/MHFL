set -e
cd ../../../ #到federatedscope目录

# Configuration
gpu=$1
dataset=$2 #cifar10, svhn, office_caltech
task=$3    #CV_high, CV_low, NLP_high, NLP_low
result_folder_name=$4
method=FedMD

# Method setup
client_file="model_heterogeneity/model_settings/model_setting_"$task"_heterogeneity.yaml"
result_floder=model_heterogeneity/result/${result_folder_name}
script_floder="model_heterogeneity/methods/"${method}
main_cfg=${script_floder}"/${method}""_on_"${dataset}"_label_change.yaml"
exp_name="RUN_"$method"_on_"$dataset"_"$task

# WandB setup
wandb_use=False
wandb_name_user=niudaidai
wandb_online_track=False
wandb_client_train_info=True
wandb_name_project="RUN_"$method"_on_"$dataset"_"$task

# FedMD-specific parameters
if [ "$task" = "CV_high" ]; then
  case "$dataset" in "cifar10" | "office_caltech" | "svhn")
    lr=0.001
    digest_epochs=1
    ;;
  *)
    echo "Unknown public_dataset value: $public_dataset"
    exit 1
    ;;
  esac
fi

# Hyperparameters
local_update_step=1
optimizer='Adam'
seed=(0 1 2)
total_round=200
patience=50
momentum=0.9
freq=1

# Define function for model training
train_model() {
  python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
    federate.total_round_num ${total_round} \
    early_stop.patience ${patience} \
    result_floder ${result_floder} \
    exp_name ${exp_name} \
    seed ${1} \
    train.local_update_steps ${local_update_step} \
    train.optimizer.type ${optimizer} \
    train.optimizer.lr ${lr} \
    train.optimizer.momentum ${momentum} \
    device ${gpu} \
    ${splitter_args} \
    wandb.use ${wandb_use} \
    wandb.name_user ${wandb_name_user} \
    wandb.name_project ${wandb_name_project} \
    wandb.online_track ${wandb_online_track} \
    wandb.client_train_info ${wandb_client_train_info} \
    eval.freq ${freq} \
    MHFL.task ${task} \
    fedmd.digest_epochs ${digest_epochs}
}

# Training parameters based on the dataset
declare -A lda_alpha_map=(
  ["cifar10"]="100 1.0 0.1"
  ["svhn"]="100 1.0 0.1"
  ["office_caltech"]="100 1.0 0.1"
)
lda_alpha=(${lda_alpha_map[$dataset]})

# Loop over parameters for HPO
for alpha in "${lda_alpha[@]}"; do
  for s in "${seed[@]}"; do
    splitter_args="data.splitter_args ""[{'alpha':${alpha}}]"
    train_model "$s"
  done
done
