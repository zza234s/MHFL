set -e
cd ../../../ #到federatedscope目录
gpu=0
dataset=femnist
exp_name=HPO_LOCAL
result_floder=model_heterogeneity/result/csv

#wandb
wandb_use=True
wandb_name_user=niudaidai
wandb_online_track=False
wandb_client_train_info=True
#set wandb.name_project in the following if statement.

if [[ $dataset = 'femnist' ]]; then
  main_cfg=model_heterogeneity/methods/Local/Local_on_femnist.yaml
  client_file=model_heterogeneity/methods/Local/model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
  exp_name=HPO_LOCAL_FEMNIST
  wandb_name_project=HPO_LOCAL_FEMNIST
elif [[ $dataset = 'cifar10' ]]; then
  main_cfg=model_heterogeneity/methods/Local/Local_on_cifar10.yaml
  client_file=model_heterogeneity/methods/Local/model_setting_5_client_on_cifar10_low_heterogeneity.yaml
  exp_name=HPO_LOCAL_CIFAR10
  wandb_name_project=HPO_LOCAL_CIFAR10
fi

alpha=(100)
local_update_step=(1)
lrs=(0.01 0.001)
optimizer=('Adam' 'SGD')
seed=(0 1 2)

for ((op = 0; op < ${#optimizer[@]}; op++)); do
  for ((lr = 0; lr < ${#lrs[@]}; lr++)); do
    for ((l = 0; l < ${#local_update_step[@]}; l++)); do
        for ((k = 0; k < ${#seed[@]}; k++)); do
          python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
            result_floder ${result_floder} \
            exp_name ${exp_name} \
            seed ${seed[$k]} \
            train.local_update_steps ${local_update_step[$l]} \
            train.optimizer.type ${optimizer[$op]} \
            train.optimizer.lr ${lrs[$lr]} \
            device ${gpu} \
            wandb.use ${wandb_use} \
            wandb.name_user ${wandb_name_user} \
            wandb.name_project ${wandb_name_project} \
            wandb.online_track ${wandb_online_track} \
            wandb.client_train_info ${wandb_client_train_info}
        done
    done
  done
done

#
#for k in {0..2}; do
#  python main.py \
#    --cfg ${main_cfg} \
#    --client_cfg ${client_file} \
#    result_floder ${result_floder} \
#    exp_name ${exp_name} \
#    seed ${k} \
#    data.splitter_args "[{'alpha': ${alpha}}]"
#done
