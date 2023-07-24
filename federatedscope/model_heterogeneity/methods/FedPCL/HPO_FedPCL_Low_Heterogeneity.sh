set -e
cd ../../../ #到federatedscope目录
gpu=0
dataset=femnist
result_floder=model_heterogeneity/result/csv

#wandb
wandb_use=False
wandb_name_user=niudaidai
wandb_online_track=False
wandb_client_train_info=True
#set wandb.name_project in the following if statement.

local_update_step=(1 5 10)
lrs=(0.01 0.001)
optimizer=('Adam' 'SGD')
seed=(0 1 2)
total_round=200
patience=50
momentum=0.9

# FedPCL额外优化的超参
# PASS

if [[ $dataset = 'femnist' ]]; then
  main_cfg=model_heterogeneity/methods/FedPCL/FedPCL_on_femnist.yaml
  client_file=model_heterogeneity/methods/FedPCL/proto_model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
  exp_name=HPO_FedPCL_FEMNIST_
  wandb_name_project=HPO_FedPCL_FEMNIST_
  for ((op = 0; op < ${#optimizer[@]}; op++)); do
    for ((lr = 0; lr < ${#lrs[@]}; lr++)); do
      for ((l = 0; l < ${#local_update_step[@]}; l++)); do
          for ((k = 0; k < ${#seed[@]}; k++)); do
            if [[ ${optimizer[$op]} = 'Adam' ]]; then
              #Adam
              python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
                federate.total_round_num ${total_round} \
                early_stop.patience ${patience} \
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
            else
              # SGD optimizer with momentum=0.9
              python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
                federate.total_round_num ${total_round} \
                early_stop.patience ${patience} \
                result_floder ${result_floder} \
                exp_name ${exp_name} \
                seed ${seed[$k]} \
                train.local_update_steps ${local_update_step[$l]} \
                train.optimizer.type ${optimizer[$op]} \
                train.optimizer.lr ${lrs[$lr]} \
                train.optimizer.momentum ${momentum} \
                device ${gpu} \
                wandb.use ${wandb_use} \
                wandb.name_user ${wandb_name_user} \
                wandb.name_project ${wandb_name_project} \
                wandb.online_track ${wandb_online_track} \
                wandb.client_train_info ${wandb_client_train_info}
            fi
          done
      done
    done
  done
elif [[ $dataset = 'cifar10' ]]; then
  main_cfg=model_heterogeneity/methods/FedPCL/FedPCL_on_cifar10.yaml
  client_file=model_heterogeneity/methods/FedPCL/proto_model_setting_5_client_on_cifar10_low_heterogeneity.yaml
  exp_name=HPO_FedPCL_CIFAR10_
  wandb_name_project=HPO_FedPCL_CIFAR10_
  lda_alpha=(100 1.0 0.1)
  for ((op = 0; op < ${#optimizer[@]}; op++)); do
    for ((lr = 0; lr < ${#lrs[@]}; lr++)); do
      for ((l = 0; l < ${#local_update_step[@]}; l++)); do
        for ((a = 0; a < ${#lda_alpha[@]}; a++)); do
            for ((k = 0; k < ${#seed[@]}; k++)); do
              if [[ ${optimizer[$op]} = 'Adam' ]]; then
                #Adam
                python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
                  federate.total_round_num ${total_round} \
                  early_stop.patience ${patience} \
                  result_floder ${result_floder} \
                  exp_name ${exp_name} \
                  seed ${seed[$k]} \
                  train.local_update_steps ${local_update_step[$l]} \
                  train.optimizer.type ${optimizer[$op]} \
                  train.optimizer.lr ${lrs[$lr]} \
                  device ${gpu} \
                  data.splitter_args "[{'alpha': ${lda_alpha[$a]}}]" \
                  wandb.use ${wandb_use} \
                  wandb.name_user ${wandb_name_user} \
                  wandb.name_project ${wandb_name_project} \
                  wandb.online_track ${wandb_online_track} \
                  wandb.client_train_info ${wandb_client_train_info}
              else
                python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
                  federate.total_round_num ${total_round} \
                  early_stop.patience ${patience} \
                  result_floder ${result_floder} \
                  exp_name ${exp_name} \
                  seed ${seed[$k]} \
                  train.local_update_steps ${local_update_step[$l]} \
                  train.optimizer.type ${optimizer[$op]} \
                  train.optimizer.lr ${lrs[$lr]} \
                  train.optimizer.momentum ${momentum} \
                  device ${gpu} \
                  data.splitter_args "[{'alpha': ${lda_alpha[$a]}}]" \
                  wandb.use ${wandb_use} \
                  wandb.name_user ${wandb_name_user} \
                  wandb.name_project ${wandb_name_project} \
                  wandb.online_track ${wandb_online_track} \
                  wandb.client_train_info ${wandb_client_train_info}
              fi
            done
        done
      done
    done
  done
fi
