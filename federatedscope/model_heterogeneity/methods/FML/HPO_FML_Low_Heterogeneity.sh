set -e
cd ../../../ #到federatedscope目录
gpu=0
dataset=cifar10
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
# FML额外优化的超参
fml_alpha=(0.3 0.5 1.0)
fml_beta=(0.3 0.5 1.0)
temp=0
if [[ $dataset = 'femnist' ]]; then
  main_cfg=model_heterogeneity/methods/FML/FML_on_femnist.yaml
  client_file=model_heterogeneity/methods/FML/model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
  exp_name=HPO_FML_FEMNIST
  wandb_name_project=HPO_FML_FEMNIST
  total_round=400
  T=4
  for ((op = 0; op < ${#optimizer[@]}; op++)); do
    for ((lr = 0; lr < ${#lrs[@]}; lr++)); do
      for ((l = 0; l < ${#local_update_step[@]}; l++)); do
        for ((fml_a = 0; fml_a < ${#fml_alpha[@]}; fml_a++)); do
          for ((fml_b = 0; fml_b < ${#fml_beta[@]}; fml_b++)); do
            for ((k = 0; k < ${#seed[@]}; k++)); do
              let temp+=1
              echo "$temp"
              if [ $temp -le 175 ]; then
                continue
              fi
              #Adam
              if [[ ${optimizer[$op]} = 'Adam' ]]; then
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
                  wandb.client_train_info ${wandb_client_train_info} \
                  fml.meme_model.T ${T} \
                  model.T ${T} \
                  fml.alpha ${fml_alpha[$fml_a]} \
                  fml.beta ${fml_beta[$fml_b]}
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
                  wandb.client_train_info ${wandb_client_train_info} \
                  fml.meme_model.T ${T} \
                  model.T ${T} \
                  fml.alpha ${fml_alpha[$fml_a]} \
                  fml.beta ${fml_beta[$fml_b]}
              fi
            done
          done
        done
      done
    done
  done
elif [[ $dataset = 'cifar10' ]]; then
  main_cfg=model_heterogeneity/methods/FML/FML_on_cifar10.yaml
  client_file=model_heterogeneity/methods/FML/model_setting_5_client_on_cifar10_low_heterogeneity.yaml
  exp_name=HPO_FML_CIFAR10
  wandb_name_project=HPO_FML_CIFAR10
  lda_alpha=(100 1.0 0.1)
  T=5
  for ((op = 0; op < ${#optimizer[@]}; op++)); do
    for ((lr = 0; lr < ${#lrs[@]}; lr++)); do
      for ((l = 0; l < ${#local_update_step[@]}; l++)); do
        for ((a = 0; a < ${#lda_alpha[@]}; a++)); do
          for ((fml_a = 0; fml_a < ${#fml_alpha[@]}; fml_a++)); do
            for ((fml_b = 0; fml_b < ${#fml_beta[@]}; fml_b++)); do
              for ((k = 0; k < ${#seed[@]}; k++)); do
                let temp+=1
                echo "$temp"
                if [ $temp -le 11 ]; then
                  continue
                fi

                #Adam
                if [[ ${optimizer[$op]} = 'Adam' ]]; then
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
                    wandb.client_train_info ${wandb_client_train_info} \
                    fml.meme_model.T ${T} \
                    model.T ${T} \
                    fml.alpha ${fml_alpha[$fml_a]} \
                    fml.beta ${fml_beta[$fml_b]}
                else
                  # SGD with momentum=0.9
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
                    wandb.client_train_info ${wandb_client_train_info} \
                    fml.meme_model.T ${T} \
                    model.T ${T} \
                    fml.alpha ${fml_alpha[$fml_a]} \
                    fml.beta ${fml_beta[$fml_b]}
                fi
              done
            done
          done
        done
      done
    done
  done

fi
