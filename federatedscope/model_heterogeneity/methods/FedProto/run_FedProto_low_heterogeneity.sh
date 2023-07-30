#TODO 该脚本未完成
set -e
cd ../../../ #到federatedscope目录
gpu=$1
dataset=$2
result_floder=model_heterogeneity/result/run

#wandb
wandb_use=False
wandb_name_user=niudaidai
wandb_online_track=False
wandb_client_train_info=True
#set wandb.name_project in the following if statement.

local_update_step=1
lrs=0.01
optimizer=('Adam')
seed=(0 1 2)
total_round=200
patience=50
momentum=0.9
# FedProto额外优化的超参
proto_weight=0.1

temp=0
if [[ $dataset = 'cifar10' ]]; then
  main_cfg=model_heterogeneity/methods/fedproto/fedproto_on_cifar10.yaml
  client_file=model_heterogeneity/methods/proto_model_setting_5_client_on_cifar10_low_heterogeneity.yaml
  exp_name=FedProto_CIFAR10
  wandb_name_project=FedProto_CIFAR10
fi

  for ((l = 0; l < ${#local_update_step[@]}; l++)); do
    for k in {0..3}; do
      python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
        result_floder ${result_floder} \
        exp_name ${exp_name} \
        seed ${k} \
        fedproto.proto_weight ${proto_weight[$p]} \
        train.local_update_steps ${local_update_step[$l]} \
        device ${gpu}
    done
  done
done
