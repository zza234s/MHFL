set -e
cd ../../../ #到federatedscope目录
gpu=0
dataset=cifar10
exp_name=test_hpo_fccl_temp
result_floder=model_heterogeneity/result/csv

if [[ $dataset = 'femnist' ]]; then
  main_cfg=model_heterogeneity/methods/FML/FML_on_femnist.yaml #这里还没改
  client_file=model_heterogeneity/methods/FML/model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
elif [[ $dataset = 'cifar10' ]]; then
  main_cfg=model_heterogeneity/methods/FCCL/FCCL_on_cifar10.yaml
  client_file=model_heterogeneity/methods/FCCL/FCCL_5_client_on_cifar10_low_heterogeneity.yaml
fi


local_update_step=(1 5 15)
lrs=(0.01 0.001)
optimizer=('Adam' 'SGD')
seed=(0 1)

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
          device ${gpu}
      done
    done
  done
done
