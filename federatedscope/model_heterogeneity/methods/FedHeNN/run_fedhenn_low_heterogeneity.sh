set -e
cd ../../../ #到federatedscope目录
gpu=0
dataset=cifar10

if [[ $dataset = 'femnist' ]]; then
  main_cfg=model_heterogeneity/methods/FedHeNN/fedhenn_on_femnist.yaml
  client_file=model_heterogeneity/methods/FedHeNN/proto_model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
  exp_name=fedhenn_femnist_5_client_low
elif [[ $dataset = 'cifar10' ]]; then
  main_cfg=model_heterogeneity/methods/FedHeNN/fedhenn_on_cifar10.yaml
  client_file=model_heterogeneity/methods/FedHeNN/proto_model_setting_5_client_on_cifar10_low_heterogeneity.yaml
  exp_name=fedhenn_cifar10_5_client_low
fi

result_floder=model_heterogeneity/result/csv
eta=0.001
local_update_step=(10)
seed=(2)
for ((l = 0; l < ${#local_update_step[@]}; l++)); do
  for ((k = 0; k < ${#seed[@]}; k++)); do
    python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
      result_floder ${result_floder} \
      exp_name ${exp_name} \
      seed ${seed[$k]} \
      fedhenn.eta ${eta} \
      train.local_update_steps ${local_update_step[$l]} \
      device ${gpu}
  done
done
