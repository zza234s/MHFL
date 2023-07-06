set -e
cd ../../../ #到federatedscope目录

main_cfg=model_heterogeneity/methods/FedHeNN/fedhenn_on_cifar10.yaml
dataset=cifar10

if [[ $dataset = 'femnist' ]]; then
  client_file=model_heterogeneity/methods/FedHeNN/model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
  exp_name=fedhenn_femnist_5_client_low
elif [[ $dataset = 'cifar10' ]]; then
  client_file=model_heterogeneity/methods/FedHeNN/proto_model_setting_5_client_on_cifar10_low_heterogeneity.yaml
  exp_name=fedhenn_cifar10_5_client_low
fi

result_floder=model_heterogeneity/result/csv
eta=0.001
local_update_step=(10 20)

for ((l = 0; l < ${#local_update_step[@]}; l++)); do
  for k in {0..3}; do
    python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
      result_floder ${result_floder} \
      exp_name ${exp_name} \
      seed ${k} \
      fedhenn.eta ${eta} \
      train.local_update_steps ${local_update_step[$l]}
  done
done
