set -e
cd ../../../ #到federatedscope目录
gpu=0
dataset=femnist

if [[ $dataset = 'femnist' ]]; then
  main_cfg=model_heterogeneity/methods/FML/FML_on_femnist.yaml
  client_file=model_heterogeneity/methods/FML/model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
  exp_name=FML_femnist_5_client_low
elif [[ $dataset = 'cifar10' ]]; then
  main_cfg=model_heterogeneity/methods/FML/FML_on_cifra10.yaml
  client_file=model_heterogeneity/methods/FML/model_setting_5_client_on_cifar10_low_heterogeneity.yaml
  exp_name=FML_cifar10_5_client_low
fi
result_floder=model_heterogeneity/result/csv

local_update_step=(1 10 20)

for ((l = 0; l < ${#local_update_step[@]}; l++)); do
  for k in {0..3}; do
    python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
      result_floder ${result_floder} \
      exp_name ${exp_name} \
      seed ${k} \
      train.local_update_steps ${local_update_step[$l]} \
      device ${gpu}
  done
done

