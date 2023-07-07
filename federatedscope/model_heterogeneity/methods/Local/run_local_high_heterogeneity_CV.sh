set -e
cd ../../../ #到federatedscope目录

dataset=femnist

if [[ $dataset = 'femnist' ]]; then
  main_cfg=model_heterogeneity/methods/Local/Local_on_femnist.yaml
  client_file=model_heterogeneity/methods/Local/model_setting_5_client_on_femnist_high_heterogeneity.yaml
  exp_name=local_femnist_5_client_high_heterogeneity
elif [[ $dataset = 'cifar10' ]]; then
  main_cfg=model_heterogeneity/methods/Local/Local_on_cifar10.yaml
  client_file=model_heterogeneity/methods/Local/model_setting_5_client_on_cifar10_high_heterogeneity.yaml
  exp_name=local_cifar10_5_client_low_high_heterogeneity
fi

result_floder=model_heterogeneity/result/csv


for k in {0..3}
do
  python main.py --cfg ${main_cfg} --client_cfg ${client_file} result_floder ${result_floder} exp_name ${exp_name} seed ${k}
done
