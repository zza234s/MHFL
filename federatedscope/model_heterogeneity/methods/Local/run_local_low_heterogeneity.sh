set -e
cd ../../../ #到federatedscope目录

dataset=femnist

if [[ $dataset = 'femnist' ]]; then
  main_cfg=model_heterogeneity/methods/Local/Local_on_femnist.yaml
  client_file=model_heterogeneity/methods/Local/model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
  exp_name=local_femnist_5_client_low
elif [[ $dataset = 'cifar10' ]]; then
  main_cfg=model_heterogeneity/methods/Local/Local_on_cifar10.yaml
  client_file=model_heterogeneity/methods/Local/model_setting_5_client_on_cifar10_low_heterogeneity.yaml
  exp_name=local_cifar10_5_client_low
fi

result_floder=model_heterogeneity/result/csv
alpha=0.1
for k in {0..2}; do
  python main.py \
        --cfg ${main_cfg} \
        --client_cfg ${client_file} \
        result_floder ${result_floder} \
        exp_name ${exp_name} \
        seed ${k} \
        data.splitter_args "[{'alpha': ${alpha}}]"
done
