set -e
cd ../../../ #到federatedscope目录
gpu=0
dataset=cifar10

if [[ $dataset = 'femnist' ]]; then
  main_cfg=model_heterogeneity/methods/FedPCL/FedPCL_on_femnist.yaml
  client_file=model_heterogeneity/methods/FedPCL/proto_model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
  exp_name=FedPCL_femnist_5_client_low_heterogeneity
elif [[ $dataset = 'cifar10' ]]; then
  main_cfg=model_heterogeneity/methods/FedPCL/FedPCL_on_cifar10.yaml
  client_file=model_heterogeneity/methods/FedPCL/proto_model_setting_5_client_on_cifar10_low_heterogeneity.yaml
  exp_name=FedPCL_cifar10_5_client_low_heterogeneity
fi
result_floder=model_heterogeneity/result/csv

for k in {0..3}; do
  python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
    result_floder ${result_floder} \
    exp_name ${exp_name} \
    seed ${k} \
    device ${gpu}
done