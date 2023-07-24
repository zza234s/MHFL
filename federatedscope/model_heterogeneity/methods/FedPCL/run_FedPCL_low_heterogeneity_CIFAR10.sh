set -e
cd ../../../ #到federatedscope目录
gpu=0
dataset=cifar10

if [[ $dataset = 'femnist' ]]; then
#  main_cfg=model_heterogeneity/methods/fedproto/fedhenn_on_femnist.yaml
  client_file=model_heterogeneity/methods/fedproto/proto_model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
  exp_name=fedproto_femnist_5_client_low
elif [[ $dataset = 'cifar10' ]]; then
#  main_cfg=model_heterogeneity/methods/fedproto/fedproto_on_cifar10.yaml
  client_file=model_heterogeneity/methods/fedproto/proto_model_setting_5_client_on_cifar10_low_heterogeneity.yaml
  exp_name=fedproto_cifar10_5_client_low
fi
result_floder=model_heterogeneity/result/csv

local_update_step=(1 10 20)
proto_weight=(0.1 1.0 10.0)
for ((p = 0; p < ${#proto_weight[@]}; p++)); do
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
