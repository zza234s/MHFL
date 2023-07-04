set -e
cd ../../../ #到federatedscope目录

main_cfg=model_heterogeneity/methods/Local/Local_on_cifar10.yaml
client_file=model_heterogeneity/methods/Local/model_setting_10_client_on_cifar10_low_heterogeneity.yaml
client_num=10

result_floder=model_heterogeneity/result/csv
exp_name=local_cifar_5_client_low


for k in {0..3}
do
  python main.py --cfg ${main_cfg} --client_cfg ${client_file} result_floder ${result_floder} exp_name ${exp_name} seed ${k}
done
