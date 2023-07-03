set -e
cd ../../../../ #到项目根目录

main_cfg=federatedscope/model_heterogeneity/methods/Local/Local_on_cifar10.yaml
client_file=federatedscope/model_heterogeneity/methods/Local/model_setting_10_client_on_cifar10_low_heterogeneity.yaml
client_num=10
csv_name=./local_5_client
for k in {0..3}
do
  python federatedscope/main.py --cfg ${main_cfg} csv_name${csv_name} --client_cfg ${client_file} seed ${k}
done
