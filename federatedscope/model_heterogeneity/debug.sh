cd ../../ #到项目根目录

python federatedscope/main.py --cfg federatedscope/model_heterogeneity/methods/FedHeNN/fedhenn_on_cifar10.yaml \
--client_cfg federatedscope/model_heterogeneity/methods/FedHeNN/fedhenn_per_client_on_cifar10.yaml

--cfg D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\fedproto\fedproto_on_cifar10.yaml \
--client_cfg D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\fedproto\fedproto_per_client_on_cifa10.yaml

#DENSE
--cfg
model_heterogeneity/methods/DENSE/for_check/DENSE_on_cifar10_for_check.yaml
--client_cfg
model_heterogeneity/methods/DENSE/for_check/model_setting_5_client_on_cifar10_low_heterogeneity.yaml
exp_name
main_test_DENSE_5_clients_on_cifar10_low_heterogeneity

# FedMD
#--cifar10
--cfg model_heterogeneity/methods/FedMD/FedMD_on_cifar10.yaml
--client_cfg model_heterogeneity/methods/FedMD/model_setting_5_client_on_cifar10_low_heterogeneity.yaml
exp_name
main_test_FedMD_5_clients_on_cifar10_low_heterogeneity

#--femnist
--cfg model_heterogeneity/methods/FedMD/FedMD_on_femnist.yaml
--client_cfg model_heterogeneity/methods/FedMD/model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml

#FedHeNN
#--cfg
#D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\FedHeNN\fedhenn_on_cifar10.yaml
#--client_cfg
#D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\FedHeNN\fedhenn_per_client_on_cifar10.yaml


#FSFL

#--cfg
#D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\FSFL\FSFL_on_femnist.yaml
#--client_cfg
#D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\FSFL\FSFL_per_client_on_femnist.yaml

#FCCL
#--cfg
#/data/hxh2022/FederatedScope/FedMM/federatedscope/model_heterogeneity/methods/FCCL/FCCL_on_officehome.yaml
#--client_cfg
#/data/hxh2022/FederatedScope/FedMM/federatedscope/model_heterogeneity/methods/FCCL/FCCL_per_client_on_officehome.yaml
=======
--cfg
D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\FSFL\FSFL_on_femnist.yaml
--client_cfg
D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\FSFL\FSFL_per_client_on_femnist.yaml

#LOCAL
--cfg
D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\Local\Local_on_cifar10.yaml
--client_cfg
D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\Local\model_setting_5_client_on_cifar10.yaml