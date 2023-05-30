cd ../../ #到项目根目录

python federatedscope/main.py --cfg federatedscope/model_heterogeneity/methods/FedHeNN/fedhenn_on_cifar10.yaml \
--client_cfg federatedscope/model_heterogeneity/methods/FedHeNN/fedhenn_per_client_on_cifar10.yaml

#--cfg D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\fedproto\fedproto_on_cifar10.yaml \
#--client_cfg D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\fedproto\fedproto_per_client_on_cifa10.yaml


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