cd ../../ #到项目根目录

########################## FedPCL ########################################################
# CIFAR10
--cfg
model_heterogeneity/methods/FedPCL/FedPCL_on_cifar10.yaml
--client_cfg
model_heterogeneity/methods/FedPCL/proto_model_setting_5_client_on_cifar10_low_heterogeneity.yaml
exp_name
Manual_test_FedPCL_cifar10

# FEMNIST
--cfg
model_heterogeneity/methods/FedPCL/FedPCL_on_femnist.yaml
--client_cfg
model_heterogeneity/methods/FedPCL/proto_model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
exp_name
Manual_test_FedPCL_femnist

## office_caltech_for_check
--cfg
model_heterogeneity/methods/FedPCL/for_check/FedPCL_on_office_caltech.yaml
--client_cfg
model_heterogeneity/methods/FedPCL/proto_model_setting_5_client_on_cifar10_low_heterogeneity.yaml
##################################################################################################


#---------------------------------------------------FSFL----------------------------------------------
#--FEMNIST
--cfg
model_heterogeneity/methods/FSFL/FSFL_on_femnist.yaml
--client_cfg
model_heterogeneity/methods/model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
exp_name
Manual_test_FSFL_FEMNIST


#-- EMNIST for check
--cfg
model_heterogeneity/methods/FSFL/for_check/FSFL_on_EMNIST.yaml
--client_cfg
model_heterogeneity/methods/model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
exp_name
Manual_test_FSFL_EMNIST


#------------------------------------------------------------------------------------------------------


########################################################################################################
#FedProto
#--FEMNIST
--cfg
model_heterogeneity/methods/fedproto/fedproto_on_femnist.yaml
--client_cfg
model_heterogeneity/methods/fedproto/proto_model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
exp_name
Manual_FedProto_FEMNIST_



# FML
#--CIFAR10
--cfg
model_heterogeneity/methods/FML/FML_on_cifar10.yaml
--client_cfg
model_heterogeneity/methods/FML/model_setting_5_client_on_cifar10_low_heterogeneity.yaml
exp_name
Manual_HPO_FML_5_clients_on_cifa10_low_heterogeneity



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
--cfg
D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\FedHeNN\fedhenn_on_cifar10.yaml
--client_cfg
D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\FedHeNN\fedhenn_per_client_on_cifar10.yaml



#FCCL
--cfg
/data/hxh2022/FederatedScope/FedMM/federatedscope/model_heterogeneity/methods/FCCL/FCCL_on_officehome.yaml
--client_cfg
/data/hxh2022/FederatedScope/FedMM/federatedscope/model_heterogeneity/methods/FCCL/FCCL_per_client_on_officehome.yaml
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