cd ../../ #到项目根目录

# 注册的数据集测试
# SVHN
--cfg
model_heterogeneity/methods/Local/Local_on_SVHN.yaml
--client_cfg
model_heterogeneity/methods/model_setting_5_client_on_CIFAR10_low_heterogeneity.yaml
exp_name
SVHN_test

# FedDistill
# CIFAR10
--cfg
model_heterogeneity/methods/FedDistill/FedDistill_on_cifar10.yaml
--client_cfg
model_heterogeneity/model_settings/model_setting_CV_low_heterogeneity.yaml

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
model_heterogeneity/methods/FedPCL/FedPCL_on_office_caltech.yaml
--client_cfg
model_heterogeneity/model_settings/model_setting_CV_high_heterogeneity.yaml
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
# office_caltech
--cfg
model_heterogeneity/methods/fedproto/FedProto_on_office_caltech.yaml
--client_cfg
model_heterogeneity/model_settings/model_setting_CV_low_heterogeneity.yaml
result_floder
model_heterogeneity/result/temp

#--FEMNIST
--cfg
model_heterogeneity/methods/fedproto/fedproto_on_femnist.yaml
--client_cfg
model_heterogeneity/methods/fedproto/proto_model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml
exp_name
Manual_FedProto_FEMNIST_

#--CIFAR10
--cfg
model_heterogeneity/methods/fedproto/fedproto_on_cifar10.yaml
--client_cfg
model_heterogeneity/model_settings/model_setting_CV_low_heterogeneity.yaml

# FML
#--CIFAR10
--cfg
model_heterogeneity/methods/FML/FML_on_cifar10.yaml
--client_cfg
model_heterogeneity/methods/FML/model_setting_5_client_on_cifar10_low_heterogeneity.yaml
exp_name
Manual_HPO_FML_5_clients_on_cifa10_low_heterogeneity

#FedHeNN
python federatedscope/main.py --cfg federatedscope/model_heterogeneity/methods/FedHeNN/fedhenn_on_cifar10.yaml \
  --client_cfg federatedscope/model_heterogeneity/methods/FedHeNN/fedhenn_per_client_on_cifar10.yaml

#DENSE
# --CIFAR10
--cfg
model_heterogeneity/methods/DENSE/DENSE_on_cifar10.yaml
--client_cfg
model_heterogeneity/model_settings/model_setting_CV_low_heterogeneity.yaml
exp_name
main_test_DENSE_5_clients_on_cifar10_low_heterogeneity

# FedMD
#--cifar10
--cfg
model_heterogeneity/methods/FedMD/FedMD_on_cifar10.yaml
--client_cfg
model_heterogeneity/methods/FedMD/model_setting_5_client_on_cifar10_low_heterogeneity.yaml
exp_name
main_test_FedMD_5_clients_on_cifar10_low_heterogeneity

#--femnist
--cfg
model_heterogeneity/methods/FedMD/FedMD_on_femnist.yaml
--client_cfg
model_heterogeneity/methods/FedMD/model_setting_5_client_on_FEMNIST_low_heterogeneity.yaml

#--office_caltech
--cfg
model_heterogeneity/methods/FedMD/FedMD_on_office_caltech.yaml
--client_cfg
model_heterogeneity/model_settings/model_setting_CV_low_heterogeneity.yaml
exp_name
manual_fedmd

#FedHeNN
--cfg
D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\FedHeNN\fedhenn_on_cifar10.yaml
--client_cfg
D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\FedHeNN\fedhenn_per_client_on_cifar10.yaml

#FCCL
# office_10
--cfg
model_heterogeneity/methods/FCCL/FCCL_on_office_caltech.yaml
--client_cfg
model_heterogeneity/model_settings/model_setting_CV_low_heterogeneity.yaml
exp_name
debug_fccl

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
model_heterogeneity/methods/Local/Local_on_office_caltech.yaml
--client_cfg
D:\ZHL_WORK\Architecture_heterogeneous_FL\FS3\federatedscope\model_heterogeneity\methods\Local\model_setting_5_client_on_cifar10.yaml
