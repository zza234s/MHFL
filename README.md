# Toward Better Model Heterogeneous Federated Learning: A Benchmark and Evaluation Work

This is an evaluation work for Model Heterogeneous Federated Learning (MHFL). It also aims to provide an easy-to-use MHFL benchmark.

这是一个模型异构联邦学习的评估性工作, 同时旨在提供一个易于使用的模型异构联邦学习基准。



The code of this project is under construction.

项目的代码正在完善中~~

## Acknowledgment

All code implementations are based on the FederatedScope V0.3.0: https://github.com/alibaba/FederatedScope 

We are grateful for their outstanding work.


## Currently Supported Algorithms

#### Basic baseline

- **LOCAL**: each client only performs local training without the FL process.

#### Methods based on knowledge distillation (with public dataset)

| Abbreviation | Title                                                        | Venue                 | Materials                                                    |
| ------------ | ------------------------------------------------------------ | --------------------- | ------------------------------------------------------------ |
| FedMD        | FedMD: Heterogenous Federated Learning via Model Distillation | NeurIPS 2019 Workshop | [[pub](https://arxiv.org/abs/1910.03581)] [[repository](https://github.com/diogenes0319/FedMD_clean)] |
| FSFL         | Few-Shot Model Agnostic Federated Learning                   | ACM MM 2022           | [[pub](https://dl.acm.org/doi/abs/10.1145/3503161.3548764)] [[repository](https://github.com/FangXiuwen/FSMAFL)] |
| FCCL         | Learn from Others and Be Yourself in Heterogeneous Federated Learning | CVPR 2022             | [[pub](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf)] [[repository](https://github.com/WenkeHuang/FCCL)] |

#### **Methods without public dataset**

| Abbreviation | Title                                                        | Venue        | Materials                                                    |
| ------------ | ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ |
| FML          | Federated Mutual Learning                                    | ArXiv 2020   | [[pub](https://arxiv.org/abs/2006.16765)] [[repository](https://github.com/ZJU-DAI/Federated-Mutual-Learning)] |
| FedHeNN      | Architecture Agnostic Federated Learning for Neural Networks | ICML 2022    | [[pub](https://proceedings.neurips.cc/paper/2020/hash/18df51b97ccd68128e994804f3eccc87-Abstract.html)] |
| FedProto     | FedProto: Federated Prototype Learning across Heterogeneous Clients | AAAI 2022    | [[pub](https://arxiv.org/abs/2105.00243)] [[repository](https://github.com/zza234s/FedProto)] |
| FedPCL       | Federated Learning from Pre-Trained Models: A Contrastive Learning Approach | NeurIPS 2022 | [[pub](https://openreview.net/forum?id=mhQLcMjWw75)] [[repository](https://github.com/yuetan031/FedPCL)] |
| FedGH        | FedGH: Heterogeneous Federated Learning with Generalized Global Header | ACM MM 2023  | [[pub](https://arxiv.org/abs/2303.13137)] [[repository](https://github.com/LipingYi/FedGH)] |



#### Methods being implemented (Coming soon)

| Abbreviation | Title                                                        | Venue                                               | progress bar    |
| ------------ | ------------------------------------------------------------ | --------------------------------------------------- | --------------- |
| DENSE        | [DENSE: Data-Free One-Shot Federated Learning](https://arxiv.org/abs/2112.12371) | NeurIPS 2022                                        | [########--]95% |
| FedKD        | [Communication-efficient federated learning via knowledge distillation](https://www.nature.com/articles/s41467-022-29763-x) | NC 2022                                             | [----------]0%  |
| FedDistill   | [Federated Knowledge Distillation](https://www.cambridge.org/core/books/abs/machine-learning-and-wireless-communications/federated-knowledge-distillation/F679266F85493319EB83635D2B17C2BD) | Machine Learning and Wireless Communications (2022) | [----------]0%  |
| TBD          | ...                                                          | ...                                                 | ...             |







## Models & Dataset

#### Model setting

We hope to test whether the above methods work well under different model heterogeneous setups. To this end, we conduct experiments with the following two settings.

1. **Low model heterogeneity**: We used five CNN models that are a bit different in terms of the number of channels and layers.
2. **High model heterogeneity:**  We tried five models with large differences, including MLP, CNN, ResNet, and wide ResNet.



For details of the model architecture, please refer to: MHFL/federatedscope/model_heterogeneity/model_settings



#### Dataset

Currently, we conduct experiments on three benchmark datasets: CIFAR-10, SVHN, and office-10.



## Quickly Start

### Step 1. Install FederatedScope

Users need to clone the source code and install FederatedScope (we suggest python version >= 3.9).

- clone the source code

```python
git clone https://github.com/zza234s/MHFL
cd MHFL
```

- install the required packages:

```python
conda create -n fs python=3.9
conda activate fs

# Install pytorch
conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 torchtext=0.11.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

- Next, after the required packages is installed, you can install FederatedScope from `source`:

```python
pip install -e .[dev]
```



### Step 2. Run Algorithm (Take running FedProto as an example)

- Enter the "federatedscope" folder

```python
cd federatedscope
```

- Run the script

```python
python main.py --cfg model_heterogeneity/methods/FedProto/FedProto_on_cifar10.yaml --client_cfg model_heterogeneity/model_settings/model_setting_CV_low_heterogeneity.yaml

```




## PS

Please feel free to contact me.

My email is: hanlinzhou@zjut.edu.cn

My WeChat is: poipoipoi8886
