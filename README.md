# MrTF

The source code of our works on federated learning:
* Submitted to ECML-PKDD 2023 Journal Track (Data Mining and Knowledge Discovery, DMKD Journal): MrTF: Model Refinery for Transductive Federated Learning.


# Content
* Personal Homepage
* Basic Introduction
* Running Tips
* Citation

## Personal Homepage
  * [Homepage](https://www.lamda.nju.edu.cn/lixc/)

## Basic Introduction
  * We consider a real-world scenario that a newly-established pilot project needs to make inferences for newly-collected data, but it does not have any labeled data for training.
  * We resort to federated learning (FL) and abstract this scene as transductive federated learning (TFL).
  * To facilitate TFL, we propose several techniques including stabilized teachers, rectified distillation, and clustered label refinery.
  * The proposed Model refinery framework for Transductive Federated learning (MrTF) shows superiorities towards other FL methods on several benchmarks.
  * Related Federated Learning codes could be found in our FL repository [FedRepo](https://github.com/lxcnju/FedRepo)

## Environment Dependencies
The code files are written in Python, and the utilized deep learning tool is PyTorch.
  * `python`: 3.7.3
  * `numpy`: 1.21.5
  * `torch`: 1.9.0
  * `torchvision`: 0.10.0
  * `pillow`: 8.3.1

## Datasets
We provide several datasets including (if can not download, please copy the links to a new browser window):
  * \[[MNIST](https://www.lamda.nju.edu.cn/lixc/data/MNIST.zip)\]
  * \[[SVHN](https://www.lamda.nju.edu.cn/lixc/data/SVHN.zip)\]
  * \[[CIFAR-10](https://www.lamda.nju.edu.cn/lixc/data/CIFAR10.zip)\]
  * \[[CIFAR-100](https://www.lamda.nju.edu.cn/lixc/data/CIFAR100.zip)\]

## Running Tips
  * `python train_fedavg.py`: the baseline of FedAvg
  * `python train_feddf.py`: the baseline of FedDF
  * `python train_mrtf.py`: our proposed algorithm for transductive federated learning.

FL algorithms and hyper-parameters could be set in these files.


## Citation
  * Xin-Chun Li, Yang Yang, De-Chuan Zhan. MrTF: Model Refinery for Transductive Federated Learning.
  * \[[BibTex](https://dblp.org/pid/246/2947.html)\]
