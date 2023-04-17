import os
import random
from collections import namedtuple
import numpy as np

import torch

from datasets.feddata import FedData
from fedavg import FedAvg
from basic_nets import get_basic_net

from paths import save_dir
from config import default_param_dicts

torch.set_default_tensor_type(torch.FloatTensor)


def construct_model(args):
    try:
        input_size = args.input_size
    except Exception:
        input_size = None

    try:
        input_channel = args.input_channel
    except Exception:
        input_channel = None

    model = get_basic_net(
        net=args.net,
        n_classes=args.n_classes,
        input_size=input_size,
        input_channel=input_channel,
    )
    return model


def main_federated(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # DataSets
    try:
        n_clients = args.n_clients
    except Exception:
        n_clients = None

    try:
        nc_per_client = args.nc_per_client
    except Exception:
        nc_per_client = None

    try:
        dir_alpha = args.dir_alpha
    except Exception:
        dir_alpha = None

    feddata = FedData(
        dataset=args.dataset,
        split=args.split,
        n_clients=n_clients,
        nc_per_client=nc_per_client,
        dir_alpha=dir_alpha,
        n_max_sam=args.n_max_sam,
    )
    csets, gset = feddata.construct()

    try:
        nc = int(args.dset_ratio * len(csets))
        clients = list(csets.keys())
        sam_clients = np.random.choice(
            clients, nc, replace=False
        )
        csets = {
            c: info for c, info in csets.items() if c in sam_clients
        }
    except Exception:
        pass

    feddata.print_info(csets, gset)

    # Model
    model = construct_model(args)
    print(model)
    print([name for name, _ in model.named_parameters()])
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    algo = FedAvg(
        csets=csets,
        gset=gset,
        model=model,
        args=args
    )
    algo.train()

    fpath = os.path.join(
        save_dir, args.fname
    )
    algo.save_logs(fpath)
    print(algo.logs)


def main_digits_label():
    # split by label
    # dataset, n_clients, dir_alpha
    datasets = ["mnist", "svhn"]
    nc_per_clients = [5]

    # paris: n_clients, c_ratio, local_epochs, max_round
    pairs = [
        (1000, 0.01, 50, 500),
    ]

    # for test
    pairs = [
        (1000, 0.01, 1, 5),
    ]

    for d in [0]:
        dataset = datasets[d]
        for nc_per_client in nc_per_clients:
            for pair in pairs:
                for lr in [0.01]:
                    for dp_sigma in [0.0]:
                        para_dict = {}
                        for k, vs in default_param_dicts[dataset].items():
                            para_dict[k] = random.choice(vs)

                        para_dict["dataset"] = dataset
                        para_dict["split"] = "label"
                        para_dict["nc_per_client"] = nc_per_client
                        para_dict["n_clients"] = pair[0]
                        para_dict["c_ratio"] = pair[1]
                        para_dict["local_epochs"] = pair[2]
                        para_dict["max_round"] = pair[3]
                        para_dict["test_round"] = max(1, int(pair[3] / 100))
                        para_dict["lr"] = lr
                        para_dict["dp_sigma"] = dp_sigma
                        para_dict["fname"] = "fedavg-label-{}.log".format(
                            dataset
                        )

                        main_federated(para_dict)


def main_digits_dirichlet():
    # split by label
    # dataset, n_clients, dir_alpha
    datasets = ["mnist", "svhn"]
    dir_alphas = [1.0]

    # paris: n_clients, c_ratio, local_epochs, max_round
    pairs = [
        (1000, 0.01, 50, 500),
    ]

    # for test
    pairs = [
        (1000, 0.01, 1, 5),
    ]

    for d in [0]:
        dataset = datasets[d]
        for dir_alpha in dir_alphas:
            for pair in pairs:
                for lr in [0.01]:
                    for dp_sigma in [0.0]:
                        para_dict = {}
                        for k, vs in default_param_dicts[dataset].items():
                            para_dict[k] = random.choice(vs)

                        para_dict["dataset"] = dataset
                        para_dict["split"] = "dirichlet"
                        para_dict["dir_alpha"] = dir_alpha
                        para_dict["n_clients"] = pair[0]
                        para_dict["c_ratio"] = pair[1]
                        para_dict["local_epochs"] = pair[2]
                        para_dict["max_round"] = pair[3]
                        para_dict["test_round"] = max(1, int(pair[3] / 100))
                        para_dict["lr"] = lr
                        para_dict["dp_sigma"] = dp_sigma
                        para_dict["fname"] = "fedavg-dir-{}.log".format(
                            dataset
                        )

                        main_federated(para_dict)


def main_cifar_label():
    datasets = ["cifar10"]
    nc_per_clients = {
        "cifar10": [5, 3],
        "cifar100": [100, 50, 30],
        "cinic10": [10, 5, 3],
    }

    # paris: n_clients, c_ratio, local_epochs, max_round
    pairs = [
        (100, 0.1, 2, 200),
        (100, 0.1, 10, 100),
        (100, 0.1, 50, 50),
        (1000, 0.01, 5, 1000),
        (1000, 0.01, 50, 500),
        (1000, 0.01, 100, 200),
    ]

    # for test
    pairs = [
        (1000, 0.01, 1, 5),
    ]

    for d in [0]:
        dataset = datasets[d]
        for nc_per_client in nc_per_clients[dataset]:
            for pair in pairs:
                for net in ["VGG8"]:
                    for lr in [0.05]:
                        para_dict = {}
                        for k, vs in default_param_dicts[dataset].items():
                            para_dict[k] = random.choice(vs)

                        para_dict["dataset"] = dataset
                        para_dict["split"] = "label"
                        para_dict["nc_per_client"] = nc_per_client
                        para_dict["n_clients"] = pair[0]
                        para_dict["c_ratio"] = pair[1]
                        para_dict["local_epochs"] = pair[2]
                        para_dict["max_round"] = pair[3]
                        para_dict["test_round"] = max(1, int(pair[3] / 100))
                        para_dict["lr"] = lr
                        para_dict["dp_sigma"] = 0.0
                        para_dict["fname"] = "fedavg-label-{}.log".format(
                            dataset
                        )

                        main_federated(para_dict)


def main_cifar_dirichlet():
    datasets = ["cifar10"]
    dir_alphas = [1.0, 0.1]

    # paris: n_clients, c_ratio, local_epochs, max_round
    pairs = [
        (100, 0.1, 2, 200),
        (100, 0.1, 10, 100),
        (100, 0.1, 50, 50),
        (1000, 0.01, 5, 1000),
        (1000, 0.01, 50, 500),
        (1000, 0.01, 100, 200),
    ]

    # for test
    pairs = [
        (1000, 0.01, 1, 5),
    ]

    for d in [0]:
        dataset = datasets[d]
        for dir_alpha in dir_alphas:
            for pair in pairs:
                for net in ["VGG8"]:
                    for lr in [0.05]:
                        para_dict = {}
                        for k, vs in default_param_dicts[dataset].items():
                            para_dict[k] = random.choice(vs)

                        para_dict["dataset"] = dataset
                        para_dict["split"] = "dirichlet"
                        para_dict["dir_alpha"] = dir_alpha
                        para_dict["n_clients"] = pair[0]
                        para_dict["c_ratio"] = pair[1]
                        para_dict["local_epochs"] = pair[2]
                        para_dict["max_round"] = pair[3]
                        para_dict["test_round"] = max(1, int(pair[3] / 100))
                        para_dict["lr"] = lr
                        para_dict["dp_sigma"] = 0.0
                        para_dict["fname"] = "fedavg-dir-{}.log".format(
                            dataset
                        )

                        main_federated(para_dict)


if __name__ == "__main__":
    main_digits_label()
    main_digits_dirichlet()

    main_cifar_label()
    main_cifar_dirichlet()
