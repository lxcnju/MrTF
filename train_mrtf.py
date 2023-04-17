import os
import random
from collections import namedtuple
import numpy as np

import torch

from datasets.feddata import FedData
from mrtf import MrTF
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

        n_test = int(args.dset_ratio * len(gset.xs))
        inds = np.random.permutation(len(gset.xs))
        gset.xs = gset.xs[inds[0:n_test]]
        gset.ys = gset.ys[inds[0:n_test]]
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

    algo = MrTF(
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
    nc_per_clients = [5, 3]

    # ref_way, add_glo, glo_w
    gmr_pairs = [
        ("ALL", True, True),
    ]

    # paris: n_clients, c_ratio, local_epochs, max_round
    pairs = [
        (100, 0.1, 5, 200),
    ]

    # for test
    pairs = [
        (1000, 0.01, 1, 5),
    ]

    for d in [0, 1]:
        dataset = datasets[d]
        for nc_per_client in nc_per_clients:
            for ref_way, add_glo, glo_w in gmr_pairs:
                for pair in pairs:
                    for lr in [0.01]:
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
                        para_dict["ref_way"] = ref_way
                        para_dict["add_glo"] = add_glo
                        para_dict["glo_w"] = glo_w
                        para_dict["ref_alpha"] = 0.1
                        para_dict["ref_steps"] = 500
                        para_dict["dp_sigma"] = 0.0
                        para_dict["fname"] = "mrtf-label-{}.log".format(
                            dataset
                        )

                        main_federated(para_dict)


def main_digits_dirichlet():
    # split by label
    # dataset, n_clients, dir_alpha
    datasets = ["mnist", "svhn"]
    dir_alphas = [1.0, 0.1]

    # ref_way, add_glo, glo_w
    gmr_pairs = [
        ("ALL", True, True),
    ]

    # paris: n_clients, c_ratio, local_epochs, max_round
    pairs = [
        (100, 0.1, 5, 200),
    ]

    # for test
    pairs = [
        (1000, 0.01, 1, 5),
    ]

    for d in [0]:
        dataset = datasets[d]
        for dir_alpha in dir_alphas:
            for ref_way, add_glo, glo_w in gmr_pairs:
                for pair in pairs:
                    for lr in [0.03]:
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
                        para_dict["ref_way"] = ref_way
                        para_dict["add_glo"] = add_glo
                        para_dict["glo_w"] = glo_w
                        para_dict["ref_alpha"] = 0.1
                        para_dict["ref_steps"] = 500
                        para_dict["dp_sigma"] = 0.0
                        para_dict["fname"] = "mrtf-dir-{}.log".format(
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

    # ref_way, add_glo, glo_w
    gmr_pairs = [
        ("ALL", True, True),
    ]

    # paris: n_clients, c_ratio, local_epochs, max_round
    pairs = [
        (100, 0.1, 3, 1500),
    ]

    # for test
    pairs = [
        (1000, 0.01, 1, 5),
    ]

    for d in [0]:
        dataset = datasets[d]
        for nc_per_client in nc_per_clients[dataset]:
            for ref_way, add_glo, glo_w in gmr_pairs:
                for pair in pairs:
                    for net in ["VGG8"]:
                        for lr in [0.03]:
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
                            para_dict["ref_way"] = ref_way
                            para_dict["add_glo"] = add_glo
                            para_dict["glo_w"] = glo_w
                            para_dict["ref_alpha"] = 0.1
                            para_dict["ref_steps"] = 500
                            para_dict["dp_sigma"] = 0.0
                            para_dict["fname"] = "mrtf-label-{}.log".format(
                                dataset
                            )

                            main_federated(para_dict)


def main_cifar_dirichlet():
    datasets = ["cifar10"]
    dir_alphas = [1.0, 0.1]

    # ref_way, add_glo, glo_w
    gmr_pairs = [
        ("ALL", True, True),
    ]

    # paris: n_clients, c_ratio, local_epochs, max_round
    pairs = [
        (100, 0.1, 3, 1500),
    ]

    # for test
    pairs = [
        (1000, 0.01, 1, 5),
    ]

    for d in [0]:
        dataset = datasets[d]
        for dir_alpha in dir_alphas:
            for ref_way, add_glo, glo_w in gmr_pairs:
                for pair in pairs:
                    for net in ["VGG8"]:
                        for lr in [0.03]:
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
                            para_dict["ref_way"] = ref_way
                            para_dict["add_glo"] = add_glo
                            para_dict["glo_w"] = glo_w
                            para_dict["ref_alpha"] = 0.1
                            para_dict["ref_steps"] = 500
                            para_dict["dp_sigma"] = 0.0
                            para_dict["fname"] = "mrtf-dir-{}.log".format(
                                dataset
                            )

                            main_federated(para_dict)


def main_other():
    datasets = ["femnist", "shakespeare"]
    nets = ["FeMnistNet", "CharLSTM"]
    lrs = [
        [4e-3],
        [1.47],
    ]
    max_rounds = [500, 200]

    for d in [1]:
        dataset = datasets[d]
        net = nets[d]
        for lr in lrs[d]:
            for ref_way in ["ALL"]:
                para_dict = {}
                for k, vs in default_param_dicts[dataset].items():
                    para_dict[k] = random.choice(vs)

                para_dict["dataset"] = dataset
                para_dict["net"] = net
                para_dict["lr"] = lr
                para_dict["dset_ratio"] = 0.1
                para_dict["max_round"] = max_rounds[d]
                para_dict["test_round"] = max_rounds[d]

                para_dict["ref_way"] = ref_way
                para_dict["add_glo"] = True
                para_dict["glo_w"] = True

                para_dict["ref_alpha"] = 0.1
                para_dict["ref_steps"] = 500
                para_dict["dp_sigma"] = 0.0

                para_dict["fname"] = "fedamr-{}.log".format(
                    dataset
                )

                main_federated(para_dict)


if __name__ == "__main__":
    main_digits_label()
    main_digits_dirichlet()

    main_cifar_label()
    main_cifar_dirichlet()
