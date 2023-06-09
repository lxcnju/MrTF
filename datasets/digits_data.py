import os
import copy
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms

from paths import digits_fdir

from utils import load_pickle


def load_digits_data(dataset, combine=True):
    """ Load Digits Data from pickle data
    params:
    @dataset: "mnist", "svhn", "mnistm", "usps", "syn"
    @combine: True or False
    return:
    @xs: numpy.array, (n, c, w, h)
    @ys: numpy.array, (n, ), 0-9
    """
    train_fpath = os.path.join(
        digits_fdir, "{}-train-32.pkl".format(dataset)
    )

    test_fpath = os.path.join(
        digits_fdir, "{}-test-32.pkl".format(dataset)
    )

    train_obj = load_pickle(train_fpath)
    test_obj = load_pickle(test_fpath)
    train_xs = train_obj["images"]
    train_ys = train_obj["labels"]
    test_xs = test_obj["images"]
    test_ys = test_obj["labels"]

    if combine:
        xs = np.concatenate([train_xs, test_xs], axis=0)
        ys = np.concatenate([train_ys, test_ys], axis=0)
        return xs, ys
    else:
        return train_xs, train_ys, test_xs, test_ys


class DigitsDataset(data.Dataset):
    def __init__(self, xs, ys, is_train=True):
        self.xs = copy.deepcopy(xs)
        self.ys = copy.deepcopy(ys)
        self.is_train = is_train

        if is_train is True:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([32, 32]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([32, 32]),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)
                )
            ])

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        raw_img = self.xs[index]
        label = self.ys[index]

        # transforms.ToPILImage need (H, W, C) np.uint8 input
        img = raw_img.transpose(1, 2, 0).astype(np.uint8)

        # return (C, H, W) tensor
        img = self.transform(img)

        label = torch.LongTensor([label])[0]
        return img, label
