import copy
import numpy as np

import torch
import torch.nn as nn

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders
from tools import construct_optimizer


class FedDF():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "ENS_TACCS": [],
            "BEF_TACCS": [],
        }

    def train(self):
        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam = int(self.args.c_ratio * len(self.clients))

            sam_clients = np.random.choice(
                self.clients, n_sam, replace=False
            )

            local_models = {}

            avg_loss = Averager()
            for client in sam_clients:
                local_model, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                local_models[client] = copy.deepcopy(local_model)
                avg_loss.add(loss)

            train_loss = avg_loss.item()

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
            )

            if r % self.args.test_round == 0:
                # global test loader
                bef_test_acc = self.test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                ref_model, ens_test_acc = self.refine_global(
                    r=r,
                    model=copy.deepcopy(self.model),
                    local_models=copy.deepcopy(local_models),
                    loader=self.glo_test_loader,
                )

                glo_test_acc = self.test(
                    model=ref_model,
                    loader=self.glo_test_loader,
                )

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["ENS_TACCS"].append(ens_test_acc)
                self.logs["BEF_TACCS"].append(bef_test_acc)

                print("[R:{}] [Ls:{}] [BefAc:{}] [EnsAc:{}] [GloAc:{}]".format(
                    r, train_loss, bef_test_acc, ens_test_acc, glo_test_acc
                ))

    def update_local(self, r, model, train_loader, test_loader):
        optimizer = construct_optimizer(
            model, self.args.lr, self.args
        )

        if self.args.local_steps is not None:
            n_total_bs = self.args.local_steps
        elif self.args.local_epochs is not None:
            n_total_bs = max(
                int(self.args.local_epochs * len(train_loader)), 5
            )
        else:
            raise ValueError(
                "local_steps and local_epochs must not be None together"
            )

        model.train()

        loader_iter = iter(train_loader)

        avg_loss = Averager()
        per_accs = [0.0, 0.0]

        for t in range(n_total_bs + 1):
            try:
                batch_x, batch_y = loader_iter.next()
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = loader_iter.next()

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            hs, logits = model(batch_x)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        return model, per_accs, loss

    def update_global(self, r, global_model, local_models):
        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()

            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def refine_global(self, r, model, local_models, loader):
        model.eval()

        # obtain pseudo labels
        inputs, probs, reals = self.assign_pseudo_labels(
            model=model,
            local_models=local_models,
            loader=loader,
        )
        ens_test_acc = np.mean(np.argmax(probs, axis=1) == reals)

        # 0.0 -> 1.0
        inputs = torch.FloatTensor(inputs)
        probs = torch.FloatTensor(probs)

        probs = (0.75 * torch.log(probs + 1e-8)).softmax(dim=-1)

        # refine this global model
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-6
        )

        CosLR = torch.optim.lr_scheduler.CosineAnnealingLR
        lr_scheduler = CosLR(
            optimizer, T_max=self.args.ref_steps, eta_min=1e-8
        )

        inds = np.arange(len(inputs))
        np.random.shuffle(inds)

        batch_size = 64
        epoch = (self.args.ref_steps * batch_size) / len(inputs)
        inds = np.concatenate([inds] * (int(epoch) + 1), axis=0)

        for s in range(self.args.ref_steps):
            i = batch_size * s
            j = batch_size * (s + 1)

            batch_x = inputs[inds[i:j]]
            batch_y = probs[inds[i:j]]
            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            _, logits = model(batch_x)

            loss = -1.0 * (
                batch_y * logits.log_softmax(dim=1)
            ).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            lr_scheduler.step()
        return model, ens_test_acc

    def extract(self, model, loader):
        inputs = []
        reals = []
        logits = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                batch_hs, batch_logits = model(batch_x)

                inputs.append(batch_x.detach().cpu().numpy())
                reals.append(batch_y.detach().cpu().numpy())
                logits.append(batch_logits.detach().cpu().numpy())

        inputs = np.concatenate(inputs, axis=0)
        reals = np.concatenate(reals, axis=0)
        logits = np.concatenate(logits, axis=0)
        return inputs, reals, logits

    def assign_pseudo_labels(self, model, local_models, loader):
        inputs, reals, _, = self.extract(model, loader)

        all_logits = []

        for name, local_model in local_models.items():
            _, _, logits = self.extract(local_model, loader)
            all_logits.append(logits)

        avg_logits = np.stack(all_logits, axis=0).mean(axis=0)
        probs = self.softmax(avg_logits, alpha=1.0, axis=1)
        return inputs, probs, reals

    def test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, logits = model(batch_x)
                acc = count_acc(logits, batch_y)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def label_refinery(self, feats, probs, alpha):
        cps = probs.transpose().sum(axis=1, keepdims=True)
        zero_inds = np.argwhere(cps <= 1e-6).reshape(-1)

        C = probs.shape[1]
        protos = np.dot(probs.transpose(), feats)
        protos = protos / (cps + 1e-8)

        dists = ((feats[:, None, :] - protos[None, :, :]) ** 2).sum(axis=-1)
        dists[:, zero_inds] = 1e8

        clu_labels = np.argmin(dists, axis=1)

        one_hot_mat = np.diag(np.ones(C))
        ref_probs = one_hot_mat[clu_labels]

        assert 0.0 <= alpha <= 1.0
        ref_probs = alpha * ref_probs + (1.0 - alpha) * probs
        return ref_probs

    def entropy(self, probs):
        C = probs.shape[1]
        ents = (-1.0 * probs * torch.log(probs + 1e-8)).sum(dim=1)
        return ents / C

    def softmax(self, data, alpha, axis):
        res = torch.FloatTensor(alpha * data).softmax(dim=axis)
        return res.numpy()

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
