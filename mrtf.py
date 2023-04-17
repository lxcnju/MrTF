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


class MrTF():
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
            "REF_TACCS": [],
        }

        # global probs
        self.glo_probs = None

    def train(self):
        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam = int(self.args.c_ratio * len(self.clients))

            sam_clients = np.random.choice(
                self.clients, n_sam, replace=False
            )

            # init model
            init_model = copy.deepcopy(self.model)

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

            if self.args.add_glo is True:
                local_models["glo-init"] = copy.deepcopy(init_model)
                local_models["glo-agg"] = copy.deepcopy(self.model)

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc = self.test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                glo_w = 0.75 * self.get_weight_by_loss()
                ref_model, ens_test_acc = self.refine_global(
                    r=r,
                    model=copy.deepcopy(self.model),
                    local_models=copy.deepcopy(local_models),
                    loader=self.glo_test_loader,
                    glo_w=glo_w,
                )

                ref_test_acc = self.test(
                    model=ref_model,
                    loader=self.glo_test_loader,
                )

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["ENS_TACCS"].append(ens_test_acc)
                self.logs["REF_TACCS"].append(ref_test_acc)

                print("[R:{}] [Ls:{}] [GloAc:{}] [EnsAc:{}] [RefAc:{}]".format(
                    r, train_loss, glo_test_acc, ens_test_acc, ref_test_acc
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
                cparam = local_models[client].state_dict()[name]
                try:
                    noise = self.args.dp_sigma * torch.randn_like(cparam)
                    vs.append(cparam + noise)
                except Exception:
                    vs.append(cparam)

            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()

            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def refine_global(self, r, model, local_models, loader, glo_w):
        model.eval()

        # obtain pseudo labels
        inputs, probs, reals = self.assign_pseudo_labels(
            model=model,
            local_models=local_models,
            loader=loader,
            glo_weight=glo_w
        )
        ens_test_acc = np.mean(np.argmax(probs, axis=1) == reals)

        # 0.0 -> 1.0
        if self.args.dataset == "shakespeare":
            inputs = torch.LongTensor(inputs)
        else:
            inputs = torch.FloatTensor(inputs)
        probs = torch.FloatTensor(probs)

        if self.glo_probs is None:
            self.glo_probs = copy.deepcopy(probs)
        else:
            w = 0.75 * self.get_weight_by_loss()
            self.glo_probs = (1.0 - w) * probs + w * self.glo_probs

        # alpha = 0.75 + 0.25 * self.get_weight_by_loss()
        alpha = 1.0
        probs = (alpha * torch.log(self.glo_probs + 1e-8)).softmax(dim=-1)

        # refine this global model
        model.train()
        """
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        """
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

    def get_weight_by_loss(self):
        """ 0.0 --> 1.0
        """
        last_losses = self.logs["LOSSES"][-5:]
        if len(last_losses) <= 0:
            loss = np.log(self.args.n_classes)
        else:
            loss = np.mean(last_losses)

        loss = np.clip(loss, 0.0, np.log(self.args.n_classes))
        loss = loss / np.log(self.args.n_classes)
        w = 1.0 - loss
        return w

    def extract(self, model, loader):
        inputs = []
        reals = []
        logits = []
        probs = []
        ents = []
        hs = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                batch_hs, batch_logits = model(batch_x)

                std_logi = torch.std(batch_logits)
                batch_logits = batch_logits / (std_logi + 1e-8)

                batch_probs = batch_logits.softmax(dim=-1)
                batch_ents = self.entropy(batch_probs)

                inputs.append(batch_x.detach().cpu().numpy())
                reals.append(batch_y.detach().cpu().numpy())
                hs.append(batch_hs.detach().cpu().numpy())
                logits.append(batch_logits.detach().cpu().numpy())
                probs.append(batch_probs.detach().cpu().numpy())
                ents.append(batch_ents.detach().cpu().numpy())

        inputs = np.concatenate(inputs, axis=0)
        reals = np.concatenate(reals, axis=0)
        logits = np.concatenate(logits, axis=0)
        probs = np.concatenate(probs, axis=0)
        ents = np.concatenate(ents, axis=0)
        hs = np.concatenate(hs, axis=0)
        return inputs, reals, logits, probs, ents, hs

    def assign_pseudo_labels(self, model, local_models, loader, glo_weight):
        way = self.args.ref_way
        alpha = self.args.ref_alpha

        inputs, reals, _, _, _, glo_hs = self.extract(model, loader)

        all_logits = []
        all_probs = []
        all_ents = []
        all_hs = []
        all_ws = []

        for name, local_model in local_models.items():
            if self.args.add_glo is False:
                w = 1.0 / len(local_models)
            else:
                if self.args.glo_w is False:
                    w = 1.0 / len(local_models)
                else:
                    K = sum([
                        1 for ky in local_models.keys() if "glo" not in str(ky)
                    ])
                    if "glo" in str(name):
                        w = glo_weight / (len(local_models) - K)
                    else:
                        w = (1.0 - glo_weight) / K

            infos = self.extract(local_model, loader)
            all_logits.append(infos[2])
            all_probs.append(infos[3])
            all_ents.append(infos[4])
            all_hs.append(infos[5])
            all_ws.append(w)

        all_ws = np.array(all_ws)

        if way == "AvgLogi":
            all_logits = np.stack(all_logits, axis=0)
            avg_logits = (all_ws[:, None, None] * all_logits).sum(axis=0)
            probs = self.softmax(avg_logits, alpha=1.0, axis=1)
        elif way == "AvgProb":
            all_probs = np.stack(all_probs, axis=0)
            probs = (all_ws[:, None, None] * all_probs).sum(axis=0)
        elif way == "WeiLogi":
            all_logits = np.stack(all_logits, axis=0)
            weights = self.softmax(
                -1.0 * np.array(all_ents), alpha=1.0, axis=0
            )
            weight_logits = (
                all_ws[:, None, None] * weights[:, :, None] * all_logits
            ).sum(axis=0)
            probs = self.softmax(weight_logits, alpha=1.0, axis=1)
        elif way == "WeiProb":
            all_probs = np.stack(all_probs, axis=0)
            weights = self.softmax(
                -1.0 * np.array(all_ents), alpha=1.0, axis=0
            )
            probs = (
                all_ws[:, None, None] * weights[:, :, None] * all_probs
            ).sum(axis=0)
        elif way == "AvgRef":
            all_ref_probs = []
            for hs, probs in zip(all_hs, all_probs):
                ref_probs = self.label_refinery(hs, probs, alpha=alpha)
                all_ref_probs.append(ref_probs)

            all_ref_probs = np.stack(all_ref_probs, axis=0)
            probs = (all_ws[:, None, None] * all_ref_probs).sum(axis=0)
        elif way == "GloRef":
            all_probs = np.stack(all_probs, axis=0)
            avg_probs = (all_ws[:, None, None] * all_probs).sum(axis=0)
            probs = self.label_refinery(glo_hs, avg_probs, alpha=alpha)
        elif way == "ALL":
            all_probs = np.stack(all_probs, axis=0)
            weights = self.softmax(
                -1.0 * np.array(all_ents), alpha=1.0, axis=0
            )
            avg_probs = (
                all_ws[:, None, None] * weights[:, :, None] * all_probs
            ).sum(axis=0)

            probs = self.label_refinery(glo_hs, avg_probs, alpha=alpha)
        else:
            raise ValueError("No such way: {}".format(way))
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

        protos = np.dot(probs.transpose(), feats)
        protos = protos / (cps + 1e-8)

        feats = feats / (
            np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8
        )
        protos = protos / (
            np.linalg.norm(protos, axis=-1, keepdims=True) + 1e-8
        )

        dists = ((feats[:, None, :] - protos[None, :, :]) ** 2).sum(axis=-1)
        dists[:, zero_inds] = 1e8

        ref_probs = self.softmax(-1.0 * dists, alpha=4.0, axis=-1)

        '''
        clu_labels = np.argmin(dists, axis=1)

        C = probs.shape[1]
        one_hot_mat = np.diag(np.ones(C))
        ref_probs = one_hot_mat[clu_labels]

        assert 0.0 <= alpha <= 1.0
        ref_probs = alpha * ref_probs + (1.0 - alpha) * probs
        '''
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
