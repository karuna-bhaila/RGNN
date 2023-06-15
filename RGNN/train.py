import copy
import sys
import numpy as np
import os
import datetime
from termcolor import cprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

from model import GRR_Reconstructor, GRRFS_Reconstructor
from utils import get_clusters, get_accuracy


class Trainer:
    def __init__(self, _args, gnn, optimizer='adam'):
        self.gnn = gnn
        self.optimizer_name = optimizer
        self._args = _args

        self._cached_cp = None

    def configure_optimizers(self):
        params = self.gnn.parameters()

        if self.optimizer_name == 'sgd':
            return SGD(params, lr=self._args.lr, weight_decay=self._args.weight_decay)
        elif self.optimizer_name == 'adam':
            return Adam(params, lr=self._args.lr, weight_decay=self._args.weight_decay)
        else:
            print("optimizer not configured")
            sys.exit(1)

    def train_one_epoch(self, data, optimizer, criterion):
        self.gnn.train()
        optimizer.zero_grad()

        # Forward
        logits = self.gnn(data.x, data.edge_index)
        if data.num_classes > 2:
            outputs = F.softmax(logits, dim=1)
        else:
            outputs = logits

        loss = criterion(input=outputs[data.train_mask], target=data.y[data.train_mask])

        loss = self.LLP_loss(data, outputs, loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), 5)
        optimizer.step()

        if data.num_classes == 2:
            outputs = torch.sigmoid(outputs)

        acc = get_accuracy(pred=outputs[data.train_mask], target=data.y[data.train_mask])

        return loss.item(), acc

    def LLP_loss(self, data, outputs, loss):
        if self._args.num_clusters is None or self._args.num_clusters < 1 or self._args.alpha == 0:
            return loss

        if self._cached_cp is None:
            data = get_clusters(data, num_clusters=self._args.num_clusters)

            for i in range(data.num_clusters):
                cluster_y = data.y[data.cluster_mask[i] & data.train_mask]
                c_y = torch.unsqueeze(cluster_y.sum(0).div(len(cluster_y)), 0)
                try:
                    self._cached_cp = torch.cat((self._cached_cp, c_y), dim=0)
                except:
                    self._cached_cp = c_y

            # Denoise noisy cluster label proportions
            self._cached_cp = torch.matmul(torch.inverse(data.P_Y).add(0.0), self._cached_cp.t()).t()
            self._cached_cp[self._cached_cp < 0] = 1e-5
            self._cached_cp.true_divide_(torch.sum(self._cached_cp, dim=1, keepdim=True))
            self._cached_cp = self._cached_cp + 1e-20

        # Compute cluster-based loss
        p_y = torch.sigmoid(outputs) if data.num_classes == 2 else outputs
        c_p_y_x = None
        for i in range(data.num_clusters):
            cluster_p_y_x = p_y[data.cluster_mask[i] & data.train_mask]
            temp = torch.unsqueeze(cluster_p_y_x.sum(0).div(len(cluster_p_y_x)), 0)

            try:
                c_p_y_x = torch.cat((c_p_y_x, temp), dim=0)
            except:
                c_p_y_x = temp

        c_p_y_x.true_divide_(torch.sum(c_p_y_x, dim=1, keepdim=True))
        c_p_y_x[c_p_y_x <= 0] = 1e-5
        c_p_y_x = torch.log(c_p_y_x)

        llp_loss = F.kl_div(input=c_p_y_x, target=self._cached_cp, log_target=False, reduction='batchmean')

        loss += self._args.alpha * llp_loss

        return loss

    def save_model_and_results(self, model):
        try:
            model_path = os.path.join(os.getcwd(), 'models')
            os.makedirs(model_path, exist_ok=True)

            date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
            filename = "{}_{}_{}.txt".format(self._args.dataset, self._args.seed, date)

            save_dict = {'state_dict': model}

            torch.save(save_dict, f=os.path.join(model_path, filename))

        except Exception as e:
            cprint(f'Unable to save model: {e}', "red")

    def test_model(self, data, criterion, val_mode=False, ret_pred=False):
        self.gnn.eval()

        nodes_to_test = data.val_mask if val_mode else data.test_mask

        with torch.no_grad():
            logits = self.gnn(data.x, data.edge_index)

            if data.num_classes > 2:
                outputs = F.softmax(logits, dim=1)
                loss = criterion(input=outputs[nodes_to_test], target=data.y[nodes_to_test])
            else:
                outputs = logits
                loss = criterion(input=outputs[nodes_to_test], target=data.y[nodes_to_test])
                outputs = torch.sigmoid(outputs)

            acc = get_accuracy(outputs[nodes_to_test], data.y[nodes_to_test])

        if ret_pred:
            return loss.item(), acc, outputs[nodes_to_test]
        else:
            return loss.item(), acc, None

    def fit(self, data, verbose=True, save_model=True):
        data = data.to(self._args.device)
        self.gnn = self.gnn.to(self._args.device)

        optimizer = self.configure_optimizers()

        if data.num_classes > 2:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        best_val_acc = 0.
        best_test_acc = 0.
        test_at_best_val = 0.
        res = {}
        y = None
        best_model = None

        if not np.isinf(self._args.y_eps):
            fy = GRR_Reconstructor(self._args)
            data = fy(data)

        if not np.isinf(self._args.x_eps):
            fx = GRRFS_Reconstructor(self._args)
            data = fx(data)

        for epoch in range(1, self._args.epochs + 1):
            train_loss, train_acc = self.train_one_epoch(data, optimizer, criterion)

            val_loss, val_acc, _ = self.test_model(data, criterion, val_mode=True, ret_pred=False)

            test_loss, test_acc, y = self.test_model(data, criterion, val_mode=False, ret_pred=True)

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_at_best_val = test_acc
                y_at_best_val = y
                best_model = copy.deepcopy(self.gnn.state_dict())

                res = {
                    "best_val_acc": best_val_acc,
                    "test_at_best_val": test_at_best_val,
                    "best_test_perf": best_test_acc,
                }

            if verbose:
                print("Epoch [{}/{}], Train Loss: {:.2f}, Train Acc: {:.2f}, Val Loss: {:.2f}, Val Acc: {:.2f}, Test Acc: {:.2f}".
                      format(epoch, self._args.epochs, train_loss, train_acc, val_loss, val_acc, test_acc))

        # save best model
        if save_model:
            self.save_model_and_results(best_model)

        # display best metrics
        cprint("\nBest metrics: Val Acc: {:.2f}, Test Acc: {:.2f}\n".format(best_val_acc, test_at_best_val), "blue")

        return res
