import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

from model import LabelReconstructor, FeatureReconstructor
from utils import get_clusters, get_accuracy


class Trainer:
    def __init__(self, _args, gnn, optimizer='adam'):
        self.max_epochs = _args.epochs
        self.device = _args.device
        self.lr = _args.lr
        self.weight_decay = _args.weight_decay
        self.gnn = gnn
        self.optimizer_name = optimizer

        self.alpha = _args.alpha
        self.num_clusters = _args.num_clusters

        self._cached_cp = None


    def configure_optimizers(self):
        params = self.gnn.parameters()

        if self.optimizer_name == 'sgd':
            return SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adam':
            return Adam(params, lr=self.lr, weight_decay=self.weight_decay)
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


    # Loss for LLP
    def LLP_loss(self, data, outputs, loss):
        if self.num_clusters is None or self.alpha == 0:
            return loss

        if self._cached_cp is None:
            # Obtain clusters
            assert self.num_clusters > 0
            data = get_clusters(data, num_clusters=self.num_clusters)

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
            # self._cached_cp = torch.nn.functional.one_hot(self._cached_cp.argmax(dim=1), num_classes=data.num_classes).float()

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

        loss += self.alpha * llp_loss

        return loss


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

    def fit(self, data, _args, verbose=True):
        data = data.to(self.device)
        self.gnn = self.gnn.to(self.device)

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

        if not np.isinf(_args.y_eps):
            fy = LabelReconstructor(_args)
            data = fy(data)

        if not np.isinf(_args.x_eps):
            fx = FeatureReconstructor(_args)
            data = fx(data)

        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_acc = self.train_one_epoch(data, optimizer, criterion)

            val_loss, val_acc, _ = self.test_model(data, criterion, val_mode=True)

            test_loss, test_acc, _ = self.test_model(data, criterion, val_mode=False, ret_pred=False)

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_at_best_val = test_acc
                y_at_best_val = y

                res = {
                    "best_val_acc": best_val_acc,
                    "test_at_best_val": test_at_best_val,
                    "best_test_perf": best_test_acc,
                }

                # save model here

            if verbose:
                print("Epoch [{}/{}], Train Loss: {:.2f}, Train Acc: {:.2f}, Val Loss: {:.2f}, Val Acc: {:.2f}, Test Acc: {:.2f}".
                      format(epoch, self.max_epochs, train_loss, train_acc, val_loss, val_acc, test_acc))

        # display best metrics
        print("\nBest metrics: Val Acc: {:.2f}, Test Acc: {:.2f}\n".format(best_val_acc, test_at_best_val))

        return res
