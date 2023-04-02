import numpy as np
import math

import torch
import torch.nn.functional as F


class LabelPerturbation:
    def __init__(self, y_eps=np.inf):
        self.y_eps = y_eps

    def __call__(self, data, train_and_val=True):
        if not np.isinf(self.y_eps):
            perturb_mask = data.train_mask | data.val_mask if train_and_val else data.train_mask

            q = 1.0 / (math.exp(self.y_eps) + data.num_classes - 1)
            p = q * math.exp(self.y_eps)

            y = data.y[perturb_mask]
            y = F.one_hot(y.argmax(dim=1), num_classes=data.num_classes)
            pr_y = y * p + (1 - y) * q
            out = torch.multinomial(pr_y, num_samples=1)
            y_perturbed = F.one_hot(out.squeeze(), num_classes=data.num_classes)

            data.y[perturb_mask] = y_perturbed.float()

            # set distortion matrix
            data.P_Y = torch.ones(data.num_classes, data.num_classes, device=data.y.device) * q
            data.P_Y.fill_diagonal_(p)

        else:
            data.P_Y = torch.zeros(data.num_classes, data.num_classes, device=data.y.device)
            data.P_Y.fill_diagonal_(1.0)

        return data


class FeaturePerturbation:
    def __init__(self, x_eps=np.inf, m=None):
        self.x_eps = x_eps
        self.m = m

    def __call__(self, data):
        x = data.x
        n, d = x.shape

        if not np.isinf(self.x_eps):
            if self.m is None:
                q = 1 / (math.exp(self.x_eps) + 1)
                p = math.exp(self.x_eps) * q
                pr_x = x * p + (1 - x) * q
                data.x = torch.bernoulli(pr_x)
                P = torch.full((d,), fill_value=p)
                data.P_X = P

            else:
                q1 = 1 / (math.exp(self.x_eps) + 1)
                p1 = math.exp(self.x_eps) * q1

                p1_x = x * p1 + (1 - x) * q1
                p2_x = torch.full(x.shape, fill_value=0.5)

                index = torch.rand_like(x).topk(self.m, dim=1).indices
                pr_x = p2_x.scatter(1, index, p1_x)
                del index, p1_x, p2_x

                data.x = torch.bernoulli(pr_x)
                P = torch.full((d,), fill_value=p1)
                data.P_X = P

        else:
            data.P_X = torch.full((d,), fill_value=1.0)

        return data

