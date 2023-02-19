from importlib import reload
import torch.nn as nn
import torch.nn.functional as F
import torch
import tqdm, torch, numpy as np, pandas as pd, os

import data.dataloader as dld
import utils.utils as utils
import _settings




class GoldenSectionBoundedSearch():
    gr = (1 + (5 ** 0.5)) * 0.5
    def __init__(self, func, lb, ub, tol=1e-4):
        self.func = func
        self.lb = lb
        self.ub = ub
        self.mem = {}
        self.hist = []
        self.tol = tol

        self.round_digit = -int(np.floor(np.log10(tol/2)))
        self._search(lb, ub)


    def eval(self, x):
        assert x >= self.lb and x <= self.ub
        x = np.round(x, self.round_digit)
        v = self.mem.get(x, None)
        if v is None:
            v = self.func(x)
            self.mem[x] = v
            self.hist.append(x)
        return v

    def _search(self, a, b):
        c = b - (b - a) / self.gr
        d = a + (b - a) / self.gr
        steps = int(np.ceil(np.log((b-a)/self.tol)/np.log(self.gr)))
        with tqdm.tqdm(total=steps) as pbar:
            while abs(b - a) > self.tol:
                lc = self.eval(c)
                ld = self.eval(d)
                if lc < ld:
                    b = d
                    ss_repr = f'h={c:5f} Loss:{lc:3f}'
                else:
                    a = c
                    ss_repr = f'h={d:5f} Loss:{ld:3f}'
                c = b - (b - a) / self.gr
                d = a + (b - a) / self.gr
                pbar.update(1)
                pbar.set_description(ss_repr)
        return (b + a) / 2.

    @classmethod
    def search(cls, func, lb, ub, tol=1e-4):
        o = GoldenSectionBoundedSearch(func, lb, ub, tol)
        return o._search(lb, ub), o


class KernelClassifier(torch.nn.Module):
    MODE_BASE, MODE_SORT = 'base', 'sort'
    def __init__(self, X, Y, kern, proj, cnt, num_classes=10, device='cuda:0',
                 mode=MODE_BASE, SorP=None, is_eval=False, group=None):
        super(KernelClassifier, self).__init__()

        self.kern = kern
        self.proj = proj
        self.cnt = cnt


        self.projected_Xs = X if self.proj is None else self.proj(X)

        self.mode = mode
        if self.mode == self.MODE_SORT:
            assert SorP is not None
            temp, sort_idx = torch.sort(torch.tensor(SorP), -1, descending=True)
            locs = torch.argsort(sort_idx, 1)
            self.Y = F.one_hot(locs[torch.arange(len(Y)), Y], num_classes=num_classes).float().to(Y.device)
        #self.to(device)
        elif self.mode == self.MODE_BASE:
            self.Y = F.one_hot(Y, num_classes=num_classes).float()
        if isinstance(self.cnt, dict): #stratified
            for k, cnt_k in self.cnt.items():
                self.Y[:, k] *= cnt_k[1] / float(cnt_k[0])
        else:
            assert isinstance(self.cnt, int)

        self._device = device
        self.is_eval = is_eval

        #within group functionality - not quite relevant
        self.group = group

    def to(self, device):
        self._device = device
        self.projected_Xs = self.projected_Xs.to(device)
        self.Y = self.Y.to(device)
        self.kern.to(device)
        self.proj.to(device)


    def forward(self, x, drop_self=True, SorP=None, group=None):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        x = x.to(self._device)
        # x is batched
        projected_x = x if self.proj is None else self.proj(x)
        if group is not None:
            Kis = self.kern(projected_x, self.projected_Xs[self.group == group])  # Kis is len(x) x len(self.Y)
        else:
            Kis = self.kern(projected_x, self.projected_Xs) #Kis is len(x) x len(self.Y)
        eps = 1e-10
        if drop_self:
            _ = torch.zeros((), device=Kis.device, dtype=Kis.dtype)
            Kis = torch.where(Kis < 1-eps, Kis, _)# This happens when we do LOO (i.e. background and prediction sets are the same)
        Kis = torch.where(Kis.sum(1, keepdim=True) > eps, Kis, eps * torch.ones((), device=Kis.device, dtype=Kis.dtype))
        if group is not None:
            pred = torch.matmul(Kis, self.Y[self.group == group])
        else:
            pred = torch.matmul(Kis, self.Y)
        #if torch.isnan(pred).any(): ipdb.set_trace()
        pred /= torch.sum(pred, 1, keepdim=True)
        if torch.isnan(pred).any(): ipdb.set_trace()
        if self.mode == self.MODE_SORT:
            assert SorP is not None
            if len(SorP.shape) == 1: SorP = SorP.unsqueeze(0)
            SorP = SorP.to(self._device)
            sort_idx = torch.argsort(SorP, -1, descending=True)
            locs = torch.argsort(sort_idx, 1)
            for i in range(len(x)):
                pred[i] = pred[i, locs[i]]
        return pred



class ProjectionTrainer(torch.nn.Module):
    def __init__(self, proj_name='BN-Linear', proj_kwargs = {'input_features': 120, 'output_features': 32},
                 kern_name='RBF', kern_kwargs={"h": 1},
                 num_classes=10, device='cpu', mode='base'):
        super(ProjectionTrainer, self).__init__()
        import pipeline.projection
        import pipeline.kernel
        self.proj = pipeline.projection.get_projection(proj_name, **proj_kwargs)
        self.kern = pipeline.kernel.get_kernel_object(kern_name, **kern_kwargs)
        self.num_classes = num_classes

        self._device = device
        self.mode = mode

    def to(self, device):
        self._device = device
        self.kern.to(device)
        self.proj.to(device)
        return self


    def forward(self, data, batch_size=64):

        if self.mode == 'sort':
            X, S, X_bgd, S_bgd, Y_bgd, cnt = data
            assert S.shape[1] == self.num_classes, "I think we should use the logits to train embeddings.."
            o = KernelClassifier(X_bgd, Y_bgd, self.kern, self.proj, cnt, num_classes=self.num_classes,
                                 device=self._device, mode=self.mode, SorP=S_bgd)
            out = [o.forward(X[st:st + batch_size], SorP=S[st:st+batch_size]) for st in range(0, len(X), batch_size)]
            return torch.cat(out, 0)
        else:
            X, X_bgd, Y_bgd, cnt = data
            o = KernelClassifier(X_bgd, Y_bgd, self.kern, self.proj, cnt, num_classes=self.num_classes,
                                 device=self._device, mode=self.mode)
            out = []
            for st in range(0, len(X), batch_size):
                out.append(o.forward(X[st:st+batch_size]))
            return torch.cat(out, 0)
        #return o.forward(X)


