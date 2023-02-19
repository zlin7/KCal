import numpy as np
import pandas as pd
import bisect
import tqdm
import utils.utils as utils
import torch
import _settings

TRAINED = 'trained'

class Kernel(torch.nn.Module):
    def __init__(self, h):
        super(Kernel, self).__init__()
        self.h = h

    def forward(self, x, x1):
        raise NotImplementedError()

    def set_bandwidth(self, h):
        self.h = h

    def get_bandwidth(self):
        return self.h


class RBFKernel(Kernel):
    NAME = 'RBF'
    def __init__(self, h=1, device='cuda:0'):
        super(RBFKernel, self).__init__(h)

    def forward(self, x, x1=None):
        if x1 is None: return 1
        diff = x.unsqueeze(-2) - x1
        d = torch.pow(diff, 2).sum(-1)
        return torch.exp(- d/ self.h)

class RBFKernelMean(Kernel):
    NAME = 'RBFM'
    def __init__(self, h=1, device='cuda:0'):
        super(RBFKernelMean, self).__init__(h)

    def forward(self, x, x1=None):
        if x1 is None: return 1
        diff = x.unsqueeze(-2) - x1
        d = torch.pow(diff, 2).mean(-1)
        return torch.exp(- d/ self.h)


def get_trained_kernel(key, dataset, **kwargs):
    import pipeline.trainer as trainer
    name = 'trainer-KD-%s' % (dataset)
    mode = kwargs.pop('mode', 'last')
    model, settings, _ = trainer.CallBack.load_state(name, key, mode=mode, device='cuda:0')
    return model.kern.eval()

def get_kernel_object(name, **kwargs):
    if name == TRAINED: return get_trained_kernel(**kwargs)
    if name == RBFKernel.NAME: return RBFKernel(**kwargs)
    if name == RBFKernelMean.NAME: return RBFKernelMean(**kwargs)
    raise NotImplementedError()

if __name__ == '__main__':
    Xs = torch.randn(3,2)
    xi = torch.randn(2)
    o = RBFKernel()
    ks = o(xi, Xs)
