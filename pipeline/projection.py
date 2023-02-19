import numpy as np
import pandas as pd
import bisect
import tqdm
import utils.utils as utils
import _settings
import torch

TRAINED = 'trained'

class BNLinear(torch.nn.Module):
    NAME = 'BN-Linear'
    def __init__(self, input_features, output_features):
        super(BNLinear, self).__init__()
        self.bn = torch.nn.BatchNorm1d(input_features)
        self.fc = torch.nn.Linear(input_features, output_features, bias=False)
        with torch.no_grad():
            self.fc.weight.data /= (output_features / 2.)
    def forward(self, x):
        return self.fc(self.bn(x))

class _Skip(torch.nn.Module):
    def __init__(self, input_features, output_features, act=torch.nn.ELU, act_kwargs={}):
        super(_Skip, self).__init__()
        self.bn = torch.nn.BatchNorm1d(input_features)
        self.mid = torch.nn.Linear(input_features, output_features)
        self.bn2 = torch.nn.BatchNorm1d(output_features)
        self.fc = torch.nn.Linear(output_features, output_features, bias=False)
        self.act = act(**act_kwargs)
    def forward(self, x):
        x = self.mid(self.bn(x))
        ret = self.fc(self.act(x))
        return ret + x

class SkipELU(_Skip):
    NAME = 'Skip-ELU'
    def __init__(self, input_features, output_features):
        super(SkipELU, self).__init__(input_features,  output_features, act=torch.nn.ELU)


def get_trained_projection(key, dataset, **kwargs):
    import pipeline.trainer as trainer
    name = 'trainer-KD-%s' % (dataset)
    mode = kwargs.pop('mode', 'last')
    model, settings, _ = trainer.CallBack.load_state(name, key, mode=mode, device='cuda:0')
    return model.proj.eval()


def get_projection(name, **kwargs):
    if name is None: return torch.nn.Identity()
    if name == TRAINED: return get_trained_projection(**kwargs)
    if name == BNLinear.NAME: return BNLinear(**kwargs)
    if name == SkipELU.NAME: return SkipELU(**kwargs)
    raise NotImplementedError()