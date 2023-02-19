#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import utils.utils as utils

class Optimizer(object):
    def __init__(self,
                model,
                lr0=1e-2,
                momentum=0.9,
                wd=5e-4,
                warmup_steps=1000,
                warmup_start_lr=1e-5,
                max_iter=80000,
                power=0.9,
                *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        param_list = [
                {'params': wd_params},
                {'params': nowd_params, 'weight_decay': 0},
                {'params': lr_mul_wd_params, 'lr_mul': True},
                {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr_mul': True}]
        self.optim = torch.optim.SGD(
                param_list,
                lr = lr0,
                momentum = momentum,
                weight_decay = wd)
        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)


    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr


    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = self.lr * 10
            else:
                pg['lr'] = self.lr
        if self.optim.defaults.get('lr_mul', False):
            self.optim.defaults['lr'] = self.lr * 10
        else:
            self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2:
            logger = utils.get_logger()
            logger.info('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()

    def state_dict(self):
        return self.optim.state_dict()

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

#https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/optimizers/create_optimizer.py
def create_optimizer(base_class, model, **kwargs):
    if 'true_weight_decay' in kwargs:
        weight_decay = kwargs.pop('true_weight_decay')
        assert 'weight_decay' not in kwargs
        parameters = add_weight_decay(model, weight_decay)
        return base_class(params=parameters, **kwargs)
    return base_class(model.parameters(), **kwargs)
