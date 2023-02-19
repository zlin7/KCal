#https://github.com/gpleiss/temperature_scaling

import numpy as np
import pickle
import pandas as pd
import sys
import os
import _settings

import torch
import torch.nn as nn
import pipeline.loss

class TemperatureScaling_Pytorch:
    def __init__(self, scores, labels, lr=0.01, max_iter=50):
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)
        self.scores = scores
        self.labels = labels

        while True:
            self.temp = nn.Parameter(torch.ones(1) * 1.)
            pre_ece = pipeline.loss.ECELoss(is_prob=False)(scores, labels).item()
            optimizer = torch.optim.LBFGS([self.temp], lr=lr, max_iter=max_iter)
            criterion = torch.nn.CrossEntropyLoss()
            def eval_func():
                l = self.scores / self.temp
                loss = criterion(l, self.labels)
                loss.backward()
                return loss
            optimizer.step(eval_func)

            post_ece = pipeline.loss.ECELoss(is_prob=False)(self.transform(self.scores, to_prob=False), labels).item()

            # In some very rare cases it seems to increase the loss. (CIFAR-10, Mixer, random split=5 or 7
            # Thus, add this:
            if post_ece > pre_ece:
                lr *= 0.5
            else:
                #print(f"{pre_ece}->{post_ece}, {self.temp[0].item()}")
                if self.temp[0].item() < 0:
                    #This is numerical instability. Ignore
                    self.temp = nn.Parameter(torch.ones(1) * 1.)
                break


    @classmethod
    def prob2logit(cls, p):
        return np.log(p / (1.-p))

    @classmethod
    def logit2prob(cls, l):
        l = np.exp(l)
        l = l / np.sum(l, 1)[:, np.newaxis]
        return l

    def transform(self, scores, labels=None, to_prob=False):
        l = scores / float(self.temp)
        if labels is not None:
            post_ece = pipeline.loss.ECELoss(is_prob=False)(torch.tensor(l), torch.tensor(labels)).item()
            #print(f"ECE={post_ece}")
        if to_prob: l = self.logit2prob(l)
        return l


if __name__ == '__main__':
    P = torch.rand((100, 3))
    Y = torch.randint(3, (100,))
    o = TemperatureScaling_Pytorch(P, Y)
    cP = o.transform(P)