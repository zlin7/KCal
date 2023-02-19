import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import ipdb
from typing import Callable, Optional, Union, Tuple
import numpy as np

LOG_EPSILON = 1e-5

def agg_loss(loss, reduction):
    if torch.isnan(loss).any(): ipdb.set_trace()
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum': return loss.sum()
    return loss
class ECELoss(torch.nn.Module):
    reduction: str
    #Cross entropy, but takes in the probability instead of the logits
    def __init__(self, n_bins=15, is_prob=False,
                 weight: Optional[Tensor] = None,  ignore_index: int = -100, reduction: str = 'mean') -> None:
        super(ECELoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.is_prob = is_prob

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        dim = input.dim()
        assert dim == 2, f"Expected 2 dimentions (got {dim})"

        softmaxes = input if self.is_prob else torch.softmax(input, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(target)

        loss = torch.zeros(1, device=input.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                loss += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()

class LogLoss(torch.nn.Module):
    reduction: str
    #Cross entropy, but takes in the probability instead of the logits
    def __init__(self, weight: Optional[Tensor] = None,  ignore_index: int = -100, reduction: str = 'mean', clip=1e-10) -> None:
        super(LogLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.clip = clip

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        dim = input.dim()
        assert dim == 2, f"Expected 2 dimensions (got {dim})"
        input = input.clip(self.clip)#this weight should be trivial, so I won't normalize
        input = -torch.log(input)
        if self.weight is not None:
            input = input * self.weight.unsqueeze(0)
        loss = torch.gather(input, -1, target.unsqueeze(-1)).squeeze(-1)
        if torch.isnan(loss).any():
            ipdb.set_trace()
        #loss = -torch.log(loss)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()

class MyCrossEntropy(torch.nn.Module):
    reduction: str
    #Cross entropy, but takes in the probability instead of the logits
    def __init__(self, weight: Optional[Tensor] = None,  ignore_index: int = -100, reduction: str = 'mean') -> None:
        super(MyCrossEntropy, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        assert self.weight is None or isinstance(self.weight, Tensor)
        dim = input.dim()
        assert dim == 2, f"Expected 2 dimentions (got {dim})"
        q = torch.log_softmax(input, -1)
        target = target[:, 2:]
        loss =  -(target / target.sum(1, keepdim=True) * q).sum(-1)
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()

#======================================================================================================================
#https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
#https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss_adaptive_gamma.py
from scipy.special import lambertw
def get_gamma(p=0.2):
    '''
    Get the gamma for a given pt where the function g(p, gamma) = 1
    '''
    y = ((1-p)**(1-(1-p)/(p*np.log(p)))/(p*np.log(p)))*np.log(1-p)
    gamma_complex = (1-p)/(p*np.log(p)) + lambertw(-y + 1e-12, k=-1)/np.log(1-p)
    gamma = np.real(gamma_complex) #gamma for which p_t > p results in g(p_t,gamma)<1
    return gamma


class FocalLossAdaptive(nn.Module):
    #ps = [0.2, 0.5]
    #gammas = [5.0, 3.0]
    #i = 0
    #gamma_dic = {}
    #for p in ps:
    #    gamma_dic[p] = gammas[i]
    #    i += 1
    gamma_dic = {0.2: 5.0, 0.5: 3.0}
    def __init__(self, gamma=3.0, reduction: str = 'mean'):
        super(FocalLossAdaptive, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def get_gamma_list(self, pt, device):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if (pt_sample >= 0.5):
                gamma_list.append(self.gamma)
                continue
            # Choosing the gamma for the sample
            for key in sorted(self.gamma_dic.keys()):
                if pt_sample < key:
                    gamma_list.append(self.gamma_dic[key])
                    break
        return torch.tensor(gamma_list).to(device)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        gamma = self.get_gamma_list(pt, target.device)
        loss = -1 * (1-pt)**gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
#======================================================================================================================
#MMCE


class MMCE(torch.nn.modules.loss._Loss):
    def __init__(self, ignore_index: int = -100, reduction: str = 'mean', mmce_coeff=4.0, on_weight=False) -> None:
        super(MMCE, self).__init__(size_average=None, reduce=None, reduction=reduction)
        self.ignore_index = ignore_index
        self.reduction = reduction
        assert reduction == 'mean'
        self.mmce_coeff = mmce_coeff
        self.on_weight = on_weight

    def forward(self, logits: Tensor, target: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError("Find `calibration_mmce_w_loss` from https://github.com/aviralkumar2907/MMCE/blob/master/20ng_mmce.py")
        #NOTE: I reomved my adaptation as I cannot find a license on this repository.
        #from .mmce.compute_mmce_pytorch import calibration_mmce_w_loss

        pred_prob = torch.softmax(logits, -1) + 1e-10 #This follows the original MMCE paper
        ce_error = torch.nn.CrossEntropyLoss(reduction='none')(torch.log(pred_prob), target)
        mmce_error = 1.0 * calibration_mmce_w_loss(torch.log(pred_prob), target)
        return agg_loss(ce_error, self.reduction) + self.mmce_coeff * mmce_error

if __name__ == '__main__':
    torch.manual_seed(15)
    lf1 = torch.nn.CrossEntropyLoss()
    lf2 = LogLoss(weight=torch.ones(5))

    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    p = torch.softmax(input, 1)

    loss1 = lf1(input, target)
    loss2 = lf2(p, target)
    print(f"{loss1} vs {loss2}")