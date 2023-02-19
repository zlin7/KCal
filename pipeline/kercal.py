from importlib import reload

import numpy as np
import pandas as pd
import tqdm
import utils.utils as utils
import _settings
import torch
import models.kernel_classifier



def eval_cal_preds(x, y, PS_func, quiet=True, ps_kwargs={}, ps_list_kwargs={}):
    iter_ = enumerate(x) if quiet else tqdm.tqdm(enumerate(x), desc='eval_conformal', total=len(x))
    res, preds = [], []
    for i, xi in iter_:
        ps_params = utils.merge_dict_inline({k:v[i] for k,v in ps_list_kwargs.items()}, ps_kwargs)
        pred, extra = PS_func(x=xi, **ps_params)
        preds.append(pred)
        res.append({'extra': extra, 'index': i, 'y': y[i]})
        if xi.shape[0] == 1: res[-1].update({"x": xi[0]})
    preds = pd.DataFrame(preds).rename(columns=lambda c: 'cP%d'%c)
    res = pd.DataFrame(res)
    return pd.concat([preds, res], axis=1)


def _pred_func(X_train, y_train, K_obj, X_test, batch_size=64, num_classes=10):
    model = models.kernel_classifier.KernelClassifier(X_train, y_train, K_obj, None,
                                                      cnt=len(y_train), num_classes=num_classes,
                                                      device=X_test.device)
    preds = []
    with torch.no_grad():
        for st in range(0, len(X_test), batch_size):
            ed = min(len(X_test), st + batch_size)
            preds.append(model(X_test[st:ed]).detach())
    del model
    torch.cuda.empty_cache()
    return torch.cat(preds, 0)


def _index_all(potential_tuple, mask):
    if isinstance(potential_tuple, tuple):
        return tuple([v[mask] for v in potential_tuple])
    return potential_tuple[mask]

def fit_bandwidth_within_group(K_obj, proj, Xs, Ys, pred_func=_pred_func, lb=1e-1, ub=1e1, Kfold=20, num_classes=10,
                  loss='log', split_by_group=None):
    from sklearn.model_selection import KFold, GroupKFold
    import pipeline.loss
    loss_func = {"log": pipeline.loss.LogLoss(reduction='mean'), "ece": pipeline.loss.ECELoss(reduction='mean', is_prob=True)}[loss]
    if proj is not None:
        Xs = (proj(Xs[0]), Xs[1]) if isinstance(Xs, tuple) else proj(Xs)
    base_h = K_obj.get_bandwidth()
    if Kfold == -1: Kfold = len(Xs)
    def eval_loss(h):
        K_obj.set_bandwidth(h)
        kf = KFold(n_splits=Kfold, random_state=_settings.RANDOM_SEED, shuffle=True)
        preds, y_tests = [], []
        for gid in split_by_group.unique():
            mask = split_by_group == gid
            gYs = Ys[mask]
            gXs = _index_all(Xs, mask)
            for train_index, test_index in kf.split(gYs, gYs, None):
                X_train, X_test = _index_all(gXs, train_index), _index_all(gXs, test_index)
                y_train, y_test = gYs[train_index], gYs[test_index]
                preds.append(pred_func(X_train, y_train, K_obj, X_test, num_classes=num_classes))
                y_tests.append(y_test)
        y_tests = torch.cat(y_tests, 0)
        loss_val = loss_func(torch.cat(preds, 0), y_tests).item()
        return loss_val
    h, o = models.kernel_classifier.GoldenSectionBoundedSearch.search(eval_loss, lb * base_h, ub * base_h)
    K_obj.set_bandwidth(base_h)
    return h

def fit_bandwidth(K_obj, proj, Xs, Ys, pred_func=_pred_func, lb=1e-1, ub=1e1, Kfold=20, num_classes=10,
                  loss='log', split_by_group=None):
    from sklearn.model_selection import KFold, GroupKFold
    import pipeline.loss
    loss_func = {"log": pipeline.loss.LogLoss(reduction='mean'), "ece": pipeline.loss.ECELoss(reduction='mean', is_prob=True)}[loss]
    if proj is not None:
        Xs = (proj(Xs[0]), Xs[1]) if isinstance(Xs, tuple) else proj(Xs)
    base_h = K_obj.get_bandwidth()
    if Kfold == -1: Kfold = len(Xs)
    def eval_loss(h):
        K_obj.set_bandwidth(h)
        if split_by_group is not None:
            kf = GroupKFold(n_splits=Kfold)
        else:
            kf = KFold(n_splits=Kfold, random_state=_settings.RANDOM_SEED, shuffle=True)
        #kf = KFold(n_splits=Kfold)
        preds, y_tests = [], []
        for train_index, test_index in kf.split(Ys, Ys, split_by_group):
            if isinstance(Xs, tuple):
                X_train, X_test = [v[train_index] for v in Xs], [v[test_index] for v in Xs]
            else:
                X_train, X_test = Xs[train_index], Xs[test_index]
            y_train, y_test = Ys[train_index], Ys[test_index]
            preds.append(pred_func(X_train, y_train, K_obj, X_test, num_classes=num_classes))
            y_tests.append(y_test)
        y_tests = torch.cat(y_tests, 0)
        loss_val = loss_func(torch.cat(preds, 0), y_tests).item()
        return loss_val
    h, o = models.kernel_classifier.GoldenSectionBoundedSearch.search(eval_loss, lb * base_h, ub * base_h)
    K_obj.set_bandwidth(base_h)
    return h

class KernelCal(torch.nn.Module):
    def __init__(self,  Ys=None, preds=None, Xs=None, K_obj=None, proj=None, device='cuda:0',
                 fit_bw_Fold=None, fit_loss='log', fit_split_groups=None,
                 within_group=False,
                 mode='base'):
        super(KernelCal, self).__init__()
        Xs = torch.tensor(Xs, dtype=torch.float, device=device)
        Ys = torch.tensor(Ys, dtype=torch.long,device=device)
        preds = torch.tensor(preds, dtype=torch.float, device=device)
        K_obj.to(device)
        proj.to(device)
        self.mode = mode
        self.within_group = within_group
        self.fit_split_groups = fit_split_groups

        if fit_bw_Fold:
            if within_group:
                h_star = fit_bandwidth_within_group(K_obj, proj, (Xs, preds), Ys, pred_func=self.pred_batch, Kfold=fit_bw_Fold,
                                       num_classes=preds.shape[1], loss=fit_loss, split_by_group=fit_split_groups)
            else:
                h_star = fit_bandwidth(K_obj, proj, (Xs, preds), Ys, pred_func=self.pred_batch, Kfold=fit_bw_Fold, num_classes=preds.shape[1], loss=fit_loss, split_by_group=fit_split_groups)
            K_obj.set_bandwidth(h_star)
        self.model = models.kernel_classifier.KernelClassifier(Xs, Ys, K_obj, proj,
                                                               cnt=len(Ys), num_classes=preds.shape[1],
                                                               device=device, mode=mode, SorP=preds,
                                                               group=fit_split_groups)
        self.model.eval()
        self._device = device


    def pred_batch(self, X_train, y_train, K_obj, X_test, batch_size=64, num_classes=10):
        X_train, preds_train = X_train
        X_test, preds_test = X_test
        model = models.kernel_classifier.KernelClassifier(X_train, y_train, K_obj, None,
                                                          cnt=len(y_train), num_classes=num_classes,
                                                          device=X_test.device, mode=self.mode, SorP=preds_train)
        kwargs = {}
        preds = []
        with torch.no_grad():
            for st in range(0, len(X_test), batch_size):
                ed = min(len(X_test), st + batch_size)
                if model.mode == model.MODE_SORT:
                    kwargs['SorP'] = preds_test[st:ed]
                preds.append(model(X_test[st:ed], **kwargs).detach())
        del model
        torch.cuda.empty_cache()
        return torch.cat(preds, 0)

    def predict(self, pred=None, x=None, **kwargs):
        with torch.no_grad():
            if self.model.mode == self.model.MODE_SORT:
                kwargs['SorP'] = torch.tensor(pred, dtype=torch.float, device=self._device)
            pred =  self.model(torch.tensor(x, dtype=torch.float, device=self._device),
                               **kwargs).cpu().numpy()
            if len(x.shape) == 1: pred = pred[0]
            return pred, None