import numpy as np
import pandas as pd
#import tqdm
import torch
#import matplotlib.pyplot as plt

def to_onehot(labels, K):
    new_labels = np.zeros((len(labels), K))
    new_labels[np.arange(len(labels)), labels] = 1
    return new_labels

class CalibrationEval:
    def __init__(self):
        pass

    @classmethod
    def get_bins(cls, bins):
        if isinstance(bins, int):
            bins = list(np.arange(bins+1) / bins)
        return bins

    @classmethod
    def assign_bin_old(cls, sorted_ser, bins):
        import bisect
        ret = pd.DataFrame(sorted_ser)
        bin_assign = pd.Series(0, index=sorted_ser.index)
        locs = [bisect.bisect(sorted_ser, b) for b in bins]
        locs[0], locs[-1] = 0, len(ret)
        for i, loc in enumerate(locs[:-1]):
            bin_assign.iloc[loc:locs[i+1]] = i
        ret['bin'] = bin_assign
        return ret

    @classmethod
    def assign_bin(cls, sorted_ser, bins, adaptive=False):
        ret = pd.DataFrame(sorted_ser)
        if adaptive:
            assert isinstance(bins, int)
            step = len(sorted_ser) // bins
            nvals = [step for _ in range(bins)]
            for _ in range(len(sorted_ser) % bins): nvals[-_-1] += 1
            ret['bin'] = [ith for ith, val in enumerate(nvals) for _ in range(val)]
            nvals = list(np.asarray(nvals).cumsum())
            bins = [ret.iloc[0]['conf']]
            for iloc in nvals:
                bins.append(ret.iloc[iloc-1]['conf'])
                if iloc != nvals[-1]:
                    bins[-1] = 0.5 * bins[-1] + 0.5 *ret.iloc[iloc]['conf']
        else:
            bins = cls.get_bins(bins)
            import bisect

            bin_assign = pd.Series(0, index=sorted_ser.index)
            locs = [bisect.bisect(sorted_ser, b) for b in bins]
            locs[0], locs[-1] = 0, len(ret)
            for i, loc in enumerate(locs[:-1]):
                bin_assign.iloc[loc:locs[i+1]] = i
            ret['bin'] = bin_assign
        return ret['bin'], bins

    @classmethod
    def _ECE_loss(cls, summ):
        w = summ['cnt'] / summ['cnt'].sum()
        loss = np.average((summ['conf'] - summ['acc']).abs(), weights=w)
        return loss

    @classmethod
    def ECE_confidence(cls, preds, label, bins=15, adaptive=False, return_bins=False):
        df = pd.DataFrame({"conf": preds.max(1), 'truth': label, 'pred': np.argmax(preds, 1)}).sort_values(['conf']).reset_index()
        df['acc'] = (df['truth'] == df['pred']).astype(int)
        df['bin'], bin_boundary = cls.assign_bin(df['conf'], bins, adaptive=adaptive)
        summ = pd.DataFrame(df.groupby('bin')[['acc', 'conf']].mean())#.fillna(0.)
        summ['cnt'] = df.groupby('bin').size()
        summ = summ.reset_index()
        if return_bins: return summ, cls._ECE_loss(summ), np.mean(np.square(df['conf'].values - df['acc'].values)), bin_boundary
        return summ, cls._ECE_loss(summ), np.mean(np.square(df['conf'].values - df['acc'].values))


    @classmethod
    def ECE_class(cls, preds, label, bins=15, threshold=0., adaptive=False, return_bins=False):
        K = preds.shape[1]
        summs = []
        class_losses = {}
        bin_boundaries = {}
        for k in range(K):
            msk = preds[:, k] >= threshold
            if msk.sum() == 0: continue
            df = pd.DataFrame({"conf": preds[msk, k], 'truth': label[msk]}).sort_values(['conf']).reset_index()
            df['acc'] = (df['truth'] == k).astype(int)
            df['bin'], bin_boundaries[k] = cls.assign_bin(df['conf'], bins, adaptive=adaptive)
            summ = pd.DataFrame(df.groupby('bin')[['acc', 'conf']].mean())
            summ['cnt'] = df.groupby('bin').size()
            summ['k'] = k
            summs.append(summ.reset_index())
            class_losses[k] = cls._ECE_loss(summs[-1])
        class_losses = pd.Series(class_losses)
        class_losses['avg'], class_losses['sum'] = class_losses.mean(), class_losses.sum()
        summs = pd.concat(summs, ignore_index=True)
        if return_bins: return summs, class_losses, bin_boundaries
        return summs, class_losses

    @classmethod
    def _plot_bands(cls, df, ax, title="", legend_name = 'Observed Acc', **kwargs):
        df = df.copy().sort_values('bin', ascending=True)
        df = df[df['cnt'] > 10]

        tdf = df.reindex(columns=['acc', 'conf'])
        ub, lb = tdf.max(1), tdf.min(1)
        acc_plt = ax.plot(df['conf'], df['acc'], label=legend_name, color='red', zorder=0)[0]
        conf_plt = ax.plot(df['conf'], df['conf'], linestyle='dashed', label='Prediction', color='red', zorder=0)[0]
        gap_plt = ax.fill_between(df['conf'], lb, ub, color='b', alpha=.1, label='gap')
        ax.legend(handles=[gap_plt, conf_plt, acc_plt])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        return ax

    @classmethod
    def _plot_bars(cls, df, ax, nbins, title="", legend_name = 'Observed Acc', _min=10, legend=True, **kwargs):
        df = df.copy().sort_values('bin', ascending=True)
        df['gap'] = df['acc'] - df['conf']
        nbins = np.asarray(nbins)
        df['left'] = nbins[df['bin'].values]
        df['width'] = (nbins[1:] - nbins[:-1])[df['bin'].values]
        df = df[df['cnt'] > _min]
        gap_plt = ax.bar(df['left']+0.25*df['width'], df['gap'].abs(), bottom=df.reindex(columns=['acc', 'conf']).min(1), width=df['width']*0.5,
                         label='Gap', color='indigo', zorder=10, align='edge')
        acc_plt = ax.bar(df['left'], df['acc'], bottom=0, width=df['width'],
                         label=legend_name, color='olivedrab',
                         edgecolor='black',
                         zorder=0, alpha=0.3, align='edge')
        if legend: ax.legend(handles=[gap_plt, acc_plt], loc='upper left')

        ax.plot([0, 1], [0, 1], c='red', linestyle='dashed')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_title(title)
        return ax

def routine(tP_curr, tY, nbins=20):
    import pipeline.evaluate as peval
    thres = min(0.01, 1./tP_curr.shape[1])
    overall_summ, overall_loss, brier_top1 = peval.CalibrationEval.ECE_confidence(tP_curr, tY, bins=nbins)
    _, overall_loss_adapt, _ = peval.CalibrationEval.ECE_confidence(tP_curr, tY, bins=nbins, adaptive=True)
    _, class_loss_threshold = peval.CalibrationEval.ECE_class(tP_curr, tY, bins=nbins, threshold=thres)
    _, class_loss_threshold_adapt = peval.CalibrationEval.ECE_class(tP_curr, tY, bins=nbins, threshold=thres, adaptive=True)
    acc = (np.argmax(tP_curr, 1)==tY).mean()

    _eps = 1e-3 / tP_curr.shape[1]
    nll1 = torch.nn.NLLLoss()(torch.log(torch.tensor(tP_curr).clip(_eps, 1-_eps)), torch.tensor(tY)).item()

    _one_hot_Y = np.zeros(tP_curr.shape)
    _one_hot_Y[np.arange(len(tY)), tY] = 1
    _sqs = np.square(tP_curr - _one_hot_Y)
    #from pipeline.skce import skce_eval
    import pipeline._kde_ece as _kde_ece
    return pd.Series({"ece": overall_loss * 100, 'ece_adapt': overall_loss_adapt * 100,
                      "acc": acc*100,
                      'cecet': class_loss_threshold['avg']*100, 'cecet_adapt': class_loss_threshold_adapt['avg'] * 100,
                      "brier_top1": brier_top1, 'brier': np.mean(_sqs), 
                      #'SKCE': skce_eval(tP_curr, tY), "KCE": _kde_ece.cleaner_ece_kde(tP_curr, tY) * 100,
                      'NLLTorch': nll1,
                      })

if __name__ == '__main__':
    pass