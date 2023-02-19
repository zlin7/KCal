import ipdb
import torch
import numpy as np
import baselines.imax_calib.calibration as calibration
import baselines.imax_calib.utils as utils


class ImaxCalib(torch.nn.Module):
    def __init__(self,
                 cal_setting='sCW',  # CW, sCW or top1
                 num_bins=15,
                 Q_method="imax",
                 Q_binning_stage="raw",  # bin the raw logodds or the 'scaled' logodds
                 Q_binning_repr_scheme="sample_based",
                 Q_bin_repr_during_optim="pred_prob_based",
                 Q_rnd_seed=928163,
                 Q_init_mode="kmeans"
                 ):

        super(ImaxCalib, self).__init__()

        self.cfg = {'cal_setting': cal_setting,
                    'num_bins': num_bins,
                    'Q_method': Q_method,
                    'Q_binning_stage': Q_binning_stage,
                    'Q_binning_repr_scheme': Q_binning_repr_scheme,
                    'Q_bin_repr_during_optim': Q_bin_repr_during_optim,
                    'Q_rnd_seed': Q_rnd_seed,
                    'Q_init_mode': Q_init_mode,
                    }
        self.calibrator_obj = None

    def fit(self, valid_logits, valid_labels):
        self.nclass = valid_logits.shape[1]
        self.cfg['n_classes'] = self.nclass
        valid_labels = utils.to_onehot(valid_labels, self.nclass)
        valid_probs = utils.to_softmax(valid_logits)
        valid_logodds = utils.quick_logits_to_logodds(valid_logits, probs=valid_probs)
        self.calibrator_obj = calibration.learn_calibrator(self.cfg,
                                     logits=valid_logits,
                                     logodds=valid_logodds,
                                     y=valid_labels,
                                     )
        #ipdb.set_trace()
        return self
    def transform(self, logits, normalize=False):
        probs = utils.to_softmax(logits)
        logodds = utils.quick_logits_to_logodds(logits, probs)
        cal_logits, cal_logodds, cal_probs, assigned = self.calibrator_obj(logits, logodds)
        if normalize:
            el = cal_probs / (1- cal_probs)
            cal_probs = el / el.sum(1)[:, np.newaxis]
        return cal_probs
