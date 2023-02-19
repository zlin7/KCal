from importlib import reload
import torch




import pipeline.trainer as trainer
import pipeline.loss as lf
import data.dataloader as dld
import models.CNNs
import pipeline.main
import utils.utils as utils
import _settings

def train_IIIC(lr=1e-3, n_epochs=50, batch_size=64, weight_decay=1e-5, continue_from_key=None,
               datakwargs={},
                optimizer_class=torch.optim.AdamW,
               model_class = models.CNNs.CNNEncoder2D_IIIC, model_kwargs={'embedding_size': 64},
              use_focal=False, use_mmce=False,
                   **kwargs):
    assert len(kwargs) == 0
    kwargs['short_desc'] = ''
    assert not (use_mmce and use_focal)
    if use_focal:
        kwargs.update({'criterion_class': pipeline.loss.FocalLossAdaptive, 'criterion_kwargs': {}})
        datakwargs['majority_only'] = True
        kwargs['short_desc'] = 'focal'
        kwargs['task_type'] = trainer.CallBack.TASK_CLASS
    elif use_mmce:
        kwargs.update({'criterion_class': pipeline.loss.MMCE, 'criterion_kwargs': {}})
        datakwargs['majority_only'] = True
        kwargs['short_desc'] = 'mmce'
        kwargs['task_type'] = trainer.CallBack.TASK_CLASS
    else:
        kwargs.update({'criterion_class': lf.MyCrossEntropy, 'criterion_kwargs': {}})
        kwargs['task_type'] = trainer.CallBack.TASK_SOFTCLASS
    key = pipeline.main._train(_settings.IIICSup_NAME, datakwargs,
                               continue_from_key=continue_from_key,
                               n_epochs=n_epochs, batch_size=batch_size, eval_steps=None,

                               model_class = model_class, model_kwargs=model_kwargs,

                               optimizer_class=optimizer_class,
                               optimizer_kwargs={"lr": lr, 'weight_decay': weight_decay},

                               scheduler_class=None,
                               gpu_id=(0,1),
                               **kwargs
                               )
    return key