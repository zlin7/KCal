from importlib import reload
import torch
import tqdm

import pipeline.trainer as trainer
import data.dataloader as dld
import models.pretrained
import pipeline.main
import pipeline.evaluate
import pipeline.kernel
import pipeline.loss
import utils.utils as utils
import _settings

#use the pretrained ViT

import timm
from timm.scheduler import cosine_lr
from timm.loss import LabelSmoothingCrossEntropy
import models.kernel_classifier; reload(models.kernel_classifier)
import pipeline.kernel as kernels
import ipdb
def train_new(dataset,
              lr=2e-4, n_epochs=40, batch_size=128, continue_from_key=None,
              true_weight_decay=1e-4,
              train_split=dld.TRAIN, val_split=dld.VALID,
              model_class = models.pretrained.ViTB16_timm, model_kwargs={}, gpu_id=1,
              optimizer_class=torch.optim.SGD,
              cooldown_epochs = 10,
              datakwargs= {'sample_for_train': True},

              #eval and cache embeddings
              eval_datakwargs={'resize_for_pretrained_model': True},

              #baselines:
              use_focal=False, use_mmce=False,
              **kwargs):
    kwargs.setdefault('short_desc', '')
    assert not (use_mmce and use_focal)
    if use_focal:
        kwargs.update({'criterion_class': pipeline.loss.FocalLossAdaptive, 'criterion_kwargs': {}})
        kwargs['short_desc'] = 'focal'
    elif use_mmce:
        kwargs.update({'criterion_class': pipeline.loss.MMCE, 'criterion_kwargs': {}})
        kwargs['short_desc'] = 'mmce'
    else:
        kwargs.update({'criterion_class': LabelSmoothingCrossEntropy, 'criterion_kwargs': {'smoothing': 0.1}})
    if dataset != _settings.ImageNet1K_NAME: datakwargs.update(eval_datakwargs)

    #scheduler:
    kwargs.setdefault('scheduler_class', cosine_lr.CosineLRScheduler)
    kwargs.setdefault('scheduler_kwargs', {"t_initial": n_epochs, 'lr_min': 1e-8, 'warmup_lr_init': 0.0001, 'warmup_t': 3,  'k_decay': 1.0,
                                                 'cycle_mul': 1.0, 'cycle_decay': 0.5, 'cycle_limit': 1,
                                                 'noise_range_t': None, 'noise_pct': 0.67, 'noise_std': 1.0, 'noise_seed': 42,
                                                 'step_on': 'epoch_and_acc'}, #num_epochs = n_epochs+10
                      )

    model_kwargs['nclass'] = dld.get_nclasses(dataset)
    key = pipeline.main._train(dataset, datakwargs,
                               train_split=train_split, val_split=val_split,
                               continue_from_key=continue_from_key,
                               n_epochs=n_epochs + cooldown_epochs, batch_size=batch_size, eval_steps=None,
                               model_class = model_class, model_kwargs=model_kwargs,

                               optimizer_class=optimizer_class,
                               optimizer_kwargs={"lr": lr, 'true_weight_decay': true_weight_decay},

                               task_type=trainer.CallBack.TASK_CLASS, gpu_id=gpu_id, **kwargs
                               )
    if eval_datakwargs is not None:
        if isinstance(gpu_id, tuple) or isinstance(gpu_id, list):
            gpu_id = gpu_id[0]
        for split in [dld.TRAIN, dld.VALID, dld.TEST]:
            temp = pipeline.main.get_embeddings_and_predictions(key, split=split, dataset=dataset, datakwargs=eval_datakwargs, gpu_id=gpu_id)
            print(split, (temp[-1]['label'] == temp[-1]['pred']).mean())
    return key

def train_kernel(key, dataset, niters=5000, val_niters=500, n_epochs=50, datakwargs = {},
                 proj_name='BN-Linear', proj_kwargs={'input_features': 512, 'output_features': 32},
                 kern_name='RBF',
                 lr=2e-4, gpu_id=0, train_split=dld.TRAIN, val_split=dld.VALID,
                 optimizer_class=torch.optim.SGD, optimizer_kwargs={},
                 **kwargs):
    import models.kernel_classifier; reload(models.kernel_classifier)
    dataset = f'KD-{dataset}'
    datakwargs = utils.merge_dict_inline(datakwargs, {"key": key, 'niters_per_epoch': niters})
    val_datakwargs = utils.merge_dict_inline(datakwargs, {"niters_per_epoch":val_niters})
    model_kwargs = {'proj_kwargs': proj_kwargs, 'num_classes': dld.get_nclasses(dataset)}

    if proj_name != 'BN-Linear': model_kwargs['proj_name'] = proj_name
    if kern_name != 'RBF': model_kwargs['kern_name'] = kern_name

    key2 = pipeline.main._train(dataset, datakwargs, val_datakwargs = val_datakwargs, train_split=train_split, val_split=val_split,
                                n_epochs=n_epochs, batch_size=1, eval_steps=None,
                                model_class=models.kernel_classifier.ProjectionTrainer, model_kwargs=model_kwargs,
                                criterion_class=pipeline.loss.LogLoss, criterion_kwargs={},
                                optimizer_class=optimizer_class, optimizer_kwargs=utils.merge_dict_inline({'lr': lr}, optimizer_kwargs),
                                gpu_id=gpu_id, **kwargs)
    return key2


def train_kernel_sort(key, dataset, niters=5000, val_niters=500, n_epochs=50, datakwargs = {},
                      proj_name='BN-Linear', proj_kwargs={'input_features': 512, 'output_features': 32},
                      kern_name='RBF',
                      lr=2e-4, gpu_id=0, train_split=dld.TRAIN, val_split=dld.VALID,
                      optimizer_class=torch.optim.SGD, optimizer_kwargs={},
                      **kwargs):
    import models.kernel_classifier; reload(models.kernel_classifier)
    dataset = f'KD-{dataset}'
    datakwargs = utils.merge_dict_inline(datakwargs, {"key": key, 'niters_per_epoch': niters})
    val_datakwargs = utils.merge_dict_inline(datakwargs, {"niters_per_epoch":val_niters})
    model_kwargs = {'proj_kwargs': proj_kwargs, 'num_classes': dld.get_nclasses(dataset), 'mode': 'sort'}

    if proj_name != 'BN-Linear': model_kwargs['proj_name'] = proj_name
    if kern_name != 'RBF': model_kwargs['kern_name'] = kern_name
    key2 = pipeline.main._train(dataset, datakwargs, val_datakwargs = val_datakwargs, train_split=train_split, val_split=val_split,
                                n_epochs=n_epochs, batch_size=1, eval_steps=None,
                                model_class=models.kernel_classifier.ProjectionTrainer, model_kwargs=model_kwargs,
                                criterion_class=pipeline.loss.LogLoss, criterion_kwargs={},
                                optimizer_class=optimizer_class, optimizer_kwargs=utils.merge_dict_inline({'lr': lr}, optimizer_kwargs),
                                gpu_id=gpu_id, **kwargs)
    return key2
