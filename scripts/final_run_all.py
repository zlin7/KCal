import models.pretrained
import torch
import scripts.timm_based as tb
import _settings
gpu_id = 1

proj_name = 'Skip-ELU'

for model_class in [models.pretrained.MixerB16_timm, models.pretrained.ViTB16_timm]:
    for dataset in [_settings.CIFAR10_NAME, _settings.CIFAR100_NAME, _settings.SVHN_NAME]:
        key1 = tb.train_new(dataset, n_epochs=40, lr=2e-4, gpu_id=gpu_id, batch_size=128,
                         optimizer_class=torch.optim.SGD,
                         model_class=model_class)
        key_ker = tb.train_kernel(key1, dataset,
                               datakwargs={'datakwargs': {'resize_for_pretrained_model': True}, 'batch_size': 20},
                               proj_name=proj_name,
                               lr=4e-4, proj_kwargs={'input_features': 768, 'output_features': 32},
                               scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 10},
                               n_epochs=100, train_split='train', val_split='val', val_niters=1,
                               kern_name='RBFM', short_desc=proj_name)
        key_focal = tb.train_new(dataset, n_epochs=40, lr=2e-4, gpu_id=gpu_id, batch_size=128,
                              optimizer_class=torch.optim.SGD,
                              model_class=model_class, use_focal=True)
        key_mmce = tb.train_new(dataset, n_epochs=40, lr=2e-4, gpu_id=gpu_id, batch_size=128,
                              optimizer_class=torch.optim.SGD,
                              model_class=model_class, use_mmce=True)
        print(model_class.__name__, f"{dataset}: {key1}, {key_ker}, {key_focal}, {key_mmce}")

#IIIC
import scripts.iiic as iiic
key1 = iiic.train_IIIC(datakwargs={'train_on': 'pseudo'})

key_ker = tb.train_kernel(key1, _settings.IIICSup_NAME,
                       datakwargs={'datakwargs': {'majority_only': True}, 'batch_size': 20},
                       proj_name=proj_name, proj_kwargs={'input_features': 64, 'output_features': 32},
                       scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 10},
                       lr=4e-4, n_epochs=100, train_split='train', val_split='val', val_niters=1,
                       kern_name='RBFM', short_desc=proj_name,
                       )
key_focal = iiic.train_IIIC(datakwargs={'train_on': 'pseudo'}, use_focal=True)
key_mmce = iiic.train_IIIC(datakwargs={'train_on': 'pseudo'}, use_mmce=True)
print(f"ResNet-{_settings.IIICSup_NAME}: {key1}, {key_ker}, {key_focal}, {key_mmce}")


#ISRUC
import models.CNNs
key1 = tb.train_new(_settings.ISRUC_NAME,
                    lr=5e-3, batch_size=128, eval_datakwargs={}, datakwargs={},
                    model_class=models.CNNs.CNNEncoder2D_ISRUC)

key_ker = tb.train_kernel(key1, _settings.ISRUC_NAME,
                          datakwargs={'datakwargs': {}, 'batch_size': 20},
                          proj_name=proj_name, proj_kwargs={'input_features': 96, 'output_features': 32},
                          #lr=4e-4, n_epochs=100, train_split='train', val_split='val', val_niters=1,
                          lr=1e-3, n_epochs=100, train_split='train', val_split='val', val_niters=1,
                          scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 10},
                          kern_name='RBFM', short_desc=proj_name)
key_focal = tb.train_new(_settings.ISRUC_NAME,
                    lr=5e-3, batch_size=128, eval_datakwargs={}, datakwargs={},
                    model_class=models.CNNs.CNNEncoder2D_ISRUC, use_focal=True)
key_mmce = tb.train_new(_settings.ISRUC_NAME,
                    lr=5e-3, batch_size=128, eval_datakwargs={}, datakwargs={},
                    model_class=models.CNNs.CNNEncoder2D_ISRUC, use_mmce=True)
print(f"ResNet-{_settings.ISRUC_NAME}: {key1}, {key_ker}, {key_focal}, {key_mmce}")

#ECG
import models.mina as mina
lr = 1e-2
d = 8
key1 = tb.train_new(_settings.ECG_NAME, lr=lr, n_epochs=100,
                    model_class=mina.FreqNet,
                    scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    scheduler_kwargs={'mode': 'min', 'factor': 0.5, 'patience': 10},
                    datakwargs={'over_sample': True}, eval_datakwargs={},
                    short_desc=f'{lr}_plat')
key_ker = tb.train_kernel(key1, _settings.ECG_NAME,
                          datakwargs={'datakwargs': {}, 'batch_size': 20},
                          proj_name=proj_name, proj_kwargs={'input_features': d, 'output_features': d},
                          lr=1e-3, n_epochs=100, train_split='train', val_split='val', val_niters=1,
                          kern_name='RBFM', short_desc=proj_name + str(d),
                          scheduler_kwargs={'mode': 'min', 'factor': 0.5, 'patience': 10},)
key_focal = tb.train_new(_settings.ECG_NAME, lr=lr, n_epochs=100,
             model_class=mina.FreqNet,
             scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
             scheduler_kwargs={'mode': 'min', 'factor': 0.5, 'patience': 10},
             datakwargs={'over_sample': True}, eval_datakwargs={},
             use_focal=True)
key_mmce = tb.train_new(_settings.ECG_NAME, lr=lr, n_epochs=100,
             model_class=mina.FreqNet,
             scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
             scheduler_kwargs={'mode': 'min', 'factor': 0.5, 'patience': 10},
             datakwargs={'over_sample': True}, eval_datakwargs={},
             use_mmce=True)
print(f"FreqNet-{_settings.ECG_NAME}: {key1}, {key_ker}, {key_focal}, {key_mmce}")



#ImageNet
ImageNet_KEY = 'inception_resnet_v2'
key_ker = tb.train_kernel(ImageNet_KEY, _settings.ImageNet1K_NAME, datakwargs={'datakwargs': {}, 'batch_size': 20},
                         proj_name=proj_name, proj_kwargs={'input_features': 1536, 'output_features': 128},
                         lr=5e-2, n_epochs=100, train_split='train', val_split='val', val_niters=1,
                         kern_name='RBFM', short_desc=proj_name,
                         scheduler_kwargs={'mode': 'min', 'factor': 0.5, 'patience': 15}
                         )

key_ker_linear = tb.train_kernel(ImageNet_KEY, _settings.ImageNet1K_NAME, datakwargs={'datakwargs': {}, 'batch_size': 20},
                         proj_name='BN-Linear', proj_kwargs={'input_features': 1536, 'output_features': 128},
                         lr=5e-2, n_epochs=100, train_split='train', val_split='val', val_niters=1,
                         kern_name='RBFM', short_desc=proj_name,
                         scheduler_kwargs={'mode': 'min', 'factor': 0.5, 'patience': 15}
                         )

print(f"ImageNet: {key_ker}")