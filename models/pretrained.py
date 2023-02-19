#https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/models/utils/factory.py

import torch
import torch.distributed as dist
import timm


def _get_dist_info():
    initialized = dist.is_available() and dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def _print_at_master(str):
    def is_master():
        rank, _ = _get_dist_info()
        return rank == 0
    if is_master():
        print(str)


def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                _print_at_master(
                    'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            _print_at_master('could not load layer: {}, not in checkpoint'.format(key))
    return model


def create_model(model_name='vit_base_patch16_224', num_classes=100, model_path=None, official_pretrained=False):
    _print_at_master('creating model {}...'.format(model_name))

    #model_params = {'args': args, 'num_classes': args.num_classes}
    #args = model_params['args']
    model_name = model_name.lower()
    
    if model_name == 'mixer_b16_224_in21k':
        model = timm.create_model('mixer_b16_224_in21k', pretrained=official_pretrained, num_classes=num_classes)
    elif model_name == 'vit_base_patch16_224': # notice - qkv_bias==False currently
        model_kwargs = dict(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=None, qkv_bias=False)
        model = timm.models.vision_transformer._create_vision_transformer('vit_base_patch16_224_in21k',
                                                                          pretrained=official_pretrained,
                                                                          num_classes=num_classes, **model_kwargs)
    else:
        model = timm.create_model(model_name, pretrained=official_pretrained, num_classes=num_classes)

    if model_path and model_path!='':  # make sure to load pretrained ImageNet-1K model
        model = load_model_weights(model, model_path)
    print('done\n')

    return model

class _timmModel(torch.nn.Module):
    def __init__(self, model_name, nclass = 100, model_path=None, pretrained=True,
                 last_layer_name='head'):
        super(_timmModel, self).__init__()
        self.model = create_model(model_name, official_pretrained=pretrained, num_classes=nclass, model_path=model_path)
        self.fc = torch.nn.Linear(getattr(self.model, last_layer_name).in_features, nclass)
        print(f"embedding size={getattr(self.model, last_layer_name).in_features}")
        setattr(self.model, last_layer_name,  torch.nn.Identity())

    def get_readout_layer(self):
        return self.fc

    def forward(self, x, embed_only=False):
        x = self.model(x)
        return x if embed_only else self.fc(x)

class ViTB16_timm(_timmModel):
    def __init__(self, nclass = 100, model_path=None):
        super(ViTB16_timm, self).__init__('vit_base_patch16_224_in21k', nclass, model_path)

class MixerB16_timm(_timmModel):
    def __init__(self, nclass = 100, model_path=None):
        super(MixerB16_timm, self).__init__('mixer_b16_224_in21k', nclass, model_path)
