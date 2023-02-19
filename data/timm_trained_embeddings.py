#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

(adapted) from the timm library
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress
import persist_to_disk as ptd

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy
import ipdb, tqdm
import numpy as np

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
#_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')


def validate(args, _logger=None):
    _logger = _logger or logging.getLogger('validate')
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = create_dataset(
        root=args.data, name=args.dataset, split=args.split,
        download=args.dataset_download, load_bytes=args.tf_preprocessing, class_map=args.class_map)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    outputs = {'embed': [], 'output': [], 'target': []}
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        model(input)
        end = time.time()
        for batch_idx, (input, target) in tqdm.tqdm(enumerate(loader), total=len(loader), desc='eval'):
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                #output = model(input)
                if hasattr(model, 'head'):
                    embedding = model.forward_features(input)
                    output = model.head(embedding)
                elif hasattr(model, 'classif'):
                    embedding = model.forward_features(input)
                    embedding = model.global_pool(embedding)
                    output = model.classif(embedding)
                elif hasattr(model, 'fc'):
                    embedding = model.forward_features(input)
                    embedding = model.global_pool(embedding)
                    output = model.fc(embedding)
            outputs['embed'].append(embedding.cpu().numpy())
            outputs['output'].append(output.cpu().numpy())
            outputs['target'].append(target.cpu().numpy())
            if valid_labels is not None:
                raise Exception("Don't do this")
                output = output[:, valid_labels]
            loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'])

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    if hasattr(model, 'head'):
        head = model.head.to('cpu')
    elif hasattr(model, 'classif'):
        head = model.classif.to('cpu')
    elif hasattr(model, 'fc'):
        head = model.fc.to('cpu')
    outputs = {k: np.concatenate(v) for k, v in outputs.items()}
    return results, outputs, head, dataset.parser.class_to_idx


#============================================================================
import utils.utils as utils
import functools
import _settings
@ptd.persistf(skip_kwargs=['batch_size', 'gpu_id'], cache=ptd.READONLY)
def _eval_and_save_results(data_path, model_name='vit_base_patch16_224', batch_size=64):
    args = [data_path, '--model', model_name, '--pretrained', '-b', str(batch_size), '-j', '4']
    args = parser.parse_args(args)
    perf, res, head, class_to_idx = validate(args, _logger=utils.get_logger(f'eval_{model_name}_{os.path.basename(data_path)}'))
    res.pop('output')
    return perf, res, head, class_to_idx


@functools.lru_cache(maxsize=2)
def eval_and_save_results(model_name='vit_base_patch16_224', batch_size=64, split='val', gpu_id=0):
    #84.532 / 97.296
    print(model_name, batch_size, split, gpu_id)
    import data.timm_trained_embeddings as tte
    data_path = os.path.join(_settings.ImageNet1K_PATH, split)
    device = utils.gpuid_to_device(gpu_id)
    perf, res, head, class_to_idx = tte._eval_and_save_results(data_path, model_name, batch_size=batch_size)
    embed = res['embed']
    res['output'] = []
    with torch.no_grad():
        #print(torch.cuda.memory_summary('cuda:0', True))
        #ipdb.set_trace()
        head.eval()
        head.to(device)
        #print(torch.cuda.memory_summary('cuda:0', True))
        #ipdb.set_trace()
        for st in tqdm.tqdm(range(0, len(res['embed']), batch_size), desc='gen output on the fly'):
            ed = min(st+batch_size, len(embed))
            pred = head(torch.tensor(embed[st:ed], device=device, dtype=torch.float)).detach().cpu().numpy()
            res['output'].append(pred)
        head.to('cpu')
        #print(torch.cuda.memory_summary('cuda:0', True))
        #ipdb.set_trace()
    res['output'] = np.concatenate(res['output'])
    return perf, res, head, class_to_idx

if __name__ == '__main__':
    pass
    #res = eval_and_save_results(split='val')