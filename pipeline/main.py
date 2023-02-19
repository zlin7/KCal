from importlib import reload
import pipeline.trainer as trainer
import data.dataloader as dld
import _settings
import utils.utils as utils
from torch.utils.data import DataLoader
import torch
import pandas as pd, numpy as np
import os

import tqdm
import persist_to_disk as ptd

import pipeline.kernel
import pipeline.projection
import pipeline.kercal

def _normalize_resplit(resplit={dld.VALID: 50, dld.TEST: 50, 'seed': _settings.RANDOM_SEED}, val_split=dld.VALID, test_split=dld.TEST):
    _sum = resplit[val_split] + resplit[test_split]
    resplit[val_split], resplit[test_split] = resplit[val_split] / _sum, resplit[test_split] / _sum
    resplit.setdefault('seed', _settings.RANDOM_SEED)
    return resplit

@ptd.persistf(expand_dict_kwargs=['datakwargs'], skip_kwargs=['device', 'batch_size'], switch_kwarg='cache')
def _get_embeddings_and_predictions(key, split=dld.VALID, dataset=dld._settings.CIFAR10_NAME, datakwargs={},
                                    device=None, num_workers=0, batch_size=128, **kwargs):
    test_data = dld.get_default_dataset(dataset, split, **datakwargs)
    name = 'trainer-%s' % (test_data.DATASET)
    mode = kwargs.pop('mode', 'last')
    model, settings, _ = trainer.CallBack.load_state(name, key, mode=mode, device=device)
    task_type = settings['task_type']

    device = device or torch.device('cuda:{}'.format(0))
    model = model.to(device)
    collate_fn = test_data._collate_func if hasattr(test_data, '_collate_func') else None
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    _, labels, _, embeds, indices = trainer.CallBack._eval_model(model, test_loader, device, forward_kwargs={'embed_only': True})
    #_, _, _, outputs2, indices2 = trainer.CallBack._eval_model(model, test_loader, device, task_type=task_type)
    linear_m = model.get_readout_layer().to(device)
    with torch.no_grad():
        outputs = []
        for st in tqdm.tqdm(range(0, len(embeds), batch_size), desc='embed2pred'):
            ed = min(st+batch_size, len(embeds))
            outputs.extend(linear_m(torch.tensor(embeds[st:ed], device=device)).detach().tolist())
    preds = np.argmax(np.asarray(outputs), 1)
    embeds = pd.DataFrame(embeds, index=indices)
    predsdf = pd.DataFrame(outputs, index=indices, columns=['S%d'%i for i in range(len(outputs[0]))])
    predsdf['pred'] = preds
    predsdf['label'] = labels
    embeds.index.name = predsdf.index.name = 'index'
    #preds = pd.DataFrame({"index": indices2, 'yhat': preds}).set_index('index')
    linear_m = model.get_readout_layer().to('cpu')
    return embeds, linear_m, predsdf

@ptd.persistf(expand_dict_kwargs='all', skip_kwargs=['gpu_id', 'debug'], groupby=['dataset'], switch_kwarg='cache')
def _train(dataset, datakwargs={},  **kwargs):
    debug = kwargs.get('debug', False)
    train_split = kwargs.pop('train_split', dld.TRAIN)
    val_split = kwargs.pop('val_split', dld.VALID)
    train_data = dld.get_default_dataset(dataset, split=train_split, **datakwargs)
    val_datakwargs = kwargs.pop('val_datakwargs', datakwargs.copy())
    if not kwargs.pop('skip_val', False):
        val_data = dld.get_default_dataset(dataset, split=val_split, **val_datakwargs)
    else:
        val_data = None
    short_desc = kwargs.get('short_desc', '')
    print(short_desc)
    cb = trainer.CallBack(train_data, val_data, **kwargs)
    key = cb.train(num_workers=8 if _settings._ON_SERVER and not debug else 0)
    print(key)
    return key


def get_new_idx_resampled_class_frequency(labels, ratios, seed):
    np.random.seed(seed)
    idx = []
    for k, tser in labels.groupby(labels):
        tidx = np.random.choice(tser.index, int(ratios[k] * len(tser)), replace=True)
        idx.extend(list(tidx))
    return idx
#===================
def get_embeddings_and_predictions(key, split=dld.VALID, dataset=dld._settings.CIFAR10_NAME, datakwargs={},
                                   gpu_id=-1, mode='last',
                                   drift=None,
                                   **kwargs):
    import scipy.special
    if mode != 'last': kwargs['mode'] = mode
    device = utils.gpuid_to_device(gpu_id)
    if dataset == _settings.ImageNet1K_NAME:
        import data.timm_trained_embeddings as tte
        perf, res, linear_m, class_to_idx = tte.eval_and_save_results(key, split={dld.TRAIN: dld.TRAIN}.get(split, dld.VALID),
                                                                      batch_size=64, gpu_id=gpu_id)
        embeds = pd.DataFrame(res['embed'])
        predsdf = pd.DataFrame(res['output'], index=embeds.index, columns=['S%d'%i for i in range(len(res['output'][0]))])
        predsdf['pred'] = res['output'].argmax(1)
        predsdf['label'] = res['target']
        if split != dld.TRAIN:
            indices = sorted(dld.ImageNet1KData.split_data_stratify(predsdf['label'], seed=_settings.RANDOM_SEED, split_ratio=[50, 50])[dld.TRAIN if split == dld.TEST else dld.VALID])
            embeds, predsdf = embeds.loc[indices], predsdf.loc[indices]
    else:
        embeds, linear_m, predsdf = _get_embeddings_and_predictions(key, split, dataset, datakwargs, device, 0, **kwargs)
    linear_m.to(device)
    assert drift is None
    sdf = predsdf.reindex(columns=['S%d'%i for i in range(dld.get_nclasses(dataset))])
    pdf = scipy.special.softmax(sdf, 1).rename(columns=lambda c: c.replace("S", "P"))
    labels = pd.concat([pdf, predsdf],axis=1)
    return embeds, linear_m, labels



def _get_val_and_test_data(key,
                           dataset=dld._settings.CIFAR10_NAME,
                           val_split=dld.VALID, test_split=dld.TEST, datakwargs={},
                           resplit=None,
                           gpu_id=-1,
                           drift=None,
                           use_preds=False,
                           **kwargs):
    datakwargs = datakwargs.copy()
    if dataset == _settings.ISRUC_NAME:
        iid = datakwargs.pop('iid', False)
        by_patient = datakwargs.pop('by_patient', False)
    if dataset == _settings.ECG_NAME: iid = datakwargs.pop('iid', False)

    nclass = dld.get_nclasses(dataset)
    v_embeds, readout, v_preds = get_embeddings_and_predictions(key, split=val_split, dataset=dataset, datakwargs=datakwargs, gpu_id=gpu_id, drift=drift, **kwargs)
    t_embeds, _, t_preds = get_embeddings_and_predictions(key, split=test_split, dataset=dataset, datakwargs=datakwargs, gpu_id=gpu_id, drift=drift, **kwargs)

    if use_preds:
        v_embeds = v_preds.reindex(columns=['S%d' % _i for _i in range(dld.get_nclasses(dataset))])
        t_embeds = t_preds.reindex(columns=['S%d' % _i for _i in range(dld.get_nclasses(dataset))])
    if resplit is not None:
        if isinstance(resplit, int):
            resplit = {"seed": resplit}
        if val_split not in resplit or test_split not in resplit:
            resplit[val_split], resplit[test_split] = len(v_embeds), len(t_embeds)
        if not (dataset == _settings.ISRUC_NAME and by_patient):
            resplit = _normalize_resplit(resplit, val_split, test_split)
        #print(resplit)
        if dataset == _settings.ISRUC_NAME:
            all_embeds = pd.concat([v_embeds, t_embeds], ignore_index=False)
            all_preds = pd.concat([v_preds, t_preds], ignore_index=False)
            import data.iiic_data; reload(data.iiic_data)
            all_preds.index = all_preds.index.map(lambda x: x.replace("-", '_'))
            all_embeds.index = all_embeds.index.map(lambda x: x.replace("-", '_'))
            val_idx = []
            if by_patient:
                pats = pd.Series(all_preds.index.map(lambda x: x.split("_")[0]), all_preds.index)
                #an experiment where we only care about the size of the validation set (for each patient)
                np.random.seed(resplit['seed'])
                for _, idxx in pats.groupby(pats):
                    val_idx.append(np.random.choice(idxx.index, resplit[val_split], replace=False))
                val_idx = pd.Index(np.concatenate(val_idx))
                test_idx = all_embeds.index.difference(val_idx)
            else:
                val_idx, test_idx = data.iiic_data.IIIC_Sup.split_val_test(all_preds.index, resplit['seed'], [resplit[val_split], resplit[test_split]],
                                                                           iid=iid)
        elif dataset == _settings.ECG_NAME:
            all_embeds = pd.concat([v_embeds, t_embeds], ignore_index=False)
            all_preds = pd.concat([v_preds, t_preds], ignore_index=False)
            import data.iiic_data; reload(data.iiic_data)
            val_idx, test_idx = data.iiic_data.IIIC_Sup.split_val_test(all_preds.index, resplit['seed'], [resplit[val_split], resplit[test_split]],
                                                                       iid=iid)
        elif dataset == _settings.IIICSup_NAME:
            all_embeds = pd.concat([v_embeds, t_embeds], ignore_index=False)
            all_preds = pd.concat([v_preds, t_preds], ignore_index=False)
            import data.iiic_data; reload(data.iiic_data)
            val_idx, test_idx = data.iiic_data.IIIC_Sup.split_val_test(all_preds.index, resplit['seed'], [resplit[val_split], resplit[test_split]],
                                                                       iid=datakwargs.get('iid', False))
        else:
            all_embeds = pd.concat([v_embeds, t_embeds], ignore_index=True)
            all_preds = pd.concat([v_preds, t_preds], ignore_index=True)
            np.random.seed(resplit['seed'])
            val_idx = np.random.choice(all_embeds.index, int(resplit[val_split] * len(all_embeds)), replace=False)

            test_idx = all_embeds.index.difference(val_idx)
            #print(len(v_embeds), len(t_embeds))
        v_embeds, v_preds = all_embeds.loc[val_idx], all_preds.loc[val_idx]
        t_embeds, t_preds = all_embeds.loc[test_idx], all_preds.loc[test_idx]
    else:
        val_idx, test_idx = v_embeds.index, t_embeds.index
        #ipdb.set_trace()
    v_P, t_P = [_.reindex(columns=['P%d' % i for i in range(nclass)]) for _ in [v_preds, t_preds]]
    v_S, t_S = [_.reindex(columns=['S%d' % i for i in range(nclass)]) for _ in [v_preds, t_preds]]
    return (v_embeds.values, v_P.values, v_S.values, v_preds['label'].values, val_idx), (t_embeds.values, t_P.values, t_S.values, t_preds['label'].values, test_idx)
    #return (v_embeds.values, v_P.values, v_preds['label'].values), (t_embeds.values, t_P.values, t_preds['label'].values)


def _get_data_and_kernel_objects(key,
                          dataset=dld._settings.CIFAR10_NAME,
                          val_split=dld.VALID, test_split=dld.TEST, datakwargs={},
                          kernel_name='RBF', kernel_kwargs={'h': 5000},
                          proj_name=None, proj_kwargs={},
                          cal_kwargs={},
                          resplit=None,
                          use_preds=False,
                          **kwargs):

    (vX, vP, vS, vY, vI), (tX, tP, tS, tY, tI) = _get_val_and_test_data(key, dataset, val_split, test_split, datakwargs, resplit=resplit, drift=None, use_preds=use_preds)

    K_obj = pipeline.kernel.get_kernel_object(kernel_name, **kernel_kwargs)
    proj = pipeline.projection.get_projection(proj_name, **proj_kwargs)
    if dataset == _settings.IIICSup_NAME and not datakwargs.get('iid', False):
        cal_kwargs = utils.merge_dict_inline(cal_kwargs, {"fit_split_groups": vI.map(lambda x: x.split("_")[0])})
    elif dataset == _settings.ISRUC_NAME:
        if  not datakwargs.get('iid', False) or cal_kwargs.get('within_group', False):
            cal_kwargs = utils.merge_dict_inline(cal_kwargs, {"fit_split_groups": vI.map(lambda x: x.split("_")[0])})
    if 'gpu_id' in kwargs:
        cal_kwargs.setdefault('device', utils.gpuid_to_device(kwargs['gpu_id']))
    kercal_obj = pipeline.kercal.KernelCal(Ys=vY, preds=vP, Xs=vX, K_obj=K_obj, proj=proj, **cal_kwargs)
    return (vX, vP, vS, vY, vI), (tX, tP, tS, tY, tI), kercal_obj

@ptd.persistf(expand_dict_kwargs='all', skip_kwargs=['gpu_id'], groupby=['dataset', 'key'], switch_kwarg='cache')
def _get_calibrated_preds(key,
                          dataset=dld._settings.CIFAR10_NAME,
                          val_split=dld.VALID, test_split=dld.TEST, datakwargs={},
                          kernel_name='RBF', kernel_kwargs={'h': 5000},
                          proj_name=None, proj_kwargs={},
                          cal_kwargs={},
                          resplit=None,
                          drift=None, drift_mode=0,
                          use_preds=False,
                          **kwargs):
    assert drift is None and drift_mode == 0 #Unused parameter
    drop_max = val_split == test_split

    (vX, vP, vS, vY, vI), (tX, tP, tS, tY, tI), kercal_obj = _get_data_and_kernel_objects(key, dataset, val_split, test_split, datakwargs,
                                                                                          kernel_name, kernel_kwargs, proj_name, proj_kwargs,
                                                                                          cal_kwargs, resplit, use_preds, **kwargs)
    if cal_kwargs.get('within_group', False): #TODO: remove this as it's irrelevant to paper
        res_df = pipeline.kercal.eval_cal_preds(tX, tY, kercal_obj.predict, quiet=False,
                                                ps_kwargs={"drop_max": drop_max},
                                                ps_list_kwargs={"pred": tP, 'group': tI.map(lambda x: x.split("_")[0])})
        return res_df
    res_df = pipeline.kercal.eval_cal_preds(tX, tY, kercal_obj.predict, quiet=False,
                                            ps_kwargs={"drop_max": drop_max},
                                            ps_list_kwargs={"pred": tP})
    return res_df

def resplit_routine(val_tuple, test_tuple, vI, tI):
    ret = [[], []]
    for i, val_arr in enumerate(val_tuple[:-1]):
        test_arr = test_tuple[i]
        df = pd.concat([pd.DataFrame(val_arr, index=val_tuple[-1]), pd.DataFrame(test_arr, index=test_tuple[-1])])
        if df.shape[1] == 1: df = df.iloc[:, 0]
        ret[0].append(df.reindex(vI).values)
        ret[1].append(df.reindex(tI).values)
    ret[0].append(vI)
    ret[1].append(tI)
    return tuple(ret[0]), tuple(ret[1])


@ptd.persistf(expand_dict_kwargs='all', skip_kwargs=['gpu_id', 'splits'], groupby=['dataset', 'baseline', 'key'], switch_kwarg='cache')
def _get_calibrated_preds_baselines(key, baseline,
                          dataset=dld._settings.CIFAR10_NAME,
                          val_split=dld.VALID, test_split=dld.TEST, datakwargs={},
                          resplit=None,
                          drift=None, drift_mode=0,
                          cache=0,
                          **kwargs):
    (vX_, vP, vS, vY, vI_), (tX, tP, tS, tY, tI_) = _get_val_and_test_data(key, dataset, val_split, test_split, datakwargs, resplit=resplit, drift=drift)
    if 'splits' in kwargs:
        vI, tI = kwargs.pop('splits')
        (vX_, vP, vS, vY, vI_), (tX, tP, tS, tY, tI_) = resplit_routine((vX_, vP, vS, vY, vI_), (tX, tP, tS, tY, tI_), vI, tI)
    assert drift is None

    if baseline == 'imax':
        import baselines.imax_calib.main as imax_calib; reload(imax_calib)
        imax_calib_obj = imax_calib.ImaxCalib().fit(vS, vY)
        tP_ = imax_calib_obj.transform(tS)
    elif baseline == 'msodir':
        cache_path = os.path.join(_settings.WORKSPACE, 'dircal_extern', dataset, f"{key}_{resplit['seed']}_res.pkl")
        tP_ = pd.read_pickle(cache_path)
    elif baseline == 'dircal':
        # I ran DirCal using their scripts and save down the results.
        if len(vY) == 1000 and key == 'inception_resnet_v2':
            key = 'inception_resnet_v21K'
        if dataset == _settings.ISRUC_NAME or dataset == _settings.IIICSup_NAME or dataset == _settings.ECG_NAME:
            iid = datakwargs.get('iid', False)
            val_rat = resplit.get('val', 0.1 if dataset == _settings.IIICSup_NAME else 0.2)

            if val_rat == 0.1:
                suffix = '_10'
            elif val_rat == 0.02:
                suffix = '_2'
            elif val_rat == 0.05:
                suffix = '_5'
            elif val_rat == 0.03:
                suffix = '_3'
            elif val_rat == 0.2 and dataset != _settings.ISRUC_NAME:
                suffix = '_20'
            else:
                suffix = ''
            suffix += '_iid' if iid else ''
            key = key + suffix
        cache_path = os.path.join(_settings.WORKSPACE, 'dircal_extern', dataset, f"{key}_{resplit['seed']}_res_dir.pkl")
        tP_ = pd.read_pickle(cache_path)
    elif baseline == 'uncal':
        tP_ = tP
    elif baseline == 'ts':
        import baselines.tempscale as tsc; reload(tsc)
        tempscale_obj = tsc.TemperatureScaling_Pytorch(torch.tensor(vS), torch.tensor(vY))
        tP_ = tempscale_obj.transform(tS, tY, to_prob=True)
    else:
        raise ValueError()
    return tP_, tY

def get_calibrated_preds(key,
                          dataset=dld._settings.CIFAR10_NAME,
                          val_split=dld.VALID, test_split=dld.TEST, datakwargs={},
                          kernel_name='RBF', kernel_kwargs={'h': 10},
                          proj_name=None, proj_kwargs={},
                          cal_kwargs={},
                          resplit_seed=None,
                          use_preds=False,
                         baseline=None,
                         normalize_imax=True,
                          **kwargs):
    nclass = dld.get_nclasses(dataset)
    if baseline:
        tP, tY = _get_calibrated_preds_baselines(key, baseline, dataset, val_split, test_split, datakwargs,
                                                 resplit={"seed": resplit_seed} if isinstance(resplit_seed, int) else resplit_seed,
                                                 use_preds=use_preds,
                                                 cache=ptd.CACHE if baseline in {"imax"} else ptd.NOCACHE,
                                                 **kwargs
                                                 )
        if baseline == 'imax' and normalize_imax:
            el = tP/(1-tP)
            tP = el/el.sum(1)[:, np.newaxis]
        return tP, tY
    res_df = _get_calibrated_preds(key, dataset, val_split, test_split, datakwargs,
                                   kernel_name, kernel_kwargs, proj_name, proj_kwargs, cal_kwargs,
                                   resplit = {"seed": resplit_seed} if isinstance(resplit_seed, int) else resplit_seed,
                                   use_preds=use_preds,
                                   **kwargs)
    res_df['pred'] = np.argmax(res_df.reindex(columns=['cP%d'%i for i in range(nclass)]).values, 1)
    return res_df