import os, shutil
import ipdb
import pandas as pd

import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from tqdm import tqdm
import utils.utils as utils
import _settings
import datetime
import numpy as np

from sklearn.metrics import confusion_matrix
import pipeline.cutom_optimizers

def _to_device(data_or_model, device):
    if isinstance(device, tuple) or isinstance(device, list):
        device = device[0]
    def _to_device(d):
        try:
            return d.to(device)
        except: #if device is a list/tuple, we don't do anything as this should be dataparalle. (hacky, I know)
            return d
    if isinstance(data_or_model, tuple) or isinstance(data_or_model, list):
        return tuple([_to_device(x) for x in data_or_model])
    return _to_device(data_or_model)

def _state_dict(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    else:
        return model.state_dict()

def save_checkpoint(state, is_best, path):
    if is_best:
        filename = 'checkpoint_{0}_{1:.4f}.pth'.format(state['best_step'], state['best_val_loss'])
    else:
        filename = 'checkpoint_{0}.pth'.format(state['step'])
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    return filename

def move_tfboard_output(src, dst):
    for fname in os.listdir(src):
        if fname.startswith("events.out.tfevents"):
            shutil.copyfile(os.path.join(src, fname),
                            os.path.join(dst, fname))

def get_report(gt, pred, n_class=5):
    from sklearn.metrics import classification_report
    tmp_report = classification_report(gt, pred, output_dict=True)
    label_list = [i for i in range(n_class)]
    f1_list = [tmp_report[str(i)]['f1-score'] if str(i) in tmp_report else np.NaN for i in label_list]
    return f1_list

def get_class_weights(dataset):
    freqs = dataset.get_class_frequencies()
    weights = (freqs.sum() / freqs).sort_index()
    return weights.values



def confusion_matrix_to_F1_and_IoUs(mat):
    #mat[i,j] means true=i, pred=j
    f1s = np.zeros(len(mat))
    ious = np.zeros(len(mat))
    npreds = mat.sum(0)
    ntrues = mat.sum(1)
    for i in range(len(mat)):
        tp = mat[i][i]
        _prec = tp / max(1, npreds[i])
        _rec = tp / max(1, ntrues[i])
        _f1 = 2 * (_prec * _rec) / (_prec + _rec)
        if not pd.isnull(_f1): f1s[i] = _f1

        _iou = tp / (npreds[i] + ntrues[i] - tp)
        if not pd.isnull(_iou): ious[i] = _iou
    return f1s, ious


def compute_mat_f1(preds, gts, nclass=19):
    mat = []
    pa = [], []
    for i in range(len(gts)):
        target = gts[i].flatten()
        parsing = preds[i].flatten()
        mat.append(confusion_matrix(target, parsing, labels=np.arange(nclass)))
        pa[0].append(len(target))
        pa[1].append((target == parsing).mean())

    mat = sum(mat)
    f1s, ious = confusion_matrix_to_F1_and_IoUs(mat)
    pa = np.average(pa[1], weights=pa[0])
    return mat, f1s, ious, pa, np.average(f1s, weights=mat.sum(1)), np.average(ious, weights=mat.sum(1))
    pass

#=======================================================================================================================
class CallBack(object):
    TASK_CLASS = 'classification'
    TASK_REGRE = 'regression'
    TASK_SOFTCLASS = 'soft-classification'
    TASK_SEMSEG = 'SemanticSegmentation'
    def __init__(self,
                 train_data,
                 valid_data,

                 model_class=None,
                 model_kwargs=None,

                 continue_from_key = None, load_model_only=False,

                 n_epochs=10, batch_size=256, eval_steps=None,

                 gpu_id=0,
                 task_type=TASK_CLASS,
                 short_desc='',
                 debug=False,
                 **kwargs,
                 ):
        #Note that if we first train 10 epochs and then another 10, the result might be differet from 20 straight due to random seed?
        device = utils.gpuid_to_device(gpu_id)
        #self.log = log
        name = 'trainer-%s' % (train_data.DATASET)
        if model_class is None or model_kwargs is None:
            assert continue_from_key is not None, "Either pass in a model or continue from a previous training session"

        default_settings = {"gpu_id": gpu_id, "best_checkpoint": None, "last_checkpoint": None, 'task_type': task_type,
                                "model_class": model_class, "model_kwargs": model_kwargs,

                                "criterion_class": nn.CrossEntropyLoss if task_type == self.TASK_CLASS else nn.MSELoss,
                                "criterion_kwargs": {},

                                "optimizer_class": optim.AdamW,
                                "optimizer_kwargs": {"lr": 1e-3,  # weight_decay=1e-4
                                                     },
                                "scheduler_class": optim.lr_scheduler.ReduceLROnPlateau,
                                "scheduler_kwargs": {"mode": "min", "factor": 0.5, "patience": 30}
                                }
        if continue_from_key is not None:
            model, old_settings, checkpoint = self.load_state(name, continue_from_key, 'last')

            self.best_val_loss, self.best_val_step = checkpoint['best_val_loss'], checkpoint['best_step']
            self.best_val_mse = checkpoint.get('best_val_mse', 1e9)
            self.best_val_f1  = checkpoint.get('best_val_f1', 0.)
            self.epoch, self.step = checkpoint['epoch'], checkpoint['step']

            if load_model_only:
                settings = kwargs.copy()
                for _k in ['model_class', 'model_kwargs']:
                    settings[_k] = old_settings[_k]
                for _setting_key, _setting_val in default_settings.items():
                    settings.setdefault(_setting_key, _setting_val)
            else:
                settings = old_settings
            settings['from_key'] = continue_from_key
        else:
            model = model_class(**model_kwargs)
            settings = kwargs.copy()
            for _setting_key, _setting_val in default_settings.items():
                settings.setdefault(_setting_key, _setting_val)
            self.best_val_f1, self.best_val_loss = 0.0, 1e9
            self.best_val_mse = 1e9
            self.epoch, self.step = 0, 0
            self.best_val_loss, self.best_val_step = np.infty, -1
        settings.update({"batch_size": batch_size, "eval_steps": eval_steps, "n_epochs": n_epochs})
        self.criterion = settings['criterion_class'](**settings['criterion_kwargs'])
        self.optimizer = pipeline.cutom_optimizers.create_optimizer(settings['optimizer_class'], model, **settings['optimizer_kwargs'])
        if settings['scheduler_class'] is not None:
            settings['scheduler_step_on'] = settings['scheduler_kwargs'].pop('step_on', 'val_loss')
            self.scheduler = settings['scheduler_class'](self.optimizer, **settings['scheduler_kwargs'])
        else:
            self.scheduler = None
        if isinstance(device, tuple):
            self.model = torch.nn.DataParallel(model, device_ids=device)
        else:
            self.model = model
        _to_device(self.model, device)
        self.settings = settings
        self.task_type = settings['task_type']

        #For this run
        self.key = "%s-%s%s" % (model.__class__.__name__, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),short_desc)
        #self.save_path = os.path.join(_settings.WORKSPACE, name, self.key)
        self.save_path = os.path.join(_settings.WORKSPACE, 'trainers', f"{name.replace('trainer-', '')}-{self.key}")
        if not os.path.isdir(self.save_path): os.makedirs(self.save_path)
        if continue_from_key is not None:
            move_tfboard_output(self.save_path.replace(self.key, continue_from_key),
                                          self.save_path)
            shutil.copyfile(os.path.join(self.save_path.replace(self.key, continue_from_key), 'log.log'),
                            os.path.join(self.save_path, 'log.log'))


        #A logger simultaneously writes to many locations.
        self.full_logger = utils.FullLogger(logger=utils.get_logger(name=name + self.key, log_path=os.path.join(self.save_path, 'log.log')),
                                            neptune_ses = utils.get_neptune_logger(self.key, tag=[train_data.DATASET] + ([] if short_desc == '' else [short_desc]), continue_from_key=continue_from_key, debug=debug),
                                            tbwriter=SummaryWriter(log_dir=self.save_path))
        self.full_logger.info("Start training ...")


        self.device = device
        self.train_data, self.valid_data = train_data, valid_data
        self.n_class = len(self.train_data.LABEL_MAP)

    def train(self, num_workers=0, seed=7):
        utils.set_all_seeds(seed)
        batch_size, eval_steps, n_epochs = self.settings['batch_size'], self.settings['eval_steps'], self.settings['n_epochs']
        collate_fn = getattr(self.train_data, '_collate_func', None)
        train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        if self.valid_data is not None:
            valid_loader = DataLoader(dataset=self.valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        model, optimizer, scheduler, criterion = self.model, self.optimizer, self.scheduler, self.criterion

        self.init_training_hist()
        while self.epoch < n_epochs:
            self.epoch += 1
            for data, target, _ in tqdm(train_loader, desc='Training Epoch=%d'%self.epoch, ncols=_settings.NCOLS):
                model.train()

                optimizer.zero_grad()
                output = model(_to_device(data, self.device))
                loss = criterion(output, _to_device(target, self.device))
                loss.backward()
                optimizer.step()

                self.step += 1

                # bookkeeping
                self.bookkeep_training_step(loss, output, target)

                if eval_steps is not None and self.step % eval_steps == 0:
                    self.bookkeep_training_pereval()
                    if self.valid_data is not None: self.on_validation(valid_loader)

                    if self.check_early_stop():
                        self.on_training_end()
                        return self.key

                if not os.path.exists(self.save_path):
                    raise Exception("It seems like the path is deleted. Assuming this means stop!")
            if eval_steps is None:
                self.bookkeep_training_pereval()
                if self.valid_data is not None: self.on_validation(valid_loader)
                if self.check_early_stop():
                    self.on_training_end()
                    return self.key
        self.on_training_end()
        return self.key

    def init_training_hist(self):
        self.training_history = {'train_pred': [],
                                 'train_gt': [],
                                 'confusion_matrix': [],
                                 'pixel_accuracy': []}

    def bookkeep_training_step(self, loss, output, target):
        self.full_logger.log_scalar('Loss/Train', loss.item(), self.step)
        if self.task_type == self.TASK_SEMSEG:
            #label and output are too expensive to store
            out = output[0]
            parsing = out.detach().argmax(1).cpu().numpy().flatten()
            target = target.flatten()
            mat = confusion_matrix(target, parsing, labels=np.arange(len(self.train_data.CLASSES)))
            self.training_history['confusion_matrix'].append(mat)
            self.training_history['pixel_accuracy'].append((len(target), (target.numpy() == parsing).mean()))

        if self.task_type == self.TASK_CLASS:
            self.training_history['train_pred'].extend(np.argmax(output.tolist(), axis=1))
        elif self.task_type == self.TASK_REGRE:
            self.training_history['train_pred'].extend(output.tolist())
        elif self.task_type == self.TASK_SOFTCLASS:
            self.training_history['train_pred'].extend(np.argmax(output.tolist(), axis=1))
        self.training_history['train_gt'].extend(target.tolist())

    def bookkeep_training_pereval(self):
        if self.task_type == self.TASK_SEMSEG:
            mat = sum(self.training_history['confusion_matrix'])
            f1s, ious = confusion_matrix_to_F1_and_IoUs(mat)
            pa = np.average([_pa for _, _pa in self.training_history['pixel_accuracy']],
                             weights=[_cnt for _cnt, _ in self.training_history['pixel_accuracy']])

            self.full_logger.log_scalar('Train/PixAcc', pa, self.step)
            for cls_name, cls in self.train_data.LABEL_MAP.items():
                self.full_logger.log_scalar('Train/F1_%s' % cls_name, f1s[cls], self.step)
                self.full_logger.log_scalar('Train/IoU_%s' % cls_name, ious[cls], self.step)
        if self.task_type == self.TASK_CLASS or self.task_type == self.TASK_SOFTCLASS:
            train_gt = self.training_history['train_gt']
            if self.task_type == self.TASK_SOFTCLASS: train_gt = [int(x[0]) for x in train_gt]
            tmp_report = get_report(train_gt, self.training_history['train_pred'], n_class=self.n_class)
            train_f1 = np.mean(tmp_report) #TODO: in the future I think this should consider weights
            self.full_logger.log_scalar('Train/F1', train_f1, self.step)
            if hasattr(self.train_data, 'LABEL_MAP') and len(self.train_data.LABEL_MAP) <= 10:
                for cls_name, cls in self.train_data.LABEL_MAP.items():
                    self.full_logger.log_scalar('Train/F1_%s' % cls_name, tmp_report[cls], self.step)
            self.full_logger.log_scalar('Train/Acc', sum(np.asarray(self.training_history['train_pred']) == np.asarray(train_gt)) / float(len(train_gt)), self.step)
        elif self.task_type == self.TASK_REGRE:
            diff = np.asarray(self.training_history['train_gt']) - np.asarray(self.training_history['train_pred'])
            diff_sqr = np.power(diff, 2)
            self.full_logger.log_scalar('Train/MSE', np.mean(diff_sqr), self.step)
            if hasattr(self.train_data, 'LABEL_MAP'):
                for cls_name, cls in self.train_data.LABEL_MAP.items():
                    self.full_logger.log_scalar('Train/MSE_%s' % cls_name, np.mean(diff_sqr[:, cls]), self.step)
        self.init_training_hist()



    def on_validation(self, valid_loader):
        ## Validation
        #tmp_report, val_f1, val_loss = self._eval_model(self.model, valid_loader, self.device, self.criterion)
        all_val_pred, all_val_gt, val_loss, _, _ = self._eval_model(self.model, valid_loader, self.device, self.criterion,
                                                                    task_type=self.task_type)
        val_f1, val_mse = np.NaN, np.NaN
        if self.task_type == self.TASK_SEMSEG:
            mat, f1s, ious, pa, val_f1, val_iou = compute_mat_f1(all_val_pred, all_val_gt, nclass=len(valid_loader.dataset.CLASSES))
            self.full_logger.log_scalar('Loss/Validation', val_loss, self.step)
            self.full_logger.log_scalar('Validation/PixAcc', pa, self.step)
            for cls_name, cls in self.train_data.LABEL_MAP.items():
                self.full_logger.log_scalar('Validation/F1_%s' % cls_name, f1s[cls], self.step)
                self.full_logger.log_scalar('Validation/IoU_%s' % cls_name, ious[cls], self.step)
        elif self.task_type == self.TASK_CLASS or self.task_type == self.TASK_SOFTCLASS:
            if self.task_type == self.TASK_SOFTCLASS: all_val_gt = [int(x[0]) for x in all_val_gt]
            tmp_report = get_report(all_val_gt, all_val_pred, n_class=self.n_class)
            val_f1 = np.mean(tmp_report)

            self.full_logger.log_scalar('Loss/Validation', val_loss, self.step)
            self.full_logger.log_scalar('Validation/F1', val_f1, self.step)
            if hasattr(self.train_data, 'LABEL_MAP') and len(self.train_data.LABEL_MAP) <=10:
                for cls_name, cls in self.train_data.LABEL_MAP.items():
                    self.full_logger.log_scalar('Validation/F1_%s' % cls_name, tmp_report[cls], self.step)
            val_acc = sum(np.asarray(all_val_pred) == np.asarray(all_val_gt)) / float(len(all_val_gt))
            self.full_logger.log_scalar('Validation/Acc', val_acc, self.step)
            self.best_val_f1 = max(self.best_val_f1, val_f1)
        elif self.task_type == self.TASK_REGRE:
            diff = np.asarray(all_val_gt) - np.asarray(all_val_pred)
            diff_sqr = np.power(diff, 2)

            val_mse = np.mean(diff_sqr)
            self.full_logger.log_scalar('Loss/Validation', val_loss, self.step)
            self.full_logger.log_scalar('Validation/MSE', val_mse, self.step)
            if hasattr(self.train_data, 'LABEL_MAP'):
                for cls_name, cls in self.train_data.LABEL_MAP.items():
                    self.full_logger.log_scalar('Validation/MSE_%s' % cls_name, np.mean(diff_sqr[:, cls]), self.step)
            self.best_val_mse = min(self.best_val_mse, val_mse)

        # update learning rate, if condition
        if self.scheduler is not None:
            step_on = self.settings['scheduler_step_on']
            if step_on is None:
                self.scheduler.step()
            elif step_on == 'val_loss':
                self.scheduler.step(val_loss)
            elif step_on == 'epoch_and_acc':
                self.scheduler.step(self.epoch, val_acc)
            else:
                raise ValueError()

        # save best
        if val_loss < self.best_val_loss:
            self.best_val_step = self.step
            self.best_val_loss = val_loss

            self.full_logger.info(f"Best Model at {self.best_val_step}, f1={val_f1}, mse={val_mse}, loss={self.best_val_loss}")
            old_best = self.settings.get('best_checkpoint', None)
            self.settings['best_checkpoint'] = save_checkpoint({
                'epoch': self.epoch,
                'step': self.step,
                'val_loss': val_loss,
                'state_dict': _state_dict(self.model),
                'best_step': self.best_val_step,
                'val_f1': val_f1,
                'best_val_f1': self.best_val_f1,
                'val_mse': val_mse,
                'best_val_mse': self.best_val_mse,
                'best_val_loss': self.best_val_loss,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
            }, is_best=True, path=self.save_path)
            if old_best is not None and os.path.isfile(old_best):
                os.remove(old_best)

    def check_early_stop(self):
        try:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.full_logger.info('====> Learning Rates: {}'.format(current_lr))
            self.full_logger.log_scalar('LearningRate', current_lr, self.step)
            if current_lr < 1e-5:
                self.full_logger.info("Early stop")
                return True
        except:
            pass
        return False

    def on_validation_end(self):
        pass

    @classmethod
    def load_state(cls, name, key, mode='last', device=None):
        assert mode in {"last", "best"}
        #save_path = os.path.join(_settings.WORKSPACE, name, key)
        save_path = os.path.join(_settings.WORKSPACE, 'trainers', f"{name.replace('trainer-', '')}-{key}")
        settings = torch.load(os.path.join(save_path, 'settings.pkl'), map_location=device)
        model = settings['model_class'](**settings['model_kwargs'])
        if device: _to_device(model, device)
        checkpoint = torch.load(os.path.join(save_path, settings['%s_checkpoint' % mode]), map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        return model, settings, checkpoint


    def on_training_end(self):
        if self.task_type == self.TASK_SEMSEG:
            self.full_logger.info(f"Best (weighted) F1 on Validation = {self.best_val_f1}")
        elif self.task_type == self.TASK_CLASS:
            self.full_logger.info(f"Best F1 on Validation = {self.best_val_f1}")
        elif self.task_type == self.TASK_REGRE:
            self.full_logger.info(f"Best MSE on Validation = {self.best_val_mse}")

        self.settings['last_checkpoint'] = save_checkpoint({'epoch': self.epoch,
                                                            'step': self.step,
                                                                      'val_loss': None,
                                                            'state_dict': _state_dict(self.model),
                                                            'best_step': self.best_val_step,
                                                                'val_f1': None,
                                                            'best_val_f1': self.best_val_f1,
                                                                'val_mse': None,
                                                                'best_val_mse': self.best_val_mse,
                                                                'best_val_loss': self.best_val_loss,
                                                                'optimizer': self.optimizer.state_dict(),
                                                                'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
                                                            }, is_best=False, path=self.save_path)

        torch.save(self.settings, os.path.join(self.save_path, 'settings.pkl'))


    @classmethod
    def _eval_model(cls, model,  dataloader, device, criterion=None, task_type=TASK_CLASS, forward_kwargs={}):
        model.eval()
        _to_device(model, device)

        all_val_pred = []
        all_val_gt = []
        all_val_loss = 0.0
        all_val_indices = []
        outputs = []
        with torch.no_grad():
            for data, target, indices in tqdm(dataloader, ncols=_settings.NCOLS, desc='Evaluating...'):
                data = _to_device(data, device)
                target = _to_device(target, device)
                output = model(data, **forward_kwargs)
                if task_type == cls.TASK_SEMSEG:
                    out = output[0]
                    parsing = out.detach().argmax(1).cpu().int().numpy()
                    all_val_pred.extend(parsing)
                    all_val_gt.extend(target.cpu().int().numpy())
                else:
                    outputs.extend(output.tolist())
                    if task_type == cls.TASK_CLASS or task_type == cls.TASK_SOFTCLASS:
                        all_val_pred.extend(np.argmax(output.tolist(), axis=1))
                    elif task_type == cls.TASK_REGRE:
                        all_val_pred.extend(output.tolist())
                    all_val_gt.extend(target.tolist())
                try:
                    all_val_indices.extend(indices.tolist())
                except:
                    all_val_indices.extend(list(indices))
                if criterion is not None: all_val_loss += criterion(output, target).cpu().numpy() * len(indices)
        return all_val_pred, all_val_gt, all_val_loss / len(all_val_gt), outputs, all_val_indices

    @classmethod
    def eval_test(cls, key, test_data, device=None, batch_size=256, get_indices=False, mode='best',
                  raw_output=False):
        name = 'trainer-%s' % (test_data.DATASET)
        model, settings, _ = cls.load_state(name, key, mode=mode)
        task_type = settings['task_type']


        device = device or utils.gpuid_to_device(0)
        _to_device(model, device)
        collate_fn = test_data._collate_func if hasattr(test_data, '_collate_func') else None
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        criterion = settings['criterion_class'](**settings['criterion_kwargs'])

        all_test_pred, all_test_gt, test_loss, outputs, all_test_indices = cls._eval_model(model, test_loader, device, criterion, task_type=task_type)
        if not raw_output:
            if task_type == cls.TASK_SEMSEG:
                mat, f1s, ious, pa, test_f1, test_iou = compute_mat_f1(all_test_pred, all_test_gt, nclass=len(test_loader.dataset.CLASSES))
                print("Loss/Test = %f"%(test_loss))
                print("Test/PixAcc = %f" % (pa))
                print("Test/F1 = %f"%(test_f1))
                print("Test/IoU = %f" % (test_iou))
                for cls_name, cls in test_data.LABEL_MAP.items():
                    print('Test/F1_%s = %f' % (cls_name, f1s[cls]))
                    print('Test/IoU_%s = %f' % (cls_name, ious[cls]))
                return mat, f1s, ious, pa, test_f1, test_iou, test_loss

            elif task_type == cls.TASK_CLASS or task_type == cls.TASK_SOFTCLASS:
                if task_type == cls.TASK_SOFTCLASS: all_test_gt = [int(x[0]) for x in all_test_gt]
                tmp_report = get_report(all_test_gt, all_test_pred, n_class=len(test_data.LABEL_MAP))
                test_f1 = np.mean(tmp_report)

                print("Loss/Test = %f"%(test_loss))
                print("Test/F1 = %f"%(test_f1))
                if hasattr(test_data, 'LABEL_MAP') and len(test_data) <= 10:
                    for cls_name, cls in test_data.LABEL_MAP.items():
                        print('Validation/F1_%s = %f' % (cls_name, tmp_report[cls]))
            elif task_type == cls.TASK_REGRE:
                diff = np.asarray(all_test_gt) - np.asarray(all_test_pred)
                diff_sqr = np.power(diff, 2)

                test_mse = np.mean(diff_sqr)
                print('Loss/Test=%f'%(test_loss))
                print('Test/MSE=%f' % (test_mse))
                if hasattr(test_data, 'LABEL_MAP'):
                    for cls_name, cls in test_data.LABEL_MAP.items():
                        print('Validation/MSE_%s = %f' % (cls_name, np.mean(diff_sqr[:, cls])))
        if get_indices:
            return all_test_pred, all_test_gt, test_loss, outputs, all_test_indices
        else:
            return all_test_pred, all_test_gt, test_loss, outputs

if __name__ == '__main__':
    pass