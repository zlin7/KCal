import numpy as np
import os, glob
import bisect
import ipdb
import _settings as _settings
import torch
from torch.utils.data import Dataset
import pandas as pd
import utils as utils
from importlib import reload
from sklearn.model_selection import train_test_split
import torchvision
import tqdm
import persist_to_disk as ptd
reload(_settings)

import random
from PIL import ImageDraw

TRAIN = 'train'
TRAIN1 = 'train1' #train could be further split into train1 and val1
VALID1 = 'val1'
VALID = 'val'
TEST = 'test'

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x

@ptd.persistf()
def _split_data(seed, split_ratio, n):
    train, val = train_test_split(np.arange(n), test_size=split_ratio[1] / float(sum(split_ratio)),
                                  random_state=seed)
    return sorted(train), sorted(val)

@ptd.persistf()
def _get_perm(seed, n):
    np.random.seed(seed)
    perm = np.random.permutation(n)
    return perm

def get_split_indices(seed, split_ratio, n):
    perm = _get_perm(seed, n)
    split_ratio = np.asarray(split_ratio).cumsum() / sum(split_ratio)
    cuts = [int(_s* n) for _s in split_ratio]
    if len(split_ratio) == 3:
        return {TRAIN1: perm[:cuts[0]], VALID1:perm[cuts[0]:cuts[1]], VALID: perm[cuts[1]:]}
    else:
        assert len(split_ratio) == 2
        return {TRAIN: perm[:cuts[0]], VALID: perm[cuts[0]:]}

class DatasetWrapper(Dataset):
    def __init__(self, split=TRAIN):
        super(DatasetWrapper, self).__init__()
        self.split = split
        assert hasattr(self, 'DATASET'), "Please give this dataset a name"
        assert hasattr(self, 'LABEL_MAP'), "Please give a name to each class {NAME: class_id}"
    def is_train(self):
        return self.split == TRAIN or self.split == TRAIN1
    def is_test(self):
        return self.split == TEST
    def is_valid(self):
        return self.split == VALID or self.split == VALID1
    def idx2pid(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        return idx
    #def get_class_frequencies(self):
    #    raise NotImplementedError()
    @classmethod
    def split_data(cls, seed=_settings.RANDOM_SEED, split_ratio=[90, 10], n=50000):
        import data.dataloader as dld
        return dld._split_data(seed, tuple(split_ratio), n)

    @classmethod
    def split_data_new(cls, seed=_settings.RANDOM_SEED, split_ratio=[90, 10], n=50000):
        import data.dataloader as dld
        return dld.get_split_indices(seed, split_ratio, n)

    @classmethod
    def split_data_stratify(cls, labels, seed=_settings.RANDOM_SEED, split_ratio=[90, 10]):
        import data.dataloader as dld
        from collections import defaultdict
        labels = labels.copy().sort_index()
        ret = defaultdict(list)
        for c, cidx in labels.groupby(labels):
            curr = dld.get_split_indices(c+seed, split_ratio, len(cidx))
            for k, v in curr.items(): ret[k].append(cidx.index[v])
        return {k: np.concatenate(v) for k,v in ret.items()}

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        if idx >= self.n: raise IndexError("%d is out of range (%d elements)"%(idx, self.n))
        idx = self.indices[idx]
        x, y = self._data[idx]
        if hasattr(self, '_labels'): assert y == self._labels[idx]
        return x, y, idx

    def get_class_frequencies(self):
        return pd.Series({i: 1 for i in range(len(self.LABEL_MAP))})



class Cifar10Data(DatasetWrapper):
    DATASET = _settings.CIFAR10_NAME
    LABEL_MAP = {k: i for i,k in enumerate(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])}
    def __init__(self, split=TRAIN, seed=_settings.RANDOM_SEED, sample_for_train=False, resize_for_pretrained_model=False, image_size=224):
        super(Cifar10Data, self).__init__(split)
        self._seed = seed
        normalize = torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        if resize_for_pretrained_model: #https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/data_loading/data_loader.py
            from randaugment import RandAugment
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((image_size, image_size)),
                CutoutPIL(cutout_factor=0.5),
                RandAugment(),
                torchvision.transforms.ToTensor(),
            ])
            valid_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor(),
            ])
        else:
            from timm.data.auto_augment import auto_augment_transform
            valid_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize,
            ])
            train_transform = torchvision.transforms.Compose([
                #torchvision.transforms.RandomCrop(32, padding=4),
                #torchvision.transforms.RandomHorizontalFlip(),
                auto_augment_transform(config_str = 'original', hparams={"img_mean": (0.4914, 0.4822, 0.4465)}),
                torchvision.transforms.ToTensor(),
                normalize,
            ])
        self._data = torchvision.datasets.CIFAR10(_settings.CIFAR10_PATH, train=not self.is_test(), download=True,
                                                  transform=train_transform if split == TRAIN and sample_for_train else valid_transform)
        if self.is_test():
            self.indices = np.arange(len(self._data))
        else:
            split_ratio = [90, 10] if split in {TRAIN, VALID} else [80, 10, 10]
            self.indices = sorted(self.split_data_new(seed=self._seed, split_ratio=split_ratio, n=len(self._data))[split])
        self.n = len(self.indices)

class Cifar100Data(DatasetWrapper):
    DATASET = _settings.CIFAR100_NAME
    LABEL_MAP = {k:i for i,k in enumerate(['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'])}
    def __init__(self, split=TRAIN, seed=_settings.RANDOM_SEED, sample_for_train=False, resize_for_pretrained_model=False, image_size=224):
        super(Cifar100Data, self).__init__(split)
        self._seed = seed
        if resize_for_pretrained_model: #https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/data_loading/data_loader.py
            from randaugment import RandAugment
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((image_size, image_size)),
                CutoutPIL(cutout_factor=0.5),
                RandAugment(),
                torchvision.transforms.ToTensor(),
            ])
            valid_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor(),
            ])
        else:
            normalize = torchvision.transforms.Normalize(mean=(0.5074,0.4867,0.4411), std=(0.2011,0.1987,0.2025))
            valid_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize,
            ])
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ])
        self._data = torchvision.datasets.CIFAR100(_settings.CIFAR100_PATH, train=not self.is_test(), download=True,
                                                   transform=train_transform if self.is_train() and sample_for_train else valid_transform)
        if self.is_test():
            self.indices = np.arange(len(self._data))
        else:
            split_ratio = [90, 10] if split in {TRAIN, VALID} else [80, 10, 10]
            self.indices = sorted(self.split_data_new(seed=self._seed, split_ratio=split_ratio, n=len(self._data))[split])
        self.n = len(self.indices)

class SVHNData(DatasetWrapper):
    DATASET = _settings.SVHN_NAME
    LABEL_MAP = {i:i for i in range(10)}
    def __init__(self, split=TRAIN, seed=_settings.RANDOM_SEED, sample_for_train=False, resize_for_pretrained_model=False, image_size=224):
        super(SVHNData, self).__init__(split)
        self._seed = seed
        if resize_for_pretrained_model:
            from randaugment import RandAugment
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((image_size, image_size)),
                CutoutPIL(cutout_factor=0.5),
                RandAugment(),
                torchvision.transforms.ToTensor(),
            ])
            valid_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.ToTensor(),
            ])
        else:
            normalize = torchvision.transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442), std=(0.19803012, 0.20101562, 0.19703614))
            valid_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize,
            ])
            train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ])
        self._data = torchvision.datasets.SVHN(_settings.SVHN_PATH, split='test' if self.is_test() else 'train', download=True,
                                                   transform=train_transform if self.is_train() and sample_for_train else valid_transform)
        if self.is_test():
            self.indices = np.arange(len(self._data))
        else:
            split_ratio = [90, 10] if split in {TRAIN, VALID} else [80, 10, 10]
            self.indices = sorted(self.split_data_new(seed=self._seed, split_ratio=split_ratio, n=len(self._data))[split])
        self.n = len(self.indices)


def _count_classes(data):
    ret = []
    wnid_to_idx = sorted([(class_id, folder) for folder, class_id in data.wnid_to_idx.items()])
    for class_id,  folder in wnid_to_idx:
        n = len(glob.glob(os.path.join(data.root, data.split, folder, '*.JPEG')))
        ret.extend([class_id] * n)
    return pd.Series(ret)

class ImageNet1KData(DatasetWrapper):
    DATASET = _settings.ImageNet1K_NAME
    LABEL_MAP = {i:i for i in range(1000)}
    def __init__(self, split=TRAIN, seed=_settings.RANDOM_SEED, sample_for_train=False, image_size=224):
        super(ImageNet1KData, self).__init__(split)
        self._seed = seed
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        from randaugment import RandAugment
        valid_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        self._data = torchvision.datasets.ImageNet(_settings.ImageNet1K_PATH,
                                                   #split='val' if self.is_test() else 'train',
                                                   'train' if self.is_train() else 'val',
                                                   transform=train_transform if self.is_train() and sample_for_train else valid_transform)
        self._labels = _count_classes(self._data)
        if self.is_train():
            self.indices = np.arange(len(self._data))
        else:
            #self.indices = np.arange(len(self._data))
            #split_ratio = [96, 4] if split in {TRAIN, VALID} else [92, 4, 4]
            #need to split val into val and test, becuase training set follows a different distribution than all
            self.indices = sorted(self.split_data_stratify(self._labels, seed=self._seed, split_ratio=[50,50])[TRAIN if split == TEST else VALID])
        self.n = len(self.indices)

class ISRUCData(DatasetWrapper):
    DATASET = _settings.ISRUC_NAME
    DATA_PATH = _settings.ISRUC_PATH
    CHANNELS = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']  # https://arxiv.org/pdf/1910.06100.pdf
    LABEL_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4} #R is called "5" in the oringinal data, but use 4 here
    CLASS_FREQ = {"W": 221, "N1": 126, "N2": 281, "N3": 214, "R":62} #Basing on patient_8_1
    EPOCH_LENGTH = 30 * 200  #30 seconds * 200 Hz

    @classmethod
    def split_data(cls, seed=_settings.RANDOM_SEED, split_ratio=[70, 15, 15],
                   toy_version=False): #train:test = 9:1 in SLEEPER
        if toy_version: return [1,24], [2], [3]
        patients = [int(x) for x in os.listdir(cls.DATA_PATH) if x != '8'] #patient 8's EDF file is missing channels
        n_patients = len(patients)
        assert n_patients == 100 - 1
        train_val, test = train_test_split(patients, test_size = split_ratio[2] / float(sum(split_ratio)),
                                           random_state=seed)
        train, val = train_test_split(train_val, test_size = split_ratio[1] / float(sum(split_ratio[:2])), random_state=seed)
        #print(train[:5])
        return {TRAIN: sorted(train), VALID: sorted(val), TEST: sorted(test)}

    @classmethod
    def new_split_data(cls, seed=_settings.RANDOM_SEED, val_test_split_ratio=[6,24], old_split_ratio=[70, 15, 15]):
        #the original split has too many validation probably. But We want to keep the exact training set
        #So do this
        old_split = cls.split_data(seed, old_split_ratio)
        old_val_pids = ['1', '9', '11', '14', '34', '35', '36', '38', '39', '45', '56', '64', '67', '80', '81']
        old_test_pids = ['2', '20', '22', '27', '29', '32', '42', '53', '58', '65', '68', '72', '79', '87', '96']
        assert len(set(map(int, old_val_pids+old_test_pids)).intersection(set(old_split[TRAIN]))) == 0
        val_test = old_split[VALID] + old_split[TEST]
        val, test = train_test_split(val_test, test_size=val_test_split_ratio[1] / float(sum(val_test_split_ratio)), random_state=seed)
        return {TRAIN: sorted(old_split[TRAIN]), VALID: sorted(val), TEST: sorted(test)}

    @classmethod
    def clear_cache(cls):
        for pid in range(1, 100):
            caches = glob.glob(os.path.join(cls.DATA_PATH, f"{pid}/{pid}_*.pkl"))
            for cache_file in caches:
                if os.path.isfile(cache_file):
                    print("Removing {}".format(cache_file))
                    os.remove(cache_file)


    @classmethod
    def find_channels(cls, potential_channels):
        #channels = ['F3-A2', 'F4-A1', 'C3-A2', 'C4-A1', 'O1-A2', 'O2-A1'] #https://arxiv.org/pdf/1910.06100.pdf

        keep = {}
        for c in potential_channels:
            new_c = c.replace("-M2", "").replace("-A2", "").replace("-M1", "").replace("-A1", "")#https://www.ers-education.org/lrmedia/2016/pdf/298830.pdf
            if new_c in cls.CHANNELS:
                assert new_c not in keep
                keep[new_c] = c
        assert len(keep) == len(cls.CHANNELS), f"Something's wrong among columns={potential_channels}"
        return {v:k for k,v in keep.items()}

    @classmethod
    def load_data(cls, patient_ids=[1], clear_cache=False, save_mem=True):

        import time; st = time.time()

        labels_1, labels_2 = {}, {}
        actual_data, actual_columns = {}, {}
        bad_pids = []
        for pid in tqdm.tqdm(patient_ids):
            cache_path = os.path.join(cls.DATA_PATH, f'{pid}/{pid}_Channels={"_".join(cls.CHANNELS)}.pkl')
            if clear_cache and os.path.isfile(cache_path): os.remove(cache_path)
            if not os.path.isfile(cache_path):
                import mne
                EEG_raw_df = mne.io.read_raw_edf(os.path.join(cls.DATA_PATH, f'{pid}/{pid}.edf')).to_data_frame()
                try:
                    rename_dict = cls.find_channels(EEG_raw_df.columns)
                except Exception as err:
                    print(pid, err)
                    bad_pids.append(pid)
                    continue
                labels_1[pid] = pd.read_csv(os.path.join(cls.DATA_PATH, f"{pid}/{pid}_1.txt"), header=None)[0]
                labels_2[pid] = pd.read_csv(os.path.join(cls.DATA_PATH, f"{pid}/{pid}_2.txt"), header=None)[0]
                actual_data[pid] = EEG_raw_df.rename(columns=rename_dict).reindex(columns=cls.CHANNELS)
                actual_columns[pid] = {v: k for k, v in rename_dict.items()}

                # ipdb.set_trace()
                pd.to_pickle((labels_1[pid], labels_2[pid], actual_data[pid], actual_columns[pid]),
                             cache_path)
            else:
                labels_1[pid], labels_2[pid], actual_data[pid], actual_columns[pid] = pd.read_pickle(cache_path)
                assert len(actual_columns[pid]) == 6
            labels_1[pid][labels_1[pid] == 5] = 4
            labels_2[pid][labels_2[pid] == 5] = 4

            assert len(actual_data[pid]) % cls.EPOCH_LENGTH == 0
            n_epoch = int(len(actual_data[pid]) / cls.EPOCH_LENGTH)
            assert n_epoch == len(labels_1[pid])
            if n_epoch != len(labels_2[pid]):
                print(f"WARNING - Petient {pid}'s Label 2 is weird. Missing {n_epoch - len(labels_2[pid])} / {n_epoch}")

            if save_mem:
                epoch_cache_path = os.path.join(cls.DATA_PATH, f'{pid}/{pid}_epochs_Channels={"_".join(cls.CHANNELS)}')
                if not os.path.isdir(epoch_cache_path): os.makedirs(epoch_cache_path)
                actual_datapaths = []
                for curr_idx in range(n_epoch):
                    curr_idx_cache_path = os.path.join(epoch_cache_path, '%d.npy'%curr_idx)
                    actual_datapaths.append(curr_idx_cache_path)
                    if os.path.isfile(curr_idx_cache_path): continue
                    x = actual_data[pid].iloc[curr_idx * cls.EPOCH_LENGTH:(curr_idx+1) * cls.EPOCH_LENGTH].values.T
                    np.save(curr_idx_cache_path, x)
                actual_data[pid] = actual_datapaths

        print("Took %f seconds"%(time.time() - st))
        return labels_1, labels_2, actual_data, actual_columns

    def __init__(self, split=TRAIN,
                 seq_len=1, overlap=False,
                 clear_cache=False,
                 to_tensor=True,
                 save_mem=True,
                 toy_version=False,
                 split_ratios=[70, 6, 24],
                 seed=_settings.RANDOM_SEED,
                 iid=False):
        super(ISRUCData, self).__init__(split)
        self.save_mem = save_mem
        self._seed = seed

        self.seq_len = seq_len
        self.overlap = overlap

        #self.patients = sorted(self.split_data(seed=self._seed, toy_version=toy_version, split_ratio=split_ratios)[self.split])
        assert split_ratios[0] == 70
        self.patients = sorted(self.new_split_data(seed=self._seed, val_test_split_ratio=split_ratios[1:])[self.split])

        self.labels_1, self.labels_2, self.actual_data, self.actual_columns = self.load_data(self.patients, clear_cache, save_mem=save_mem)
        self.patients = sorted([pid for pid in self.labels_1.keys()]) #can have bad patient_ids

        self.npoints_by_patients = pd.Series(0, index=self.patients)

        for pid in self.patients:
            self.npoints_by_patients[pid] = len(self.labels_1[pid]) - self.seq_len + 1

        self.ndata = self.npoints_by_patients.sum()
        self.cumu_npoints_by_patients = self.npoints_by_patients.cumsum()

        self.to_tensor=to_tensor

    def get_class_frequencies(self):
        return pd.concat(self.labels_1.values(), ignore_index=True).value_counts()

    def __len__(self):
        return self.ndata

    def idx2pid(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        if idx >= self.ndata: raise IndexError("%d is out of range (%d elements)" % (idx, self.ndata))
        patient_id = self.cumu_npoints_by_patients.index[bisect.bisect(self.cumu_npoints_by_patients.values, idx)]
        return patient_id

    def get_raw_data(self, idx):
        import mne
        pid = self.idx2pid(idx)
        curr_idx = idx - (self.cumu_npoints_by_patients[pid] - self.npoints_by_patients[pid])
        EEG_raw = mne.io.read_raw_edf(os.path.join(self.DATA_PATH,  f'{pid}/{pid}.edf'), preload=True)
        return EEG_raw, curr_idx, pid, self.actual_columns[pid]

    def get_second_label(self, idx):
        patient_id = self.idx2pid(idx)
        curr_idx = idx - (self.cumu_npoints_by_patients[patient_id] - self.npoints_by_patients[patient_id])
        y = int(self.labels_2[patient_id][curr_idx])
        return y

    def __getitem__(self, idx):
        patient_id = self.idx2pid(idx)

        curr_idx = idx - (self.cumu_npoints_by_patients[patient_id] - self.npoints_by_patients[patient_id])
        st = curr_idx * self.EPOCH_LENGTH
        if self.save_mem:
            x = np.concatenate([np.load(self.actual_data[patient_id][_ci]) for _ci in range(curr_idx, curr_idx + self.seq_len)],
                               axis=1)
        else:
            x = self.actual_data[patient_id].iloc[st:(st + self.EPOCH_LENGTH * self.seq_len)].values.T

        y = int(self.labels_1[patient_id][curr_idx])

        if self.to_tensor and y is not None: y = torch.tensor(y, dtype=torch.long)
        return torch.tensor(x, dtype=torch.float), y, f"{patient_id}-{curr_idx}"

class ECGDataset(DatasetWrapper):
    # preprocessing following https://github.com/hsd1503/MINA
    DATASET = _settings.ECG_NAME
    DATA_PATH = os.path.join(_settings.ECG_PATH, "processed_data_full")
    LABEL_MAP = {"N": 0, "O": 1, "A": 2, "~": 3} #(5076, 2415, 758, 279)

    @classmethod
    def load_data_by_dataset(cls, dataset='train', data_path=DATA_PATH):
        assert dataset in {'train', 'val', 'test'}
        from collections import Counter
        import pickle as dill
        with open(os.path.join(data_path, 'mina_info.pkl'), 'rb') as fin:
            res = dill.load(fin)
            Y = res['Y_%s'%dataset]
            pids = res['pid_%s'%dataset]
            N = len(Y)
            assert N == len(pids)
        with open(os.path.join(data_path, 'mina_X_%s.bin'%dataset), 'rb') as fin:
            X = np.swapaxes(np.load(fin), 0, 1)
            assert 4 == X.shape[0] and X.shape[1] == N
        with open(os.path.join(data_path, 'mina_K_%s_beat.bin'%dataset), 'rb') as fin:
            K_beat = np.swapaxes(np.load(fin), 0, 1)
            assert K_beat.shape[0] == 4 and K_beat.shape[1] == N
        with open(os.path.join(data_path, 'mina_knowledge.pkl'), 'rb') as fin:
            res = dill.load(fin)
            K_rhythm = np.swapaxes(res['K_%s_rhythm'%dataset], 0, 1)
            K_freq = np.swapaxes(res['K_%s_freq' % dataset], 0, 1)
            assert K_rhythm.shape[0] == 4 and K_rhythm.shape[1] == N
            assert K_freq.shape[0] == 4 and K_freq.shape[1] == N

        print(Counter(Y))
        print(K_beat.shape, K_rhythm.shape, K_freq.shape)
        return X, Y, pids, K_beat, K_rhythm, K_freq #(nchannels=4, N, d)

    def __init__(self, split=TRAIN, seed=_settings.RANDOM_SEED, toy_version=False, float32=True, over_sample=False):
        super(ECGDataset, self).__init__(split)
        data_path = self.TOY_DATA_PATH if toy_version else self.DATA_PATH
        if over_sample: data_path += '_oversampled'
        if float32: data_path += '_float'
        self.X, self.Y, self.pids, self.K_beat, self.K_rhythm, self.K_freq = self.load_data_by_dataset(split, data_path)

        self.Y = np.asarray([self.LABEL_MAP[y] for y in self.Y])
        self.N = len(self.Y)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return (self.X[:, i, :], self.K_beat[:, i, :], self.K_rhythm[:, i, :], self.K_freq[:, i, :]), self.Y[i], f"{self.idx2pid(i)}_{i}"

    def idx2pid(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        return f"{self.split}{self.pids[idx]}"

    @classmethod
    def _collate_func(cls, batch):
        X = np.array([[_x[0][0][c] for _x in batch] for c in range(4)])
        K_beat = np.array([[_x[0][1][c] for _x in batch] for c in range(4)])
        K_rhythm = np.array([[_x[0][2][c] for _x in batch] for c in range(4)])
        K_freq = np.array([[_x[0][3][c] for _x in batch] for c in range(4)])

        #X = torch.tensor([[_x[0][0][c] for _x in batch] for c in range(4)], dtype=torch.float)
        #K_beat = torch.tensor([[_x[0][1][c] for _x in batch] for c in range(4)], dtype=torch.float)
        #K_rhythm = torch.tensor([[_x[0][2][c] for _x in batch] for c in range(4)], dtype=torch.float)
        #K_freq = torch.tensor([[_x[0][3][c] for _x in batch] for c in range(4)], dtype=torch.float)
        Y = torch.tensor([_x[1] for _x in batch], dtype=torch.long)
        #idx = torch.tensor([_x[2] for _x in batch], dtype=torch.long)
        idx = np.array([_x[2] for _x in batch])
        data_tuple = (X, K_beat, K_rhythm, K_freq)
        data_tuple = [torch.tensor(_, dtype=torch.float) for _ in data_tuple]
        #return tuple([_.permute(1,0,2) for _ in data_tuple]), Y, idx
        return tuple(data_tuple), Y, idx


class KernelData(DatasetWrapper):
    DATASET = None
    LABEL_MAP = None
    SAMPLE_SIMPLE, SAMPLE_MODSTRATIFY, SAMPLE_RAWSTRATIFY = 'simple', 'modstratify', 'rawstratify'
    def __init__(self, key, dataset=_settings.CIFAR10_NAME, split=TRAIN, seed=_settings.RANDOM_SEED, datakwargs={},
                 batch_size=300, batch_size_pred=64, sample_method=SAMPLE_MODSTRATIFY,
                 niters_per_epoch=10000,
                 use_preds=False,  #use_preds refer to the input to the metric space
                 incl_logits=False,  #for the sorting based method
                 gpu_id=-1,
                 **kwargs):
        super(KernelData, self).__init__(split)
        assert sample_method in {self.SAMPLE_SIMPLE, self.SAMPLE_MODSTRATIFY, self.SAMPLE_RAWSTRATIFY}
        self.sample_method = sample_method
        self.batch_size = batch_size
        self.batch_size_pred = batch_size_pred

        self.DATASET = f'KD-{dataset}'
        self.LABEL_MAP = {i:i for i in range(get_nclasses(dataset))}

        import pipeline.main
        X, _, preds = pipeline.main.get_embeddings_and_predictions(key, split=split, dataset=dataset, datakwargs=datakwargs, gpu_id=gpu_id)
        logits = preds.reindex(columns=['S%d'%_i for _i in range(get_nclasses(dataset))])
        self.X = logits if use_preds else X
        self.Y = preds['label']
        if incl_logits:
            self.logits = logits
        np.random.seed(seed+1)

        self.batch_size_pred = min(self.batch_size_pred, len(self.Y))
        self.niters_per_epoch = niters_per_epoch

        self.incl_logits = incl_logits
        self.nclasses = get_nclasses(dataset)

    def __len__(self):
        return self.niters_per_epoch

    def __getitem__(self, idx):
        if self.niters_per_epoch == 1:
            # Previously, we record the loss with test targets.
            # It had no impact on downstream experiments as we always take the last checkpoint.
            # Now, kernel_classifier.py understands LOO logic, so we don't need to use test tagets anymore.
            if self.incl_logits:
                all_data = torch.tensor(self.X.values, dtype=torch.float), \
                           torch.tensor(self.logits.values, dtype=torch.float), \
                           torch.tensor(self.X.values, dtype=torch.float), \
                           torch.tensor(self.logits.values, dtype=torch.float), \
                           torch.tensor(self.Y.values, dtype=torch.long), \
                           {k: (1,1) for k in range(self.nclasses)}
            else:
                all_data = torch.tensor(self.X.values, dtype=torch.float), \
                           torch.tensor(self.X.values, dtype=torch.float), \
                           torch.tensor(self.Y.values, dtype=torch.long), \
                           {k: (1,1) for k in range(self.nclasses)}
            label = torch.tensor(self.Y.values, dtype=torch.long)
            return all_data, label, list(range(len(label)))
        #sample self.batch_size many datapoints as the neighbors, and then sample some data to predict

        #We have to sample pred_indices first, otherwise some classes might not get sampled, if there are class imbalance
        pred_indices = np.random.choice(self.Y.index, self.batch_size_pred, replace=False)
        background = self.Y.loc[self.Y.index.difference(pred_indices)]
        if self.sample_method == self.SAMPLE_SIMPLE:
            cnt = len(background)
            indices = np.random.choice(background.index, min(cnt, self.batch_size), replace=False)
        elif self.sample_method == self.SAMPLE_MODSTRATIFY:
            cnt, indices = {}, []
            for k, all_idx_class_k in background.groupby(background):
                all_idx_class_k = all_idx_class_k.index
                if self.batch_size >= len(all_idx_class_k):
                    indices.append(all_idx_class_k)
                else:
                    indices.append(np.random.choice(all_idx_class_k, self.batch_size, replace=False))
                cnt[k] = (len(indices[-1]), len(all_idx_class_k))
            indices = np.concatenate(indices)
        elif self.sample_method == self.SAMPLE_RAWSTRATIFY:
            cnt, indices = {}, []
            for k, all_idx_class_k in background.groupby(background):
                all_idx_class_k = all_idx_class_k.index
                curr_batch_size = int(np.round(self.nclasses * self.batch_size /len(background) * len(all_idx_class_k)))
                indices.append(np.random.choice(all_idx_class_k, curr_batch_size, replace=False))
                cnt[k] = (len(indices[-1]), len(all_idx_class_k))
            indices = np.concatenate(indices)
        else:
            raise NotImplementedError()
        assert len(set(pred_indices).intersection(set(indices))) == 0

        if self.incl_logits:
            all_data = torch.tensor(self.X.loc[pred_indices].values, dtype=torch.float), \
                       torch.tensor(self.logits.loc[pred_indices].values, dtype=torch.float), \
                       torch.tensor(self.X.loc[indices].values, dtype=torch.float), \
                       torch.tensor(self.logits.loc[indices].values, dtype=torch.float), \
                       torch.tensor(self.Y.loc[indices].values, dtype=torch.long), \
                       cnt
        else:
            all_data = torch.tensor(self.X.loc[pred_indices].values, dtype=torch.float), \
                       torch.tensor(self.X.loc[indices].values, dtype=torch.float), \
                       torch.tensor(self.Y.loc[indices].values, dtype=torch.long), \
                       cnt

        label = torch.tensor(self.Y.loc[pred_indices].values, dtype=torch.long)

        idx = [idx * i for i in range(len(pred_indices))]
        return all_data, label, idx

    @classmethod
    def _collate_func(cls, batch):
        assert len(batch) == 1
        all_data, label, idx = batch[0]
        return all_data, label, idx



def get_default_dataset(dataset=_settings.CIFAR10_NAME, split=VALID, seed=_settings.RANDOM_SEED, **kwargs):
    if dataset == _settings.ISRUC_NAME: return ISRUCData(split=split, seed=seed, **kwargs)
    if dataset == _settings.ECG_NAME: return ECGDataset(split=split, seed=seed, **kwargs)
    if dataset == _settings.CIFAR10_NAME: return Cifar10Data(split=split, seed=seed, **kwargs)
    if dataset == _settings.CIFAR100_NAME: return Cifar100Data(split=split, seed=seed, **kwargs)
    if dataset == _settings.ImageNet1K_NAME: return ImageNet1KData(split=split, seed=seed, **kwargs)
    if dataset == _settings.SVHN_NAME: return SVHNData(split=split, seed=seed, **kwargs)
    if dataset == _settings.IIICSup_NAME:
        import data.iiic_data
        return data.iiic_data.IIIC_Sup(split=split, seed=seed, **kwargs)
    if dataset.startswith('KD-'):
        kwargs['dataset'] = dataset.split("KD-")[1]
        return KernelData(split=split, seed=seed, **kwargs)
    raise NotImplementedError()

def get_nclasses(dataset=_settings.CIFAR10_NAME):
    if dataset == _settings.ISRUC_NAME: return len(ISRUCData.LABEL_MAP)
    if dataset == _settings.ECG_NAME: return len(ECGDataset.LABEL_MAP)
    if dataset == _settings.CIFAR10_NAME: return len(Cifar10Data.LABEL_MAP)
    if dataset == _settings.CIFAR100_NAME: return len(Cifar100Data.LABEL_MAP)
    if dataset == _settings.ImageNet1K_NAME: return len(ImageNet1KData.LABEL_MAP)
    if dataset == _settings.SVHN_NAME: return len(SVHNData.LABEL_MAP)
    if dataset == _settings.IIICSup_NAME:
        import data.iiic_data
        return len(data.iiic_data.IIIC_Sup.CLASSES)
    if dataset.startswith('KD-'): return get_nclasses(dataset.split("KD-")[1])
    raise NotImplementedError()

def get_default_datawargs(dataset):
    if dataset in {_settings.CIFAR10_NAME, _settings.CIFAR100_NAME, _settings.SVHN_NAME}:
        return {'resize_for_pretrained_model': True}
    if dataset == _settings.IIICSup_NAME:
        return {'majority_only': True, 'iid':True}
    if dataset == _settings.ISRUC_NAME:
        return {'iid': True}
    if dataset == _settings.ECG_NAME:
        return {'iid': True}

    raise NotImplementedError()


