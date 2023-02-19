from importlib import reload

import numpy as np
import os, glob

import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.signal import butter, lfilter
import scipy.stats
from scipy.interpolate import interp1d
import tqdm, ipdb

import _settings
from data.dataloader import TRAIN, VALID, TEST, get_split_indices

def denoise_channel(ts, bandpass, signal_freq):
    """
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1

    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts)
    return np.array(ts_out)


def noise_channel(ts, mode, degree):
    """
    Add noise to ts

    mode: high, low, both
    degree: degree of noise, compared with range of ts

    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)

    """
    len_ts = len(ts)
    num_range = np.ptp(ts) + 1e-4  # add a small number for flat signal

    ### high frequency noise
    if mode == 'high':
        noise = degree * num_range * (2 * np.random.rand(len_ts) - 1)
        out_ts = ts + noise

    ### low frequency noise
    elif mode == 'low':
        noise = degree * num_range * (2 * np.random.rand(len_ts // 100) - 1)
        x_old = np.linspace(0, 1, num=len_ts // 100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind='linear')
        noise = f(x_new)
        out_ts = ts + noise

    ### both high frequency noise and low frequency noise
    elif mode == 'both':
        noise1 = degree * num_range * (2 * np.random.rand(len_ts) - 1)
        noise2 = degree * num_range * (2 * np.random.rand(len_ts // 100) - 1)
        x_old = np.linspace(0, 1, num=len_ts // 100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind='linear')
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts
    return out_ts

class _IIICDatasets(torch.utils.data.Dataset):
    def __init__(self, split):
        super(_IIICDatasets, self).__init__()
        self.split = split

        self.bandpass1 = (1, 3)
        #         self.bandpass2 = (30, 60)
        self.n_length = 2000
        self.n_channels = 16
        self.n_classes = 6
        self.signal_freq = 200

    def add_noise(self, x, ratio):
        """
        Add noise to multiple ts
        Input:
            x: (n_length, n_channel)
        Output:
            x: (n_length, n_channel)
        """
        n_channels = len(x)
        for i in range(n_channels):
            if np.random.random() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i, :] = noise_channel(x[i, :], mode=mode, degree=0.05)
        return x

    def remove_noise(self, x, ratio):
        """
        Remove noise from multiple ts
        Input:
            x: (n_length, n_channel)
        Output:
            x: (n_length, n_channel)
        """
        n_channels = len(x)
        for i in range(n_channels):
            rand = np.random.random()
            if rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq)
            else:
                pass
        return x

    def crop(self, x):
        l = np.random.randint(1, self.n_length - 1)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x

    def channel_flipping(self, x):
        if np.random.random() > 0.5:
            x[:4, :], x[4:8, :] = x[4:8, :], x[:4, :]
            x[8:12, :], x[12:, :] = x[12:, :], x[8:12, :]
        return x

    def augment(self, x):
        # np.random.shuffle(x)
        t = np.random.random()
        if t > 0.75:
            x = self.add_noise(x, ratio=0.5)
        elif t > 0.5:
            x = self.remove_noise(x, ratio=0.5)
        elif t > 0.25:
            x = self.channel_flipping(x)
        else:
            x = self.crop(x)
        return x

    def _normalized(self, x):
        lb = np.percentile(x, 2.5)
        ub = np.percentile(x, 97.5)
        #x = (x - lb) / np.clip(ub - lb, 1e-3, None)
        x = x / np.clip(ub - lb, 1e-3, None)
        return x


class IIIC_Sup(_IIICDatasets):
    DATASET = _settings.IIICSup_NAME
    CLASSES = ['Others', 'Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA']
    LABEL_MAP = {_n:_i for _i, _n in enumerate(CLASSES)}
    def __init__(self, split, data_dir=_settings.IIICSup_PATH,
                 thres_record=3, seed=_settings.RANDOM_SEED,
                 majority_only=False, train_on='pseudo',
                 iid=False):

        super(IIIC_Sup, self).__init__(split)
        assert train_on in {'training', 'pseudo'}
        label_path = os.path.join(data_dir, 'data_1109')
        data_path = os.path.join(data_dir, '17_data_EEG_1115')
        def read_with_thres(prefix, thres):
            X = np.load(os.path.join(data_path, f'{prefix}_X.npy'))
            key = np.load(os.path.join(label_path, f'{prefix}_key.npy'))
            Y = np.load(os.path.join(label_path, f'{prefix}_Y.npy'))
            votes = pd.DataFrame(Y)
            votes['key'] = key
            votes['majority'] = np.argmax(Y, 1)
            mask = votes.sum(1) > thres
            if prefix == 'pseudo':
                train_key = np.load(os.path.join(label_path, f'training_key.npy'))
                mask = mask & votes['key'].isin(train_key)

            return X[mask], votes[mask]

        self.X, self.label = read_with_thres(train_on if split == TRAIN else 'test', thres_record)
        self.label = self.label.reset_index().drop('index', axis=1)
        if split == TRAIN:
            self.label = self.label.drop_duplicates(subset=['key'], keep='first')
            self.X = self.X[self.label.index]

        self.label['pid'] = self.label['key'].map(lambda s: s.split('_')[0])
        if split != TRAIN:
            if iid: #Changed to 0.9 and 0.1 on 1/8/2022
                sample_indices = get_split_indices(seed, [0.9, 0.1], len(self.label))
                self.label = self.label.iloc[sample_indices[TRAIN if split == TEST else VALID]]
                self.X = self.X[self.label.index]
            else:
                pids = sorted(self.label['pid'].unique())
                pid_indices = get_split_indices(seed, [0.9, 0.1], len(pids))
                pids = [pids[i] for i in pid_indices[TRAIN if split == TEST else VALID]]
                self.label = self.label[self.label['pid'].isin(pids)]
                self.X = self.X[self.label.index]
        self.majority_only = majority_only

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        record = self.label.iloc[idx]
        X = self.X[idx]
        X = self._normalized(X)
        V = record.iloc[:len(self.CLASSES)].values
        y = record['majority']
        entropy = scipy.stats.entropy(V.astype(float))

        if self.majority_only:
            all_labels = torch.tensor(record['majority'], dtype=torch.long)
        else:
            all_labels = torch.tensor([y, entropy] + list(V), dtype=torch.float)
        return torch.tensor(X, dtype=torch.float), all_labels, record['key']

    @classmethod
    def split_val_test(cls, keys, seed, ratio=[0.8, 0.2], iid=False):
        ratio = [ratio[1], ratio[0]] #to accomodate the ordering in get_split_indices
        df = pd.DataFrame({"key": keys})
        df['pid'] = df['key'].map(lambda s: s.split('_')[0])
        df = df.set_index('key')
        if iid:
            sample_indices = get_split_indices(seed, ratio, len(df))
            val_keys = df.iloc[sample_indices[VALID]].index
            test_keys = df.iloc[sample_indices[TRAIN]].index
        else:
            pids = sorted(df['pid'].unique())
            pid_indices = get_split_indices(seed, ratio, len(pids))
            val_pids = set([pids[i] for i in pid_indices[VALID]])
            test_pids = set([pids[i] for i in pid_indices[TRAIN]])
            val_keys = df[df['pid'].isin(val_pids)].index
            test_keys = df[df['pid'].isin(test_pids)].index
        return val_keys, test_keys

