# import modules
import torch
import pickle
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

# relative imports
from base import BaseDataLoader
from IPython.terminal.debugger import set_trace as keyboard


class AmazonDataset(Dataset):
    """Amazon Labels Dataset."""

    def __init__(self, pkl_file, level=1):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # obtain valid rows for level
        self.level = f'level{level}'
        valid_rows = data[self.level] >= 0

        # compute n_samples and n_labels
        self.n_samples = valid_rows.sum()
        self.n_labels = np.unique(data[self.level].values)

        # obtain titles and descriptions
        self.titles = data['title'][valid_rows]
        self.descriptions = data['description'][valid_rows]

        self.titles = self.titles.astype(np.int64)
        self.descriptions = self.descriptions.astype(np.int64)

        # obtain labels
        self.labels = data[self.level][valid_rows].values
        self.labels = self.labels.astype(np.int64)

        # compute class weights
        self.weights = compute_class_weight('balanced',
                                            np.unique(self.labels),
                                            self.labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        title = self.titles[idx]
        description = self.descriptions[idx]
        label = self.labels[idx]
        return title, description, label


class AmazonDataLoader(BaseDataLoader):
    """
    amazon data loader inherited from BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, training=True,
                 level=1):

        # set data directory
        self.data_dir = data_dir

        # setup pickle file
        if training:
            pkl_file = 'train.pkl'
        else:
            pkl_file = 'test.pkl'
        pkl_file = join(self.data_dir, pkl_file)

        self.dataset = AmazonDataset(pkl_file, level)

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)
