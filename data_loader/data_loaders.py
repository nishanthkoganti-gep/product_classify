# import modules
import pandas as pd
from os.path import join

# relative imports
from base import BaseDataLoader


class DataLoader(BaseDataLoader):
    """
    amazon data loader inherited from BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, training=True,
                 level=1):

        # set data directory
        self.data_dir = data_dir

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)
