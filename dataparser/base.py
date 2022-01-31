import os

import numpy as np
import torch

from torch.utils import data
from torchvision.datasets.folder import (default_loader,
                                         has_file_allowed_extension,
                                         IMG_EXTENSIONS)

from dataparser.gas import GasDataset
from dataparser.power import PowerDataset
from dataparser.hepmass import HEPMASSDataset
from dataparser.miniboone import MiniBooNEDataset
from dataparser.bsds300 import BSDS300Dataset


def load_dataset(name, split, frac=None):
    """Loads and returns a requested dataset.

    Args:
        name: string, the name of the dataset.
        split: one of 'train', 'val' or 'test', the dataset split.
        frac: float between 0 and 1 or None, the fraction of the dataset to be returned.
            If None, defaults to the whole dataset.

    Returns:
        A Dataset object, the requested dataset.

    Raises:
         ValueError: If any of the arguments has an invalid value.
    """

    if split not in ['train', 'val', 'test']:
        raise ValueError('Split must be one of \'train\', \'val\' or \'test\'.')

    if frac is not None and (frac < 0 or frac > 1):
        raise ValueError('Frac must be between 0 and 1.')

    try:
        return {
            'power': PowerDataset,
            'gas': GasDataset,
            'hepmass': HEPMASSDataset,
            'miniboone': MiniBooNEDataset,
            'bsds300': BSDS300Dataset
        }[name](split=split, frac=frac)

    except KeyError:
        raise ValueError('Unknown dataset: {}'.format(name))


def get_uci_dataset_range(dataset_name):
    """
    Returns the per dimension (min, max) range for a specified UCI dataset.

    :param dataset_name:
    :return:
    """
    train_dataset = load_dataset(dataset_name, split='train')
    val_dataset = load_dataset(dataset_name, split='val')
    test_dataset = load_dataset(dataset_name, split='test')
    train_min, train_max = np.min(train_dataset.data, axis=0), np.max(train_dataset.data, axis=0)
    val_min, val_max = np.min(val_dataset.data, axis=0), np.max(val_dataset.data, axis=0)
    test_min, test_max = np.min(test_dataset.data, axis=0), np.max(test_dataset.data, axis=0)
    min_ = np.minimum(train_min, np.minimum(val_min, test_min))
    max_ = np.maximum(train_max, np.maximum(val_max, test_max))
    return np.array((min_, max_))


def get_uci_dataset_max_abs_value(dataset_name):
    """
    Returns the max of the absolute values of a specified UCI dataset.

    :param dataset_name:
    :return:
    """
    range_ = get_uci_dataset_range(dataset_name)
    return np.max(np.abs(range_))

def batch_generator(loader, num_batches=int(1e10)):
    batch_counter = 0
    while True:
        for batch in loader:
            yield batch
            batch_counter += 1
            if batch_counter == num_batches:
                return

