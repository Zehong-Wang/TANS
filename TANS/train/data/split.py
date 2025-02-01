import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

from torch_sparse import SparseTensor
from utils.utils import idx2mask


def get_split(label, params):
    splits = []
    for _ in params['seed']:
        if params['label_setting'] == 'ratio':
            split = ratio_split(label, params['train_ratio'], params['val_ratio'])
        elif params['label_setting'] == 'number':
            split = num_split(label, params['train_num'], params['val_num'])
        else:
            raise ValueError('Unknown label setting')
        splits.append(split)
    return splits


def ratio_split(label, train_ratio=0.1, val_ratio=0.1):
    n = label.shape[0]
    train_num = int(n * train_ratio)
    val_num = int(n * val_ratio)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + val_num]
    test_indices = perm[train_num + val_num:]

    train_mask = idx2mask(train_indices, label.shape[0])
    val_mask = idx2mask(val_indices, label.shape[0])
    test_mask = idx2mask(test_indices, label.shape[0])

    return {
        'train': train_mask,
        'val': val_mask,
        'test': test_mask
    }


def num_split(label, train_num=20, val_num=50):
    train_indices = []
    val_indices = []
    test_indices = []
    num_classes = label.max() + 1

    for l in range(num_classes):
        idx = torch.where(label == l)[0]
        idx = idx[torch.randperm(idx.size(0))]
        train_idx = idx[:train_num]
        val_idx = idx[train_num:train_num + val_num]
        test_idx = idx[train_num + val_num:]
        train_indices.append(train_idx)
        val_indices.append(val_idx)
        test_indices.append(test_idx)

    train_indices = torch.cat(train_indices, dim=0)
    val_indices = torch.cat(val_indices, dim=0)
    test_indices = torch.cat(test_indices, dim=0)

    train_mask = idx2mask(train_indices, label.shape[0])
    val_mask = idx2mask(val_indices, label.shape[0])
    test_mask = idx2mask(test_indices, label.shape[0])

    return {
        'train': train_mask,
        'val': val_mask,
        'test': test_mask
    }
