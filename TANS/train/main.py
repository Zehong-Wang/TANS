import os
import os.path as osp
import copy
import yaml
import random
import time

import wandb

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

from model.model import GNN
from data.data import get_data
from data.split import get_split
from utils.eval import evaluate, dataset2metric
from utils.args import get_args
from utils.logger import Logger
from utils.early_stop import EarlyStopping
from utils.utils import seed_everything, combine_dicts, target_sampling, DA_sampling, sampling, CMD, \
    random_string, flip_edges, get_device, get_device_from_model


def discrepancy(encoder, src_data, data, cmd, n_moments=3):
    z1 = encoder.encode(src_data.x, src_data.edge_index, src_data.edge_attr)
    z1 = encoder.pooling(z1, src_data.k_hop_edge_index, src_data.get('k_hop_edge_attr', None))
    z2 = encoder.encode(data.x, data.edge_index, data.edge_attr)
    z2 = encoder.pooling(z2, data.k_hop_edge_index, data.get('k_hop_edge_attr', None))
    return cmd.mmatch(z1, z2, n_moments=n_moments)


def train(model, data, split, optimizer, params):
    model.train()
    device = get_device_from_model(model)

    train_mask = split["train"]
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y[train_mask].squeeze().to(device)

    z = model.encode(x, edge_index)[train_mask]
    y_pred = model.predict(z)

    loss = F.cross_entropy(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval(model, data, split, params):
    model.eval()
    device = get_device_from_model(model)

    train_mask, val_mask, test_mask = split["train"], split["val"], split["test"]

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.squeeze().to(device)

    z = model.encode(x, edge_index)
    y_pred = model.predict(z)

    train_value = evaluate(y_pred, y, train_mask, params)
    val_value = evaluate(y_pred, y, val_mask, params)
    test_value = evaluate(y_pred, y, test_mask, params)

    return {'train': train_value, 'val': val_value, 'test': test_value, 'metric': dataset2metric[params['data']]}


def run(params):
    device = get_device(params)

    graph = get_data(params, params['data'])
    splits = get_split(graph.y, params)

    model = GNN(input_dim=graph.num_dim,
                hidden_dim=params['hidden_dim'],
                output_dim=graph.num_classes,
                activation=nn.PReLU,
                num_layers=params['num_layers'],
                backbone=params['backbone'],
                normalize=params['normalize'],
                dropout=params['dropout']).to(device)
    logger = Logger()

    for idx, (seed, split) in enumerate(zip(params['seed'], splits)):
        seed_everything(seed)
        data = copy.deepcopy(graph)

        if params['feat_noise'] != 0:
            data.x = (1 - params['feat_noise']) * data.x + params['feat_noise'] * torch.randn_like(data.x)
            print("Add Gaussian noise on nodes with level {} on target!".format(params['feat_noise']))

        if params['edge_noise'] != 0:
            data = flip_edges(data, p=params['edge_noise'])
            print('Randomly flip {} edges on target!'.format(params['edge_noise']))

        task_model = copy.deepcopy(model).to(device)

        optimizer = AdamW(task_model.parameters(), lr=params['lr'], weight_decay=params['decay'])
        stopper = EarlyStopping(patience=params["early_stop"])

        for epoch in range(1, params['epochs'] + 1):
            loss = train(task_model, data, split, optimizer, params)
            result = eval(task_model, data, split, params)

            is_stop = stopper(result)
            logger.log(idx, epoch, loss, result)
            if is_stop:
                print("Early Stopping at Epoch:", epoch)
                break

            wandb.log(
                {
                    "train/loss_train": loss,
                    "train/train": result['train'],
                    "train/val": result['val'],
                    "train/test": result['test'],
                    "train/metric": result['metric'],
                }
            )

        single_best = logger.get_single_best(idx)
        wandb.log({
            "best/train": single_best["train"],
            "best/val": single_best["val"],
            "best/test": single_best["test"],
        })

    best = logger.get_best()
    wandb.log({
        "final/train": "{:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']),
        "final/val": "{:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']),
        "final/test": "{:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']),
        "final/train_mean": best['train']['mean'],
        "final/val_mean": best['val']['mean'],
        "final/test_mean": best['test']['mean'],
        "final/train_std": best['train']['std'],
        "final/val_std": best['val']['std'],
        "final/test_std": best['test']['std'],
    })
    wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})

    wandb.finish()


def main():
    params = get_args()
    params['data_path'] = osp.join(os.path.dirname(__file__), '..', '..', 'data')

    dataset = params["data"]
    emb_method = params["emb_method"]
    node_text = '_'.join(params["node_text"])

    has_node_text = dataset in ['cora', 'pubmed']

    if params["use_params"]:
        with open(f"config/finetune.yaml", "r") as f:
            default_params = yaml.safe_load(f)
            if has_node_text:
                params.update(default_params[emb_method][node_text][dataset])
            else:
                params.update(default_params[emb_method][dataset])

    wandb.init(
        project="TANS",
        name="Data:{} | Backbone:{}".format(params["data"], params["backbone"]),
        config=params,
        mode="disabled" if params["debug"] else "online",  # sweep only works in online mode
        group=params['group'],
    )
    params = dict(wandb.config)
    print(params)

    run(params)


if __name__ == "__main__":
    main()
