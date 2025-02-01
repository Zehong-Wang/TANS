import os
import os.path as osp
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp

from torch_sparse import SparseTensor
from torch_geometric.transforms import NormalizeFeatures, ToUndirected, AddLaplacianEigenvectorPE, AddRandomWalkPE, \
    OneHotDegree
from torch_geometric.datasets import Airports
from torch_geometric.utils import degree

citation_networks = ['cora', 'pubmed']
airport_networks = ['usa', 'europe', 'brazil']


def get_data(params, dataset_name):
    if dataset_name in citation_networks:
        data = citation_graph(params, dataset_name)
    elif dataset_name in airport_networks:
        data = airport_graph(params, dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return data


def citation_graph(params, dataset_name):
    data_dir = params['data_path']

    path = osp.join(data_dir, 'dataset', dataset_name, f'{dataset_name}.pt')

    data = torch.load(path)
    data = ToUndirected()(data)
    data.name = dataset_name

    text_path = '_'.join(params['node_text'] + [params['emb_method'], params['emb_model']])
    if params['wo_neigh']:
        text_path += '_wo_neigh'
    text_path = osp.join(data_dir, 'text_emb', dataset_name, "{}.pt".format(text_path))
    text_emb = torch.load(text_path).to(torch.float32)

    print("load data from", path)
    print("load text from", text_path)

    data.x = text_emb
    num_nodes = data.x.shape[0]
    num_dim = data.x.shape[1]
    num_edges = data.num_edges
    num_classes = data.y.max().item() + 1
    data.num_classes = num_classes
    data.num_dim = num_dim
    data.num_nodes = num_nodes
    data.num_edges = num_edges
    print(f"Dataset: {dataset_name}, #Nodes: {num_nodes}, #Edges: {num_edges}, #Classes: {num_classes}")

    return data


def airport_graph(params, dataset_name):
    data_dir = params['data_path']
    # dataset_name = params['data']

    data = Airports(root=osp.join(data_dir, 'dataset', 'airports'), name=dataset_name)[0]

    if params['emb_method'] == 'onehot':
        data.x = torch.eye(data.num_nodes)
    elif params['emb_method'] == 'degree':
        in_degree = degree(data.edge_index[1], data.num_nodes, dtype=torch.long).numpy()
        data.x = degree_bucketing(data.num_nodes, in_degree, max_degree=32)
    elif params['emb_method'] == 'eigen':
        data = AddLaplacianEigenvectorPE(k=32)(data)
        data.x = data.laplacian_eigenvector_pe
    elif params['emb_method'] == 'rw':
        data = AddRandomWalkPE(walk_length=32)(data)
        data.x = data.random_walk_pe
    elif params['emb_method'] == 'TANS':
        text_path = '_'.join([params['emb_method'], params['emb_model']])
        text_path = osp.join(data_dir, 'text_emb', dataset_name, f'{text_path}.pt')
        text_emb = torch.load(text_path).to(torch.float32)
        data.x = text_emb
    else:
        raise ValueError(f"Embedding method {params['emb_method']} not supported")

    data = ToUndirected()(data)

    data.num_dim = data.x.shape[1]
    data.num_classes = data.y.max().item() + 1
    data.num_nodes = data.x.shape[0]
    data.num_edges = data.edge_index.shape[1]
    data.name = dataset_name

    return data


def degree_bucketing(num_nodes, in_degree, max_degree=32):
    features = torch.zeros([num_nodes, max_degree])
    for i in range(num_nodes):
        try:
            features[i][min(in_degree[i], max_degree - 1)] = 1
        except:
            features[i][0] = 1
    return features
