#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import numpy as np
import tqdm
import torch
import networkx as nx

from torch_geometric.datasets import Airports
from torch_geometric.utils import to_networkx


def check_path(path):
    """Check if the folder exists and create it if it doesn't"""
    if not osp.exists(path):
        os.makedirs(path)
    else:
        print(f"The folder {path} already exists.")


def load_data(data_name, networkx=False):
    """Load graph data from different datasets"""
    if data_name in ['cora', 'pubmed']:
        data_path = osp.join(osp.dirname(__file__), f'../../data/dataset/{data_name}/{data_name}.pt')
        data = torch.load(data_path)
    elif data_name in ['usa', 'europe', 'brazil']:
        data_path = osp.join(osp.dirname(__file__), f'../../data/dataset/airports')
        data = Airports(root=data_path, name=data_name)[0]

    if networkx:
        return to_networkx(data, to_undirected=True)
    else:
        return data


def compute_node_properties(G):
    """Compute topological properties for each node"""
    return {
        'square_clustering': nx.square_clustering(G),
        'clustering': nx.clustering(G),
        'degree': nx.centrality.degree_centrality(G),
        'closeness': nx.centrality.closeness_centrality(G),
        'betweenness': nx.centrality.betweenness_centrality(G)
    }


def get_one_hop_neighbors(G):
    """Get one-hop neighbors for each node"""
    neighbors = {}
    for i in tqdm.tqdm(range(G.num_nodes)):
        neighbors[i] = list(G.edge_index[1][G.edge_index[0] == i].numpy().astype(int))
    return neighbors


def main():
    # Step 1: Get one-hop neighbors for citation graphs
    citation_datasets = ['cora', 'pubmed']
    for data_name in citation_datasets:
        G = load_data(data_name, networkx=False)
        print(f"Getting neighbors for {data_name}...")

        neighbors = get_one_hop_neighbors(G)
        save_folder = osp.join(osp.dirname(__file__), f'../../data/property')
        check_path(save_folder)
        save_path = osp.join(save_folder, f'{data_name}_one_hop.npy')
        np.save(save_path, neighbors)
        print(f"Saved neighbor info to {save_path}")

    # Step 2: Get node topological properties for all datasets
    all_datasets = ['usa', 'brazil', 'europe', 'cora', 'pubmed']
    for data_name in all_datasets:
        G = load_data(data_name, networkx=True)
        print(f"Processing {data_name}...")

        topo_features = compute_node_properties(G)
        save_folder = osp.join(osp.dirname(__file__), f'../../data/property')
        check_path(save_folder)
        save_path = osp.join(save_folder, f'{data_name}_topo.npy')
        np.save(save_path, topo_features)
        print(f"Saved topological features to {save_path}")



if __name__ == "__main__":
    main()
