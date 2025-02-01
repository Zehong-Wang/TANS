#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import numpy as np
import pandas as pd

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Generate text data for graph nodes')
    parser.add_argument('--data_name', type=str, default='cora', choices=['cora', 'pubmed'],
                        help='dataset name')
    parser.add_argument('--setting', type=str, default='text_limit', choices=['text_limit', 'text_rich'],
                        help='text setting')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', choices=['gpt-4o-mini'],
                        help='model name')
    parser.add_argument('--without_neigh', action='store_true',
                        help='exclude neighbor connectivity information')
    parser.add_argument('--num_neighbor', type=int, default=5,
                        help='number of neighbors to show')
    return parser.parse_args()


def check_dir(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


# Constants and templates
graph_stats = {
    'cora': {'graph_type': 'citation network', 'node_type': 'paper', 'edge_type': 'citation between papers',
             'total_node': 2708, 'total_edge': 5429},
    'pubmed': {'graph_type': 'citation network', 'node_type': 'paper', 'edge_type': 'citation between papers',
               'total_node': 19717, 'total_edge': 44324},
}

classes = {
    'cora': ['Case Based', 'Genetic Algorithms', 'Neural Networks', 'Probabilistic Methods', 'Reinforcement Learning',
             'Rule Learning', 'Theory'],
    'pubmed': ['Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2', 'Experimental Diabetes'],
}

templates = {
    'base': "Given a node from a {} graph, where the node type is {} with {} nodes, and the edge type is {} with {} edges. ",
    'node_text': 'The textual node name is {}. The node description is {}. ',
    'connectivity': 'One similar node has name {}, and the description is {}. ',
    'property': 'The value of property "{}" is {:.4f}, ranked at {} among {} nodes. ',
    'final': {
        'cora': "Output the potential three classes of the node and provide reasons for your assessment. The classes are {}. Your answer should be less than 200 words. ",
        'pubmed': "Output the potential class of the node and provide reasons for your assessment. The classes are {}. Your answer should be less than 200 words. "
    }
}


def load_data(args):
    """Load and prepare node data"""
    names = np.load(osp.join(osp.dirname(__file__), f'../../data/dataset/{args.data_name}/title.npy'),
                    allow_pickle=True)
    descriptions = np.load(osp.join(osp.dirname(__file__), f'../../data/dataset/{args.data_name}/abstract.npy'),
                           allow_pickle=True) if args.setting == 'text_rich' else ['na'] * len(names)
    return names, descriptions


def get_similar_nodes(args):
    """Get similar nodes for each node"""
    one_hop_neighs = np.load(osp.join(osp.dirname(__file__), f'../../data/property/{args.data_name}_one_hop.npy'),
                             allow_pickle=True).item()
    similar_nodes = {}
    for node, neighs in one_hop_neighs.items():
        if len(neighs) <= args.num_neighbor:
            similar_nodes[node] = neighs
        else:
            similar_nodes[node] = np.random.choice(neighs, args.num_neighbor, replace=False)
    return similar_nodes


def process_topological_features(args):
    """Process and rank topological features"""
    topo_features = np.load(osp.join(osp.dirname(__file__), f'../../data/property/{args.data_name}_topo.npy'),
                            allow_pickle=True).item()
    rename = {
        'clustering': 'Clustering Coefficient',
        'degree': 'Node Degree',
        'square_clustering': 'Square Clustering Coefficient',
        'closeness': 'Closeness Centrality',
        'betweenness': 'Betweenness Centrality',
    }
    topo_features = {rename[k]: v for k, v in topo_features.items()}

    # Calculate ranks
    topo_features_rank = {}
    for method, values in topo_features.items():
        sorted_values = dict(sorted(values.items(), key=lambda x: x[1], reverse=True))
        rank = 0
        pre_value = -1
        node2rank = {}
        for node, value in sorted_values.items():
            if value != pre_value:
                rank += 1
                pre_value = value
            node2rank[node] = rank
        topo_features_rank[method] = node2rank

    # Restructure features
    new_topo_features = {}
    for method, values in topo_features.items():
        for node, value in values.items():
            if node not in new_topo_features:
                new_topo_features[node] = {}
            new_topo_features[node][method] = (value, topo_features_rank[method][node])

    return new_topo_features


def generate_text(node, methods, names, descriptions, similar_nodes, args):
    """Generate text description for a single node"""
    text = templates['base'].format(
        graph_stats[args.data_name]['graph_type'],
        graph_stats[args.data_name]['node_type'],
        graph_stats[args.data_name]['total_node'],
        graph_stats[args.data_name]['edge_type'],
        graph_stats[args.data_name]['total_edge']
    )

    text += templates['node_text'].format(names[node], descriptions[node])

    if not args.without_neigh:
        text += 'The following are the connectivity information of the similar nodes: '
        for similar_node in similar_nodes[node]:
            text += templates['connectivity'].format(names[similar_node], descriptions[similar_node])

    for method, (value, rank) in methods.items():
        text += templates['property'].format(method, value, rank, graph_stats[args.data_name]['total_node'])

    text += templates['final'][args.data_name].format(classes[args.data_name])
    return text


def main():
    args = parse_args()
    names, descriptions = load_data(args)
    similar_nodes = get_similar_nodes(args)
    topo_features = process_topological_features(args)

    # Generate texts for all nodes
    texts = [generate_text(node, methods, names, descriptions, similar_nodes, args)
             for node, methods in topo_features.items()]

    # Create and save dataframe
    df = pd.DataFrame({
        'question': texts,
        'answer': 'Error',
        'node_idx': range(len(texts))
    })

    save_dir = osp.join(osp.dirname(__file__), '../../data/response', args.setting, args.data_name)
    check_dir(save_dir)

    question_file = 'question_wo_neigh.csv' if args.without_neigh else 'question.csv'
    answer_file = f'answer_{args.model}_wo_neigh.csv' if args.without_neigh else f'answer_{args.model}.csv'

    df['question'].to_csv(osp.join(save_dir, question_file), index=False)
    df[['node_idx', 'answer']].to_csv(osp.join(save_dir, answer_file), index=False, sep='\t')


if __name__ == "__main__":
    main()
