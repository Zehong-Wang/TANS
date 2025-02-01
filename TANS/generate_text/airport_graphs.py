#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import numpy as np
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Generate text data for airport nodes')
    parser.add_argument('--data_name', type=str, default='brazil', choices=['usa', 'brazil', 'europe'],
                        help='dataset name')
    parser.add_argument('--setting', type=str, default='text_free', choices=['text_free'],
                        help='text setting')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', choices=['gpt-4o-mini'],
                        help='model name')
    return parser.parse_args()


def check_dir(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


# Constants and templates
graph_stats = {
    'usa': {'graph_type': 'airport network', 'node_type': 'airport', 'edge_type': 'connectivity between airports',
            'total_node': 1190, 'total_edge': 28388},
    'brazil': {'graph_type': 'airport network', 'node_type': 'airport', 'edge_type': 'connectivity between airports',
               'total_node': 131, 'total_edge': 2137},
    'europe': {'graph_type': 'airport network', 'node_type': 'airport', 'edge_type': 'connectivity between airports',
               'total_node': 399, 'total_edge': 12385}
}

templates = {
    'base': "Given a node from a {} graph, where the node type is {} with {} nodes, and the edge type is {} with {} edges. ",
    'property': 'The value of property "{}" is {:.4f}, ranked at {} among {} nodes. ',
    'final': "Output the activity level of the node and provide reasons for your assessment. Your answer should be less than 200 words. "
}


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


def generate_text(node, methods, args):
    """Generate text description for a single node"""
    text = templates['base'].format(
        graph_stats[args.data_name]['graph_type'],
        graph_stats[args.data_name]['node_type'],
        graph_stats[args.data_name]['total_node'],
        graph_stats[args.data_name]['edge_type'],
        graph_stats[args.data_name]['total_edge']
    )

    for method, (value, rank) in methods.items():
        text += templates['property'].format(method, value, rank, graph_stats[args.data_name]['total_node'])

    text += templates['final']
    return text


def main():
    args = parse_args()
    topo_features = process_topological_features(args)

    # Generate texts for all nodes
    texts = [generate_text(node, methods, args)
             for node, methods in topo_features.items()]

    # Create and save dataframe
    df = pd.DataFrame({
        'question': texts,
        'answer': 'Error',
        'node_idx': range(len(texts))
    })

    save_dir = osp.join(osp.dirname(__file__), '../../data/response', args.setting, args.data_name)
    check_dir(save_dir)

    df['question'].to_csv(osp.join(save_dir, 'question.csv'), index=False)
    df[['node_idx', 'answer']].to_csv(osp.join(save_dir, f'answer_{args.model}.csv'), index=False, sep='\t')


if __name__ == "__main__":
    main()
