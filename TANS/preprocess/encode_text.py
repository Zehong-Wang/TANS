#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import numpy as np
import pandas as pd
import tqdm
import torch
from sentence_transformers import SentenceTransformer
import argparse


def encode_text_sbert(text, model):
    """Encode text using sentence-bert model"""
    return model.encode(text)


def check_path(path):
    """Create directory if it doesn't exist"""
    if not osp.exists(path):
        os.makedirs(path)


def get_default_text(data_name, node_text):
    """Get default text features for nodes"""
    if data_name in ['usa', 'europe', 'brazil']:
        return None

    path = osp.join('..', '..', 'data', 'dataset', data_name)
    text = [np.load(osp.join(path, f'{t}.npy')) for t in node_text]
    text = list(zip(*text))
    text = [' '.join(t) for t in text]
    return text


def get_addition_text(data_name, graph_type, llm_model, wo_neigh=False):
    """Get additional text descriptions from LLM"""
    dir = osp.join('..', '..', 'data', 'response', graph_type, data_name)
    path = osp.join(dir, f'answer_{llm_model}_wo_neigh.csv' if wo_neigh else f'answer_{llm_model}.csv')
    add_text = pd.read_csv(path, sep='\t')
    return add_text['answer'].tolist()


def get_text_embeddings(texts, model):
    """Get embeddings for a list of texts"""
    embeddings = []
    for text in tqdm.tqdm(texts):
        embeddings.append(encode_text_sbert(text, model))
    return torch.tensor(embeddings, dtype=torch.float64)


def parse_args():
    parser = argparse.ArgumentParser(description='Encode text data for graph nodes')
    parser.add_argument('--data_name', type=str, default='cora',
                        choices=['cora', 'pubmed', 'usa', 'brazil', 'europe'],
                        help='dataset name')
    parser.add_argument('--enc_model', type=str, default='minilm',
                        choices=['albert', 'roberta', 'minilm', 'mpnet'],
                        help='encoder model name')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini',
                        help='LLM model name')
    parser.add_argument('--node_text', nargs='+', default=['title', 'abstract'],
                        help='node text fields')
    parser.add_argument('--wo_neigh', action='store_true',
                        help='exclude neighbor info')
    return parser.parse_args()


def main():
    # Constants
    models = {
        'albert': 'paraphrase-albert-small-v2',
        'roberta': 'all-distilroberta-v1',
        'minilm': 'all-MiniLM-L12-v2',
        'mpnet': 'all-mpnet-base-v2'
    }
    node_types = {
        'cora': 'paper',
        'pubmed': 'paper',
        'usa': 'airport',
        'europe': 'airport',
        'brazil': 'airport'
    }

    # Get arguments
    args = parse_args()
    node_type = node_types[args.data_name]

    # Determine graph type
    if args.node_text == ['none']:
        graph_type = 'text_free'
    elif args.node_text == ['title']:
        graph_type = 'text_limit'
    else:
        graph_type = 'text_rich'

    # Get text data
    template = 'The node type is {}. The node description is {}. The additional node description is {}.'
    default_text = get_default_text(args.data_name, args.node_text)
    add_text = get_addition_text(args.data_name, graph_type, args.llm_model, args.wo_neigh)

    # Combine texts
    texts = []
    for i in range(len(add_text)):
        t = default_text[i] if default_text is not None else 'na'
        add_t = add_text[i]
        texts.append(template.format(node_type, t, add_t))

    # Get embeddings
    model_sbert = SentenceTransformer(models[args.enc_model])
    embeddings = get_text_embeddings(texts, model_sbert)

    # Save embeddings
    components = ['TANS', args.llm_model, args.enc_model]
    if args.node_text is not None:
        components = args.node_text + components
    file_name = '_'.join(components)
    if args.wo_neigh:
        file_name += '_wo_neigh'

    save_dir = osp.join('..', '..', 'data', 'text_emb', args.data_name)
    check_path(save_dir)
    torch.save(embeddings, osp.join(save_dir, f'{file_name}.pt'))


if __name__ == "__main__":
    main()
