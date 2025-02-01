import argparse


def get_args():
    parser = argparse.ArgumentParser('Evaluation')

    # General Config
    parser.add_argument('--pt_data', type=str, default='na')
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--use_params', action='store_true', help='Whether to use the params')
    parser.add_argument('--group', type=str, default='base')
    parser.add_argument('--debug', action='store_true', help='Debug Mode')

    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument('--device', type=int, default=0)

    # Emb Config
    parser.add_argument('--node_text', type=str, nargs='+', default=['title', 'abstract'])
    parser.add_argument('--emb_model', type=str, default='minilm')
    parser.add_argument('--emb_method', type=str, default='TANS')

    # Model Config
    parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'mlp'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--normalize', type=str, default='none')

    # Training Config
    parser.add_argument('--label_setting', type=str, default='ratio', choices=['ratio', 'number'])
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training nodes in target dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of training nodes in target dataset')
    parser.add_argument('--train_num', type=int, default=20, help='Number of training nodes per class')
    parser.add_argument('--val_num', type=int, default=30, help='Number of validation nodes per class')

    parser.add_argument('--pt_epochs', type=int, default=500, help='Epochs for pretraining')
    parser.add_argument('--pt_lr', type=float, default=1e-3, help='Learning rate for pretraining')
    parser.add_argument('--pt_decay', type=float, default=0, help='Weight decay for pretraining')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for fine-tuning')
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--decay', type=float, default=1e-5, help='Weight decay for fine-tuning')

    # Others
    parser.add_argument('--feat_noise', type=float, default=0.0)
    parser.add_argument('--edge_noise', type=float, default=0.0)
    parser.add_argument('--eval_disc', action='store_true',
                        help='Whether to evaluate the discrepancy between source and target')
    parser.add_argument('--wo_neigh', action='store_true', help='Without using neighbors')

    args = parser.parse_args()
    return vars(args)


def get_da_args():
    parser = argparse.ArgumentParser('Evaluation')

    # General Config
    parser.add_argument('--pt_data', type=str, default='europe')
    parser.add_argument('--data', type=str, default='usa')
    parser.add_argument('--use_params', action='store_true', help='Whether to use the params')
    parser.add_argument('--group', type=str, default='base')
    parser.add_argument('--debug', action='store_true', help='Debug Mode')

    parser.add_argument('--da', action='store_true', help='Domain Adaptation')
    parser.add_argument('--transfer', action='store_true', help='Transfer Learning')

    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument('--device', type=int, default=0)

    # Emb Config
    parser.add_argument('--node_text', type=str, nargs='+', default=['title', 'abstract'])
    parser.add_argument('--emb_model', type=str, default='minilm')
    parser.add_argument('--emb_method', type=str, default='TANS')

    # Model Config
    parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'mlp'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--normalize', type=str, default='none')

    # Training Config
    parser.add_argument('--label_setting', type=str, default='ratio', choices=['ratio', 'number'])
    parser.add_argument('--train_ratio', type=float, default=0.0, help='Ratio of training nodes in target dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of training nodes in target dataset')
    parser.add_argument('--train_num', type=int, default=20, help='Number of training nodes per class')
    parser.add_argument('--val_num', type=int, default=30, help='Number of validation nodes per class')

    parser.add_argument('--pt_epochs', type=int, default=500, help='Epochs for pretraining')
    parser.add_argument('--pt_lr', type=float, default=1e-3, help='Learning rate for pretraining')
    parser.add_argument('--pt_decay', type=float, default=0, help='Weight decay for pretraining')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for fine-tuning')
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--decay', type=float, default=1e-5, help='Weight decay for fine-tuning')

    # Others
    parser.add_argument('--feat_noise', type=float, default=0.0)
    parser.add_argument('--edge_noise', type=float, default=0.0)
    parser.add_argument('--eval_disc', action='store_true',
                        help='Whether to evaluate the discrepancy between source and target')

    args = parser.parse_args()
    return vars(args)


def get_transfer_args():
    parser = argparse.ArgumentParser('Evaluation')

    # General Config
    parser.add_argument('--pt_data', type=str, default='cora')
    parser.add_argument('--data', type=str, default='pubmed')
    parser.add_argument('--use_params', action='store_true', help='Whether to use the params')
    parser.add_argument('--group', type=str, default='base')
    parser.add_argument('--debug', action='store_true', help='Debug Mode')

    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument('--device', type=int, default=0)

    # Emb Config
    parser.add_argument('--node_text', type=str, nargs='+', default=['title', 'abstract'])
    parser.add_argument('--emb_model', type=str, default='minilm')
    parser.add_argument('--emb_method', type=str, default='TANS')

    # Model Config
    parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'mlp'])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--normalize', type=str, default='none')

    # Training Config
    parser.add_argument('--label_setting', type=str, default='number', choices=['ratio', 'number'])
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training nodes in target dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of training nodes in target dataset')
    parser.add_argument('--train_num', type=int, default=20, help='Number of training nodes per class')
    parser.add_argument('--val_num', type=int, default=30, help='Number of validation nodes per class')

    parser.add_argument('--pt_epochs', type=int, default=100, help='Epochs for pretraining')
    parser.add_argument('--pt_lr', type=float, default=1e-3, help='Learning rate for pretraining')
    parser.add_argument('--pt_decay', type=float, default=0, help='Weight decay for pretraining')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs for fine-tuning')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for fine-tuning')
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--decay', type=float, default=1e-5, help='Weight decay for fine-tuning')

    # Others
    parser.add_argument('--feat_noise', type=float, default=0.0)
    parser.add_argument('--edge_noise', type=float, default=0.0)
    parser.add_argument('--eval_disc', action='store_true',
                        help='Whether to evaluate the discrepancy between source and target')
    parser.add_argument('--wo_neigh', action='store_true', help='Without using neighbors')

    args = parser.parse_args()
    return vars(args)
