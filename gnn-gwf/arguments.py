import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--num_atoms', type=int, default=10,
            help='number of atoms in the GWF framework')
    parser.add_argument('--num_classes', type=int, default=2,
            help='number of classes for the final prediction')
    parser.add_argument('--lr', type=float, help='Learning rate of gnn or atoms weights.')
    parser.add_argument('--lr_atoms', type=float, help='Learning rate of atoms graph and embeddings.')
    parser.add_argument('--num_gc_layers', type=int, default=5,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--gcn_hidden_dim', type=int, default=16, help='')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=250,
            help='number of samples to collect the loss before backpropagation')
    parser.add_argument('--gnn_weights', type=int, default=0,
            help='1 implies that the weights of atoms are gnn induced; 0 implies not induced')
    parser.add_argument('--eval_embed', type=int, default=0,
            help='1 implies that we evaluate the embedding; 0 implies we skip the evaluation')
    parser.add_argument('--plots_directory', type=str, default='test/',
            help='True implies that the weights of atoms are gnn induced')

    return parser.parse_args()

