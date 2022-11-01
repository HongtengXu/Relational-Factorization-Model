import matplotlib
matplotlib.use('Agg')
import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
# from core.encoders import *

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import ToDense

import sys
import json
from torch import optim
from torch.nn import Sequential, Linear, ReLU
from torch.distributions import Categorical

from evaluate_embedding import evaluate_embedding

from arguments import arg_parse
import math
import warnings
from typing import List, Tuple
import random
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import inspect

from AlgOT import cost_mat, ot_fgw
from diffpool.diffpool_Encoder import SoftPoolingGcnEncoder
from diffpool.diffpool_args import diffpool_arg_parse
from diffpool import diffpool_load_data
from diffpool.diffpool_graph_sampler import diffpool_GraphSampler
import diffpool.gen.feat as featgen

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class FGWF(nn.Module):
    """
    A simple PyTorch implementation of Fused Gromov-Wasserstein factorization model
    The feed-forward process imitates the proximal point algorithm or bregman admm
    """
    def __init__(self,
                 num_samples: int,
                 num_classes: int,
                 size_atoms: List,
                 dim_embedding: int = 1,
                 ot_method: str = 'ppa',
                 gamma: float = 1e-1,
                 gwb_layers: int = 5,
                 ot_layers: int = 5,
                 prior=None,
                 gnn_weights: bool = True,
                 gcn_hidden_dim: int = 16,
                 num_gc_layers: int = 5,
                 diffpool_args=None
                 ):
        """
        Args:
            num_samples: the number of samples
            size_atoms: a list, its length is the number of atoms, each element is the size of the corresponding atom
            dim_embedding: the dimension of embedding
            ot_method: ppa or b-admm
            gamma: the weight of Bregman divergence term
            gwb_layers: the number of gwb layers in each gwf module
            ot_layers: the number of ot layers in each gwb module
        """
        super(FGWF, self).__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.size_atoms = size_atoms
        self.num_atoms = len(self.size_atoms)
        self.dim_embedding = dim_embedding
        self.ot_method = ot_method
        self.gwb_layers = gwb_layers
        self.ot_layers = ot_layers
        self.gamma = gamma
        self.gnn_weights = gnn_weights

        # weights of atoms
        self.weights = nn.Parameter(torch.randn(self.num_atoms, self.num_samples))
        self.softmax = nn.Softmax(dim=0)

        self.ps = []
        self.atoms = nn.ParameterList()
        self.embeddings = nn.ParameterList()

        # basis and their node distribution
        if prior is None:
            for k in range(self.num_atoms):
                atom = nn.Parameter(torch.randn(self.size_atoms[k], self.size_atoms[k]))
                embedding = nn.Parameter(torch.randn(self.size_atoms[k], self.dim_embedding) / self.dim_embedding)
                dist = torch.ones(self.size_atoms[k], 1) / self.size_atoms[k]  # .type(torch.FloatTensor)
                self.ps.append(dist)
                self.atoms.append(atom)
                self.embeddings.append(embedding)

        else:
            if self.gnn_weights:
                # self.gnn = GINNet(gcn_hidden_dim, num_gc_layers, self.num_atoms, dataset_num_features)

                self.gnn = SoftPoolingGcnEncoder(
                    max_num_nodes=max_num_nodes, input_dim=dataset_num_features, hidden_dim=diffpool_args.hidden_dim, 
                    embedding_dim=diffpool_args.output_dim, label_dim=self.num_atoms, num_layers=diffpool_args.num_gc_layers,
                    assign_hidden_dim=diffpool_args.hidden_dim, assign_ratio=diffpool_args.assign_ratio, num_pooling=diffpool_args.num_pool,
                    bn=diffpool_args.bn, dropout=diffpool_args.dropout, linkpred=diffpool_args.linkpred, args=diffpool_args,
                    assign_input_dim=assign_input_dim).to(device)
                print("gnn initialization is working")


            num_samples = prior.__len__()
            index_samples = list(range(num_samples))
            random.shuffle(index_samples)

            base_label = []
            for k in range(self.num_atoms):
                idx = index_samples[k]
                data = prior.__getitem__(idx)
                data.edge_attr = None # set the edge attribute to none. otherwise, it will mess up with get_adjacency()
                graph = get_adjacency(data)['adj']
                graph = graph*0.9999 + 0.00005
                graph = torch.log(graph/(1-graph))   # transform the graph using inverse of sigmoid
                num_nodes = graph.size(0)
                prob = torch.ones(num_nodes, 1)/num_nodes
                if data.x is None:
                    emb = torch.ones((num_nodes, 1))
                else:
                    emb = data.x
                gt = data['y']
                self.size_atoms[k] = num_nodes
                atom = nn.Parameter(graph)
                embedding = nn.Parameter(emb)
                # atom = graph
                # embedding = emb
                self.ps.append(prob)
                self.atoms.append(atom)
                self.embeddings.append(embedding)
                base_label.append(gt[0])

            print("size of atoms are:", self.size_atoms)
            print("labels of selected graphs are", base_label)
        # self.sigmoid = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def output_atoms(self, idx: int = None):
        if idx is not None:
            return self.sigmoid(self.atoms[idx])
        else:
            return [self.sigmoid(self.atoms[idx]) for idx in range(len(self.atoms))]

    def fgwTrans(self, 
                graph: torch.Tensor, 
                prob: torch.Tensor, 
                emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trans = []
        for k in range(self.num_atoms):
            graph_k = self.output_atoms(k).data # .data only gets you the tensor, requires_grad for graph_k becomes False
            emb_k = self.embeddings[k].data

            _, tran_k = ot_fgw(graph_k, graph, self.ps[k], prob,
                               self.ot_method, self.gamma, self.ot_layers,
                               emb_k, emb)
            trans.append(tran_k)
        # tran = torch.diag(prob[:, 0])
        tran = torch.diag(prob[:, 0]).to(device)

        return trans, tran

    def fgwBarycenter(self,
             pb: torch.Tensor,
             trans: List,
             idx: int,
             masked_emb: torch.Tensor,
             masked_graph: torch.Tensor,
             batch_num_nodes: int,
             masked_assign_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve GW Barycetner problem
        barycenter = argmin_{B} sum_k w[k] * d_gw(atom[k], B) via proximal point-based alternating optimization:

        step 1: Given current barycenter, for k = 1:K, we calculate trans[k] by the OT-PPA layer.
        step 2: Given new trans, we update barycenter by
            barycenter = sum_k trans[k] * atom[k] * trans[k]^T / (pb * pb^T)

        Args:
            pb: (nb, 1) vector (torch tensor), the empirical distribution of the nodes/samples of the barycenter
            trans: a dictionary {key: index of atoms, value: the (ns, nb) initial optimal transport}
            weights: (K,) vector (torch tensor), representing the weights of the atoms

        Returns:
            barycenter: (nb, nb) matrix (torch tensor) representing the updated GW barycenter
            weights: coefficients in the barycenter problem
        """
        batch_num_nodes = np.asarray([batch_num_nodes])

        if self.gnn_weights == True:

            """ to be deleted
            x = dataset[idx]['x']
            if x is None:
                num_nodes = edge_index[0][-1] + 1
            else:
                num_nodes = x.size()[0]
            
            batch = torch.zeros(num_nodes, dtype=torch.long) # we are taking one sample at a time
            """

            # weights = self.gnn(idx_embedding, idx_edge_index, batch)
            # masked_graph, masked_emb, masked_assign_input, batch_num_nodes
            weights = self.gnn(masked_emb, masked_graph, batch_num_nodes, masked_assign_input)
            weights = F.softmax(weights)
        else:
            weights = self.softmax(self.weights[:, idx])


        # what do I need? pb, trans
        tmp1 = pb @ torch.t(pb)
        # tmp2 = pb @ torch.ones(1, self.dim_embedding)
        tmp_ones = torch.ones(1, self.dim_embedding).to(device)
        tmp2 = pb @ tmp_ones
        graph = torch.zeros(pb.size(0), pb.size(0)).to(device)
        embedding = torch.zeros(pb.size(0), self.dim_embedding).to(device)
        for k in range(self.num_atoms):
            graph_k = self.output_atoms(k)
            graph += weights[k] * (torch.t(trans[k]) @ graph_k @ trans[k])
            embedding += weights[k] * (torch.t(trans[k]) @ self.embeddings[k])
        graph = graph / tmp1
        embedding = embedding / tmp2

        return graph, embedding, weights

    def forward(self, graph: torch.Tensor, embedding: torch.Tensor, prob: torch.Tensor,
                graph_b: torch.Tensor, embedding_b:torch.Tensor, prob_b: torch.Tensor, 
                tran:torch.Tensor, ole_coeff: float, idx: int, weights: torch.Tensor):
        """
        Args:
            graph: (n, n) matrix (torch.Tensor), representing disimilarity/adjacency matrix
            embedding: (n, d) matrix (torch.Tensor)
            prob: (n, 1) vector (torch.Tensor), the empirical distribution of the nodes/samples in "graph"
            graph_b: Barycenter, (n, n) matrix (torch.Tensor), representing disimilarity/adjacency matrix
            embedding_b: Barycenter, (n, d) matrix (torch.Tensor)
            prob_b: Barycenter, (n, 1) vector (torch.Tensor), the empirical distribution of the nodes/samples in "graph"
            tran: a (n, nb) OT matrix
            ole_coeff: coefficient for the ole loss
            idx: index of the graph

        Returns:
            d_gw: the value of loss function
        """

        cluster_weights = torch.zeros(2)
        mid_index = len(weights)
        cluster_weights[0] = torch.sum(weights[:mid_index])
        cluster_weights[1] = torch.sum(weights[mid_index:])
        cluster_loss = Categorical(probs=cluster_weights).entropy()
        entropy_loss = Categorical(probs=weights).entropy()
        ole_loss = cluster_loss - entropy_loss

        cost = cost_mat(graph, graph_b, prob, prob_b, tran, embedding, embedding_b)
        return (cost * tran).sum()

    def get_embeddings(self, dataloader):
        if self.gnn_weights: # if gnn induced weights are enabled
            return self.gnn.get_embeddings(dataloader)
        else:
            embeddings = []
            ys = []
            for idx, data in enumerate(dataset):
                y= data.y
                embeddings.append(self.weights[:, idx].cpu().detach().numpy())
                ys.append(y.cpu().numpy())

            embeddings = np.asarray(embeddings)
            ys = np.concatenate(ys, 0)
            return embeddings, ys

def tsne_plots(embeddings, ys, num_classes, visualize_prefix, iter_num):
    tsne_weights = TSNE(n_components=num_classes).fit_transform(embeddings)
    # plt.figure()
    plt.figure(figsize=(6, 6))
    for i in range(num_classes):
        plt.scatter(tsne_weights[ys == i, 0],
                    tsne_weights[ys == i, 1],
                    # s=25,
                    s=4,
                    label='class {}'.format(i+1))
    # plt.legend(prop={'size': 14})
    plt.legend()
    plt.savefig('{}tsne_iter_{}.pdf'.format(visualize_prefix, iter_num), bbox_inches='tight')
    # plt.show()
    plt.close()

def svd_plots(embeddings, ys, num_classes, visualize_prefix, iter_num):
    tsne_weights = TruncatedSVD(n_components=num_classes).fit_transform(embeddings)
    # plt.figure()
    plt.figure(figsize=(6, 6))
    for i in range(num_classes):
        plt.scatter(tsne_weights[ys == i, 0],
                    tsne_weights[ys == i, 1],
                    # s=25,
                    s=4,
                    label='class {}'.format(i+1))
    # plt.legend(prop={'size': 14})
    plt.legend()
    plt.savefig('{}svd_iter_{}.pdf'.format(visualize_prefix, iter_num), bbox_inches='tight')
    # plt.show()
    plt.close()



if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        args = arg_parse()
        accuracies = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}
        epochs = args.epochs
        log_interval = 1
        dataloader_batch_size = 1
        lr = args.lr
        lr_atoms = args.lr_atoms
        DS = args.DS
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        dataset = TUDataset(path, name=DS).shuffle()

        # try:
        #     dataset_num_features = dataset.num_features
        # except:
        #     dataset_num_features = 1
        if (dataset.num_features == 0):
            dataset_num_features = 1
        else:
            dataset_num_features = dataset.num_features

        # dataloader = DataLoader(dataset, batch_size=dataloader_batch_size)
        get_adjacency = ToDense() # this is a class which takes edge_index as input and output adjacency matrix

        size_atoms = args.num_atoms * [20]
        ot_method = 'ppa' # or 'b-admm'
        gamma = 1e-1
        gwb_layers = 5
        ot_layers = 50

        diffpool_args = diffpool_arg_parse()
        max_num_nodes = diffpool_args.max_nodes # setup for diffpoolEncoder, maybe also needed for dataloader


        ######################################################################################
        # diffpool data loading
        graphs = diffpool_load_data.read_graphfile(datadir=diffpool_args.datadir, dataname=args.DS, max_nodes=max_num_nodes)

        feat = 'node-label'
    
        if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
            print('Using node features')
            input_dim = graphs[0].graph['feat_dim']
        elif feat == 'node-label' and 'label' in graphs[0].nodes[0]:
            print('Using node labels')
            for G in graphs:
                for u in G.nodes():
                    G.nodes[u]['feat'] = np.array(G.nodes[u]['label'])
        else:
            print('Using constant labels')
            # featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
            featgen_const = featgen.ConstFeatureGen(np.ones(dataset_num_features, dtype=float))
            for G in graphs:
                featgen_const.gen_node_features(G)

        # minibatch
        dataset_sampler = diffpool_GraphSampler(graphs, normalize=False, max_num_nodes=max_num_nodes,
                features=diffpool_args.feature_type)

        # assign_input_dim = len(graphs[0].nodes[0]['label']) # this should be defined and returned by the sampler
        assign_input_dim = dataset_sampler.assign_feat_dim
        print("successfully load diffpool input data")

        dataloader = torch.utils.data.DataLoader(
                dataset_sampler, 
                batch_size=dataloader_batch_size, 
                shuffle=True,
                num_workers=diffpool_args.num_workers)

        ######################################################################################


        model = FGWF(num_samples=len(dataset),
                    num_classes=args.num_classes,
                    size_atoms=size_atoms,
                    dim_embedding=dataset_num_features,
                    ot_method=ot_method,
                    gamma=gamma,
                    gwb_layers=gwb_layers,
                    ot_layers=ot_layers,
                    prior=dataset,
                    gnn_weights=args.gnn_weights, # if set true, weights of atoms are the output of a GNN net
                    gcn_hidden_dim=args.gcn_hidden_dim,
                    num_gc_layers=args.num_gc_layers,
                    diffpool_args=diffpool_args)

        if args.gnn_weights:
            optimizer = torch.optim.Adam(model.gnn.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam([model.weights], lr=lr)

        atoms_parameters = list(model.atoms.parameters()) + list(model.embeddings.parameters())
        optimizer_atoms = torch.optim.Adam(atoms_parameters, lr=lr_atoms)

        print(optimizer)
        print(optimizer_atoms)


        print('================')
        print('lr: {}'.format(lr))
        print('num_features: {}'.format(dataset_num_features))
        print('gcn_hidden_dim: {}'.format(args.gcn_hidden_dim))
        print('num_gc_layers: {}'.format(args.num_gc_layers))
        print('================')

        
        model.eval()
        emb, y = model.get_embeddings(dataloader) # get embeddings of all graphs
        Path(args.plots_directory).mkdir(parents=True, exist_ok=True) # create a directory if it does not exists
        tsne_plots(emb, y, num_classes=args.num_classes, visualize_prefix=args.plots_directory, iter_num=0)
        print("embeddings of training data are:")
        # print(emb)
        svd_plots(emb, y, num_classes=args.num_classes, visualize_prefix=args.plots_directory, iter_num=0)

        if (args.eval_embed):
            res = evaluate_embedding(emb, y)
            print("This is iteration {}".format(0))
            accuracies['logreg'].append(res[0])
            accuracies['svc'].append(res[1])
            accuracies['linearsvc'].append(res[2])
            accuracies['randomforest'].append(res[3])

        kmeans = KMeans(init='k-means++', n_clusters=args.num_classes, n_init=10)
        pred = kmeans.fit_predict(emb)
        labels = y
        acc = max([1 - np.sum(np.abs(pred - labels)) / len(dataset_sampler),
                           1 - np.sum(np.abs((1 - pred) - labels)) / len(dataset_sampler)])
        print("This epoch's accuracy is", acc)

        index_samples = list(range(len(dataset_sampler)))
        mask_nodes = True # this is set true as default

        for epoch in range(1, epochs+1):
            print("This is iteration {}".format(epoch))
            loss_all = 0
            loss_batch = 0
            count = 0

            model.train()
            random.shuffle(index_samples)
            optimizer.zero_grad()

            for idx in tqdm(index_samples):
                count += 1
                data = dataset_sampler[idx]

                # read in one graph at a time
                masked_graph = torch.tensor(data['adj'], dtype=torch.float32).to(device)
                batch_num_nodes = data['num_nodes'] if mask_nodes else None
                prob = torch.ones(batch_num_nodes, 1, dtype=torch.float32)/batch_num_nodes
                masked_emb = torch.tensor(data['feats'], dtype=torch.float32).to(device)
                masked_assign_input = torch.tensor(data['assign_feats'], dtype=torch.float32).to(device)

                graph = masked_graph
                emb = masked_emb
                graph = graph[:batch_num_nodes, :batch_num_nodes]
                emb = emb[:batch_num_nodes]

                masked_graph = torch.unsqueeze(masked_graph, dim=0)
                masked_emb = torch.unsqueeze(masked_emb, dim=0)
                masked_assign_input = torch.unsqueeze(masked_assign_input, dim=0)

                # available variables for use:
                # masked_graph, masked_emb, masked_assign_input, batch_num_nodes
                # graph, emb, prob

                # Step 1: get the GW transition matrices with all atoms
                trans, tran = model.fgwTrans(graph, prob, emb)

                # Step 2: get the barycenter
                graph_b, embedding_b, weights_k = model.fgwBarycenter(prob, trans, idx, masked_emb, masked_graph, batch_num_nodes, masked_assign_input)

                # Step 3: calculate the discrepancy between input graph and barycenter
                ole_coeff = 0 # weight entropy is the coefficient of the ole loss from Qiang Qiu's paper
                d_fgw = model(graph, emb, prob, graph_b, embedding_b, prob, tran, ole_coeff, idx, weights_k)

                loss_batch += d_fgw
                loss_all += d_fgw.detach().numpy()

                if (count % (2*args.batch_size) == 0 or count == len(dataset)):
                    sample_size = (count -1) % args.batch_size + 1
                    loss_batch /= sample_size
                    loss_batch.backward()
                    optimizer.step()
                    loss_batch = 0
                    optimizer.zero_grad()

                elif (count % args.batch_size == 0):
                    sample_size = (count -1) % args.batch_size + 1
                    loss_batch /= sample_size
                    loss_batch.backward()
                    optimizer_atoms.step()
                    loss_batch = 0
                    optimizer_atoms.zero_grad()


            print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

            if epoch % log_interval == 0:
                model.eval()
                emb, y = model.get_embeddings(dataloader) # get embeddings of all graphs
                Path(args.plots_directory).mkdir(parents=True, exist_ok=True) # create a directory if it does not exists
                tsne_plots(emb, y, num_classes=args.num_classes, visualize_prefix=args.plots_directory, iter_num=epoch)
                print("embeddings of training data are:")
                # print(emb)
                print("graphs of the atoms are:")
                # for atom in model.atoms:
                    # print(atom)
                print("embeddings of the atoms are:")
                # for embedding in model.embeddings:
                    # print(embedding)
                svd_plots(emb, y, num_classes=args.num_classes, visualize_prefix=args.plots_directory, iter_num=epoch)

                if (args.eval_embed):
                    res = evaluate_embedding(emb, y)
                    accuracies['logreg'].append(res[0])
                    accuracies['svc'].append(res[1])
                    accuracies['linearsvc'].append(res[2])
                    accuracies['randomforest'].append(res[3])
                    print(accuracies)

                pred = kmeans.fit_predict(emb)
                labels = y
                acc = max([1 - np.sum(np.abs(pred - labels)) / len(dataset_sampler),
                                   1 - np.sum(np.abs((1 - pred) - labels)) / len(dataset_sampler)])
                print("This epoch's accuracy is", acc)

        with open(args.plots_directory + 'new_log', 'a+') as f:
            s = json.dumps(accuracies)
            f.write('dataset: {}, num_gc_layers: {}, gcn_hidden_dim: {}, num_atoms: {}, batch_size: {}, epochs: {}, log_interval: {}, lr: {}, lr_atoms: {}, {}\n'.format(args.DS, args.num_gc_layers, args.gcn_hidden_dim, args.num_atoms, args.batch_size, epochs, log_interval, lr, lr_atoms, s))
