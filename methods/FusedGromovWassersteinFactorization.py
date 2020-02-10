import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from methods.AlgOT import cost_mat, ot_fgw
from methods.DataIO import StructuralDataSampler, structural_data_split
from sklearn.manifold import MDS, TSNE
from typing import List, Tuple


def fgwd(graph1, embedding1, prob1,
         graph2, embedding2, prob2, tran):
    cost = cost_mat(graph1, graph2, prob1, prob2, tran, embedding1, embedding2)
    return (cost * tran).sum()


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
                 prior=None):
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

        # weights of atoms
        self.weights = nn.Parameter(torch.randn(self.num_atoms, self.num_samples))
        self.softmax = nn.Softmax(dim=0)

        # basis and their node distribution
        if prior is None:
            self.ps = []
            self.atoms = nn.ParameterList()
            self.embeddings = nn.ParameterList()
            for k in range(self.num_atoms):
                atom = nn.Parameter(torch.randn(self.size_atoms[k], self.size_atoms[k]))
                embedding = nn.Parameter(torch.randn(self.size_atoms[k], self.dim_embedding) / self.dim_embedding)
                dist = torch.ones(self.size_atoms[k], 1) / self.size_atoms[k]  # .type(torch.FloatTensor)
                self.ps.append(dist)
                self.atoms.append(atom)
                self.embeddings.append(embedding)
        else:
            # num_atoms_per_class = int(self.num_atoms / self.num_classes)
            # counts = np.zeros((self.num_classes,))
            # self.ps = []
            # self.atoms = []
            # self.size_atoms = []
            # self.embeddings = []
            # base_label = []
            # for n in range(prior.__len__()):
            #     data = prior.__getitem__(n)
            #     graph = data[0]
            #     prob = data[1]
            #     emb = data[2]
            #     gt = int(data[3][0])
            #     if counts[gt] < num_atoms_per_class:
            #         self.size_atoms.append(graph.size(0))
            #         atom = nn.Parameter(graph)
            #         embedding = nn.Parameter(emb)
            #         self.ps.append(prob)
            #         self.atoms.append(atom)
            #         self.embeddings.append(embedding)
            #         base_label.append(gt)
            #         counts[gt] += 1

            num_samples = prior.__len__()
            index_samples = list(range(num_samples))
            random.shuffle(index_samples)
            self.ps = []
            self.atoms = nn.ParameterList()
            self.embeddings = nn.ParameterList()
            base_label = []
            for k in range(self.num_atoms):
                idx = index_samples[k]
                data = prior.__getitem__(idx)
                graph = data[0]
                prob = data[1]
                emb = data[2]
                gt = data[3]
                self.size_atoms[k] = graph.size(0)
                atom = nn.Parameter(graph)
                embedding = nn.Parameter(emb)
                self.ps.append(prob)
                self.atoms.append(atom)
                self.embeddings.append(embedding)
                base_label.append(gt[0])

            print(self.size_atoms)
            print(base_label)
        self.sigmoid = nn.Sigmoid()

    def output_weights(self, idx: int = None):
        if idx is not None:
            return self.softmax(self.weights[:, idx])
        else:
            return self.softmax(self.weights)

    def output_atoms(self, idx: int = None):
        if idx is not None:
            return self.sigmoid(self.atoms[idx])
        else:
            return [self.sigmoid(self.atoms[idx]) for idx in range(len(self.atoms))]

    def fgwb(self,
             pb: torch.Tensor,
             trans: List,
             weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        """
        tmp1 = pb @ torch.t(pb)
        tmp2 = pb @ torch.ones(1, self.dim_embedding)
        graph = torch.zeros(pb.size(0), pb.size(0))
        embedding = torch.zeros(pb.size(0), self.dim_embedding)
        for k in range(self.num_atoms):
            graph_k = self.output_atoms(k)
            graph += weights[k] * (torch.t(trans[k]) @ graph_k @ trans[k])
            embedding += weights[k] * (torch.t(trans[k]) @ self.embeddings[k])
        graph = graph / tmp1
        embedding = embedding / tmp2
        return graph, embedding

    def forward(self, graph: torch.Tensor, prob: torch.Tensor, embedding: torch.Tensor,
                index: int, trans: List, tran: torch.Tensor):
        """
        For "n" unknown samples, given their disimilarity/adjacency matrix "cost" and distribution "p", we calculate
        "d_gw(barycenter(atoms, weights), cost)" approximately.

        Args:
            graph: (n, n) matrix (torch.Tensor), representing disimilarity/adjacency matrix
            prob: (n, 1) vector (torch.Tensor), the empirical distribution of the nodes/samples in "graph"
            embedding: (n, d) matrix (torch.Tensor)
            index: the index of the "cost" in the dataset
            trans: a list of (ns, nb) OT matrices
            tran: a (n, nb) OT matrix

        Returns:
            d_gw: the value of loss function
            barycenter: the proposed GW barycenter
            tran0: the optimal transport between barycenter and cost
            trans: the optimal transports between barycenter and atoms
            weights: the weights of atoms
        """
        # variables
        weights = self.softmax(self.weights[:, index])
        graph_b, embedding_b = self.fgwb(prob, trans, weights)
        d_fgw = fgwd(graph, embedding, prob, graph_b, embedding_b, prob, tran)

        return d_fgw, self.weights[:, index], graph_b, embedding_b


def train_usl(model,
              database,
              size_batch: int = 16,
              epochs: int = 10,
              lr: float = 1e-1,
              weight_decay: float = 0,
              shuffle_data: bool = True,
              zeta: float = None,
              mode: str = 'fit',
              visualize_prefix: str = None):
    """
    training a FGWF model
    Args:
        model: a FGWF model
        database: a list of data, each element is a list representing [cost, distriubtion, feature, label]
        size_batch: the size of batch, deciding the frequency of backpropagation
        epochs: the number epochs
        lr: learning rate
        weight_decay: the weight of the l2-norm regularization of parameters
        shuffle_data: whether shuffle data in each epoch
        zeta: the weight of the regularizer enhancing the diversity of atoms
        mode: fit or transform
        visualize_prefix: display learning result after each epoch or not
    """
    if mode == 'fit':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        n = 0
        for param in model.parameters():
            if n > 0:
                param.requires_grad = False
            n += 1

        # only update partial model's parameters
        optimizer = optim.Adam([list(model.parameters())[0]], lr=lr, weight_decay=weight_decay)
    model.train()

    data_sampler = StructuralDataSampler(database)
    num_samples = data_sampler.__len__()
    index_samples = list(range(num_samples))
    index_atoms = list(range(model.num_atoms))

    best_loss = float("Inf")
    best_model = None
    for epoch in range(epochs):
        counts = 0
        t_start = time.time()
        loss_epoch = 0
        loss_total = 0
        d_fgw_total = 0
        reg_total = 0
        optimizer.zero_grad()

        if shuffle_data:
            random.shuffle(index_samples)

        for idx in index_samples:
            data = data_sampler.__getitem__(idx)
            graph = data[0]
            prob = data[1]
            emb = data[2]

            # Envelop Theorem
            # feed-forward computation of barycenter B({Ck}, w) and its transports {Trans_k}
            trans = []
            for k in range(model.num_atoms):
                graph_k = model.output_atoms(k).data
                emb_k = model.embeddings[k].data
                _, tran_k = ot_fgw(graph_k, graph, model.ps[k], prob,
                                   model.ot_method, model.gamma, model.ot_layers,
                                   emb_k, emb)
                trans.append(tran_k)
            tran = torch.diag(prob[:, 0])

            # trans = []
            # graph_b = graph
            # emb_b = emb
            # weights = model.output_weights(idx).data
            # tmp1 = prob @ torch.t(prob)
            # tmp2 = prob @ torch.ones(1, model.dim_embedding)
            # for n in range(model.gwb_layers):
            #     graph_b_tmp = 0
            #     emb_b_tmp = 0
            #     trans = []
            #     for k in range(model.num_atoms):
            #         graph_k = model.output_atoms(k).data
            #         emb_k = model.embeddings[k].data
            #         _, tran_k = ot_fgw(graph_k, graph_b, model.ps[k], prob,
            #                            model.ot_method, model.gamma, model.ot_layers,
            #                            emb_k, emb_b)
            #         trans.append(tran_k)
            #         graph_b_tmp += weights[k] * (torch.t(tran_k) @ graph_k @ tran_k)
            #         emb_b_tmp += weights[k] * (torch.t(tran_k) @ emb_k)
            #     graph_b = graph_b_tmp / tmp1
            #     emb_b = emb_b_tmp / tmp2

            # _, tran = ot_fgw(graph, graph_b, prob, prob,
            #                  model.ot_method, model.gamma, model.ot_layers,
            #                  emb, emb_b)

            d_fgw, _, _, _ = model(graph, prob, emb, idx, trans, tran)
            d_fgw_total += d_fgw
            loss_total += d_fgw

            if zeta is not None and mode == 'fit':
                random.shuffle(index_atoms)
                graph1 = model.output_atoms(index_atoms[0])
                emb1 = model.embeddings[index_atoms[0]]
                p1 = model.ps[index_atoms[0]]

                graph2 = model.output_atoms(index_atoms[1])
                emb2 = model.embeddings[index_atoms[1]]
                p2 = model.ps[index_atoms[1]]

                _, tran12 = ot_fgw(graph1.data, graph2.data, p1, p2,
                                   model.ot_method, model.gamma, model.ot_layers,
                                   emb1.data, emb2.data)
                reg = fgwd(graph1, emb1, p1, graph2, emb2, p2, tran12)

                reg_total += zeta * reg
                loss_total -= zeta * reg

            counts += 1
            if counts % size_batch == 0 or counts == num_samples:
                if counts % size_batch == 0:
                    num = size_batch
                else:
                    num = counts % size_batch
                loss_epoch += loss_total
                loss_total.backward()
                optimizer.step()

                print('-- {}/{} [{:.1f}%], loss={:.4f}, dgw={:.4f}, reg={:.4f}, time={:.2f}s.'.format(
                    counts, num_samples, counts / num_samples * 100.0,
                    loss_total / num, d_fgw_total / num, reg_total / num, time.time() - t_start))

                t_start = time.time()
                loss_total = 0
                d_fgw_total = 0
                reg_total = 0
                optimizer.zero_grad()

        if best_loss > loss_epoch.data / num_samples:
            best_model = copy.deepcopy(model)
            best_loss = loss_epoch.data / num_samples

        print('{}: Epoch {}/{}, loss = {:.4f}, best loss = {:.4f}'.format(
            mode, epoch + 1, epochs, loss_epoch / num_samples, best_loss))

        if visualize_prefix is not None:
            embeddings = tsne_weights(model)
            index = list(range(num_samples))
            labels = []
            for idx in index:
                labels.append(data_sampler.data[idx][-1])
            labels = np.asarray(labels)
            num_classes = int(np.max(labels) + 1)

            plt.figure(figsize=(6, 6))
            for i in range(num_classes):
                plt.scatter(embeddings[labels == i, 0],
                            embeddings[labels == i, 1],
                            s=4,
                            label='class {}'.format(i + 1))
            plt.legend()
            print('{}_usl_tsne_{}_{}.pdf'.format(visualize_prefix, mode, epoch+1))
            plt.savefig('{}_usl_tsne_{}_{}.pdf'.format(visualize_prefix, mode, epoch+1))
            plt.close()
    return best_model


def train_ssl(model,
              database: list,
              size_batch: int = 16,
              epochs: int = 10,
              lr: float = 1e-1,
              weight_decay: float = 0,
              shuffle_data: bool = True,
              zeta: float = None,
              mode: str = 'fit',
              ssl: float = 0.1,
              visualize_prefix: str = None):
    """
    training a FGWF model
    Args:
        model: a FGWF model
        database: a list of data, each element is a list representing [cost, distriubtion, feature, label]
        size_batch: the size of batch, deciding the frequency of backpropagation
        epochs: the number epochs
        lr: learning rate
        weight_decay: the weight of the l2-norm regularization of parameters
        shuffle_data: whether shuffle data in each epoch
        zeta: the weight of the regularizer enhancing the diversity of atoms
        mode: fit or transform
        ssl: the percentage of labeled samples
        visualize_prefix: display learning result after each epoch or not
    """
    c = ['blue', 'orange', 'red', 'green', 'yellow', 'grey']
    train_graphs, test_graphs, train_labels, test_labels = structural_data_split(database, split_rate=ssl)
    labels = train_labels + test_labels
    labels = np.asarray(labels)
    num_classes = int(np.max(labels) + 1)

    predictor = nn.Linear(model.num_atoms, num_classes)
    criterion = nn.CrossEntropyLoss()

    if mode == 'fit':
        optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()),
                               lr=lr, weight_decay=weight_decay)
    else:
        n = 0
        for param in model.parameters():
            if n > 0:
                param.requires_grad = False
            n += 1

        # only update partial model's parameters
        optimizer = optim.Adam([list(model.parameters())[0]] + list(predictor.parameters()),
                               lr=lr, weight_decay=weight_decay)

    train_sampler = StructuralDataSampler(train_graphs)
    test_sampler = StructuralDataSampler(test_graphs)

    num_samples = len(database)
    index_samples = list(range(num_samples))
    index_atoms = list(range(model.num_atoms))

    best_acc = 0
    best_model = None
    best_predictor = None
    for epoch in range(epochs):
        model.train()
        predictor.train()

        counts = 0
        t_start = time.time()
        loss_epoch = 0
        loss_total = 0
        d_fgw_total = 0
        reg_total = 0
        lle_total = 0
        optimizer.zero_grad()

        if shuffle_data:
            random.shuffle(index_samples)

        for idx in index_samples:
            if idx < len(train_graphs):
                data = train_sampler.__getitem__(idx)
            else:
                data = test_sampler.__getitem__(idx - len(train_graphs))
            graph = data[0]
            prob = data[1]
            emb = data[2]
            label = data[3]

            # Envelop Theorem
            # feed-forward computation of barycenter B({Ck}, w) and its transports {Trans_k}
            trans = []
            for k in range(model.num_atoms):
                graph_k = model.output_atoms(k).data
                emb_k = model.embeddings[k].data
                _, tran_k = ot_fgw(graph_k, graph, model.ps[k], prob,
                                   model.ot_method, model.gamma, model.ot_layers,
                                   emb_k, emb)
                trans.append(tran_k)
            tran = torch.diag(prob[:, 0])

            d_fgw, weights, _, _ = model(graph, prob, emb, idx, trans, tran)
            d_fgw_total += d_fgw
            loss_total += d_fgw

            if zeta is not None and mode == 'fit':
                random.shuffle(index_atoms)
                graph1 = model.output_atoms(index_atoms[0])
                emb1 = model.embeddings[index_atoms[0]]
                p1 = model.ps[index_atoms[0]]

                graph2 = model.output_atoms(index_atoms[1])
                emb2 = model.embeddings[index_atoms[1]]
                p2 = model.ps[index_atoms[1]]

                _, tran12 = ot_fgw(graph1.data, graph2.data, p1, p2,
                                   model.ot_method, model.gamma, model.ot_layers,
                                   emb1.data, emb2.data)
                reg = fgwd(graph1, emb1, p1, graph2, emb2, p2, tran12)

                reg_total += zeta * reg
                loss_total -= zeta * reg

            if idx < len(train_graphs):
                lle = criterion(predictor(weights.unsqueeze(0)), label)
                lle_total += 0.02 * lle
                loss_total += 0.02 * lle

            counts += 1
            if counts % size_batch == 0 or counts == num_samples:
                if counts % size_batch == 0:
                    num = size_batch
                else:
                    num = counts % size_batch
                loss_epoch += loss_total
                loss_total.backward()
                optimizer.step()

                print('-- {}/{} [{:.1f}%], loss={:.4f}, dgw={:.4f}, ssl={:.4f}, reg={:.4f}, time={:.2f}s.'.format(
                    counts, num_samples, counts / num_samples * 100.0,
                    loss_total / num, d_fgw_total / num, lle_total / num, reg_total / num, time.time() - t_start))

                t_start = time.time()
                loss_total = 0
                d_fgw_total = 0
                reg_total = 0
                lle_total = 0
                optimizer.zero_grad()

        # validation
        valid_acc = 0
        predictor.eval()
        model.eval()
        for i in range(len(test_graphs)):
            data = test_sampler.__getitem__(i)
            weights = model.weights[:, i + len(train_graphs)]  # output_weights(i + len(train_graphs))
            pred = predictor(weights.unsqueeze(0))
            _, est = torch.max(pred, 1)
            if est == data[3].data:
                valid_acc += 1

        valid_acc /= len(test_graphs)
        if best_acc <= valid_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(model)
            best_predictor = copy.deepcopy(predictor)

        print('{}: Epoch {}/{}, loss = {:.4f}, best acc = {:.4f}'.format(
            mode, epoch + 1, epochs, loss_epoch / num_samples, best_acc))

        if visualize_prefix is not None:
            embeddings = tsne_weights(model)
            embeddings_train = embeddings[:len(train_graphs), :]
            embeddings_valid = embeddings[len(train_graphs):, :]
            labels_train = labels[:len(train_graphs)]
            labels_valid = labels[len(train_graphs):]
            plt.figure(figsize=(6, 6))
            for i in range(num_classes):
                plt.scatter(embeddings_train[np.asarray(labels_train) == i, 0],
                            embeddings_train[np.asarray(labels_train) == i, 1],
                            s=4,
                            c=c[i],
                            label='train class {}'.format(i + 1))
                plt.scatter(embeddings_valid[np.asarray(labels_valid) == i, 0],
                            embeddings_valid[np.asarray(labels_valid) == i, 1],
                            c=c[i],
                            label='test class {}'.format(i + 1))
            plt.legend()
            plt.title('best acc = {:.4f}'.format(best_acc))
            print('{}_ssl_tsne_{}_{}_{}.png'.format(visualize_prefix, len(train_graphs), mode, epoch+1))
            plt.savefig('{}_ssl_tsne_{}_{}_{}.png'.format(visualize_prefix, len(train_graphs), mode, epoch+1))
            plt.close()
    return best_model, best_predictor, best_acc


def tsne_weights(model) -> np.ndarray:
    """
    Learn the 2D embeddings of the weights associated with atoms via t-SNE
    Returns:
        embeddings: (num_samples, 2) matrix representing the embeddings of weights
    """
    model.eval()
    features = model.weights.cpu().data.numpy()
    features = features.T
    if features.shape[1] == 2:
        embeddings = features
    else:
        embeddings = TSNE(n_components=2).fit_transform(features)
    return embeddings


def clustering(model) -> np.ndarray:
    """
    Taking the atoms as clustering centers, we cluster data based on their weights associated with the atoms
    """
    model.eval()
    feature = model.output_weights().data.numpy()
    return np.argmax(feature, axis=0)


def save_model(model, full_path):
    """
    Save trained model
    Args:
        model: the target model
        full_path: the path of directory
    """
    torch.save(model.state_dict(), full_path)


def load_model(model, full_path):
    """
    Load pre-trained model
    Args:
        model: the target model
        full_path: the path of directory
    """
    model.load_state_dict(torch.load(full_path))
    return model


def visualize_atoms(model, idx: int, threshold: float = 0.5, filename: str = None):
    """
    Learning the 2D embeddings of the atoms via multi-dimensional scaling (MDS)
    Args:
        model: a FGWF model
        idx: an index of the atoms
        threshold: the threshold of edge
        filename: the prefix of image name

    Returns:
        embeddings: (size_atom, 2) matrix representing the embeddings of nodes/samples corresponding to the atom.
    """
    graph = model.output_atoms(idx).cpu().data.numpy()
    emb = model.embeddings[idx].cpu().data.numpy()

    if emb.shape[1] == 1:
        cost = graph + graph.T
        emb = MDS(n_components=2, dissimilarity='precomputed').fit_transform(cost)
    elif emb.shape[1] > 2:
        emb = TSNE(n_components=2).fit_transform(emb)

    graph[graph >= threshold] = 1
    graph[graph < threshold] = 0

    plt.figure(figsize=(6, 6))
    plt.scatter(emb[:, 0], emb[:, 1], s=80)
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i, j] > 0 and i != j:
                pair = np.asarray([i, j])
                x = emb[pair, 0]
                y = emb[pair, 1]
                plt.plot(x, y, 'r-')

    if filename is None:
        plt.savefig('atom_{}.pdf'.format(idx))
    else:
        plt.savefig('{}_{}.pdf'.format(filename, idx))
