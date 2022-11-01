import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import sys

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from evaluate_embedding import evaluate_embedding
import warnings
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        # self.lns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)
            # ln = torch.nn.LayerNorm(dim)

            self.convs.append(conv)
            self.bns.append(bn)
            # self.lns.append(ln)


    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            # x = self.lns[i](x)
            xs.append(x)
            # if i == 2:
                # feature_map = x2

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):

        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        if (dataset.num_features == 0):
            num_features = 1
        else:
            num_features = dataset.num_features

        dim = 32

        self.encoder = Encoder(num_features, dim, num_gc_layers)

        self.fc1 = Linear(dim*5, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

def train(epoch):
    model.train()

    if (epoch == 1):
        model.eval()
        train_emb, train_y = model.encoder.get_embeddings(train_loader)
        test_emb, test_y = model.encoder.get_embeddings(test_loader)

        emb = np.concatenate((train_emb, test_emb), 0)
        y = np.concatenate((train_y, test_y), 0)
        tsne_plots(emb, y, num_classes=num_class, visualize_prefix=plots_directory, iter_num=0)
        svd_plots(emb, y, num_classes=num_class, visualize_prefix=plots_directory, iter_num=0)

        res = evaluate_embedding(emb, y)
        print("This is iteration {}".format(0))
        log_accuracies['logreg'].append(res[0])
        log_accuracies['svc'].append(res[1])
        log_accuracies['linearsvc'].append(res[2])
        log_accuracies['randomforest'].append(res[3])

        model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # print(data.x.shape)
        # [ num_nodes x num_node_labels ]
        # print(data.edge_index.shape)
        #  [2 x num_edges ]
        # print(data.batch.shape)
        # [ num_nodes ]
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    model.eval()
    train_emb, train_y = model.encoder.get_embeddings(train_loader)
    test_emb, test_y = model.encoder.get_embeddings(test_loader)

    emb = np.concatenate((train_emb, test_emb), 0)
    y = np.concatenate((train_y, test_y), 0)
    tsne_plots(emb, y, num_classes=num_class, visualize_prefix=plots_directory, iter_num=epoch)
    svd_plots(emb, y, num_classes=num_class, visualize_prefix=plots_directory, iter_num=epoch)

    if epoch % 25 == 0:
        res = evaluate_embedding(emb, y)
        print("This is iteration {}".format(epoch))
        log_accuracies['logreg'].append(res[0])
        log_accuracies['svc'].append(res[1])
        log_accuracies['linearsvc'].append(res[2])
        log_accuracies['randomforest'].append(res[3])

    model.train()




    return loss_all / len(train_dataset)

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def tsne_plots(embeddings, ys, num_classes, visualize_prefix, iter_num):
    tsne_weights = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(5, 5))
    for i in range(num_classes):
        plt.scatter(tsne_weights[ys == i, 0],
                    tsne_weights[ys == i, 1],
                    s=4,
                    label='class {}'.format(i+1))
    plt.legend()
    plt.savefig('{}tsne_iter_{}.jpg'.format(visualize_prefix, iter_num))
    # plt.show()
    plt.close()

def svd_plots(embeddings, ys, num_classes, visualize_prefix, iter_num):
    tsne_weights = TruncatedSVD(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(5, 5))
    for i in range(num_classes):
        plt.scatter(tsne_weights[ys == i, 0],
                    tsne_weights[ys == i, 1],
                    s=4,
                    label='class {}'.format(i+1))
    plt.legend()
    plt.savefig('{}svd_iter_{}.jpg'.format(visualize_prefix, iter_num))
    # plt.show()
    plt.close()

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        num_gc_layers = 5
        plots_directory = "results/gin_study/layer_norm/PTC_MR/"
        num_class = 3
        log_accuracies = {}
        log_accuracies['logreg'] = []
        log_accuracies['svc'] = []
        log_accuracies['linearsvc'] = []
        log_accuracies['randomforest'] = []

        for percentage in [ 1.]:
            for DS in [sys.argv[1]]:
                if 'REDDIT' in DS:
                    epochs = 200
                else:
                    epochs = 100
                path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
                accuracies = [[] for i in range(epochs)]
                #kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
                dataset = TUDataset(path, name=DS) #.shuffle()
                num_graphs = len(dataset)
                print('Number of graphs', len(dataset))
                dataset = dataset[:int(num_graphs * percentage)]
                dataset = dataset.shuffle()

                kf = KFold(n_splits=10, shuffle=True, random_state=None)
                for train_index, test_index in kf.split(dataset):

                    # x_train, x_test = x[train_index], x[test_index]
                    # y_train, y_test = y[train_index], y[test_index]
                    train_dataset = [dataset[int(i)] for i in list(train_index)]
                    test_dataset = [dataset[int(i)] for i in list(test_index)]
                    print('len(train_dataset)', len(train_dataset))
                    print('len(test_dataset)', len(test_dataset))

                    train_loader = DataLoader(train_dataset, batch_size=128)
                    test_loader = DataLoader(test_dataset, batch_size=128)
                    # print('train', len(train_loader))
                    # print('test', len(test_loader))

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = Net().to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                    for epoch in range(1, epochs+1):
                        train_loss = train(epoch)
                        train_acc = test(train_loader)
                        test_acc = test(test_loader)
                        accuracies[epoch-1].append(test_acc)
                        tqdm.write('Epoch: {:03d}, Train Loss: {:.7f}, '
                              'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                                           train_acc, test_acc))

                    with open(plots_directory + 'new_log', 'a+') as f:
                        s = json.dumps(log_accuracies)
                        f.write('{},{},{},{}\n'.format(DS, num_gc_layers, epochs, s))
                    sys.exit()
                tmp = np.mean(accuracies, axis=1)
                print(percentage, DS, np.argmax(tmp), np.max(tmp), np.std(accuracies[np.argmax(tmp)]))
                input()
