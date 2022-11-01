import numpy as np
import torch
from gin import Encoder as ginEncoder
from torch.nn import Sequential, Linear, ReLU
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# gin adapted graph neural networks
class GINNet(torch.nn.Module):
    def __init__(self, dim, num_gc_layers, num_atoms, dataset_num_features):
        super(GINNet, self).__init__()

        # if dataset.num_features:
        #     num_features = dataset.num_features
        # else:
        #     num_features = 1
        num_features = dataset_num_features
        # dim = 16

        self.encoder = ginEncoder(num_features, dim, num_gc_layers) # This encoder is GIN, from the paper how powerful are graph neural networks

        self.fc1 = Linear(dim*num_gc_layers, dim)
        self.fc2 = Linear(dim, num_atoms)

    def forward(self, x, edge_index, batch):
        num_nodes = edge_index[0][-1] + 1

        if x is None:
            x = torch.ones(num_nodes).to(device)

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x, dim=-1)
        return x[0]

    def get_embeddings(self, dataloader):
        embeddings = []
        ys = []
        dataloader_tmp = dataloader

        with torch.no_grad():
            for data in dataloader_tmp:
                x, edge_index, y= data.x, data.edge_index, data.y
                num_nodes = edge_index[0][-1] + 1

                if x is None:
                    x = torch.ones(num_nodes, 1).to(device)
                else:
                    num_nodes = x.size()[0]
                batch = torch.zeros(num_nodes, dtype=torch.long)

                x, _ = self.encoder(x, edge_index, batch)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.fc2(x)
                # return F.log_softmax(x, dim=-1)
                embeddings.append(x[0].cpu().numpy())
                ys.append(y.cpu().numpy())

        embeddings = np.asarray(embeddings)
        ys = np.concatenate(ys, 0)

        return embeddings, ys