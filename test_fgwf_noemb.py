import dev.util as util
import methods.FusedGromovWassersteinFactorization as FGWF
import numpy as np
import os
import pickle

from methods.DataIO import StructuralDataSampler, structural_data_list
from sklearn.cluster import KMeans


# data settings
names = ['IMDB-MULTI']
names = ['IMDB-BINARY']


# model params
num_atoms = 10
size_atoms = num_atoms * [20]
ot_method = 'ppa'
# ot_method = 'b-admm'
gamma = 1e-1
gwb_layers = 5
ot_layers = 50

# alg. params
size_batch = 250
epochs = 10
lr = 0.25
weight_decay = 0
shuffle_data = True
zeta = None  # the weight of diversity regularizer
mode = 'fit'
ssl = [0]

best_acc = np.zeros((len(ssl), ))
for name in names:
    filename_pkl = os.path.join(util.DATA_GRAPH_DIR, name, 'processed_data.pkl')
    graph_data, num_classes = structural_data_list(filename_pkl)
    if len(graph_data[0]) == 4:
        dim_embedding = graph_data[0][2].shape[1]
    else:
        dim_embedding = 1
    print(dim_embedding, num_classes)
    data_sampler = StructuralDataSampler(graph_data)
    labels = []
    for sample in graph_data:
        labels.append(sample[-1])
    labels = np.asarray(labels)

    for i in range(len(ssl)):
        p = ssl[i]
        if p == 0:
            model = FGWF.FGWF(num_samples=len(graph_data),
                              num_classes=num_classes,
                              size_atoms=size_atoms,
                              dim_embedding=dim_embedding,
                              ot_method=ot_method,
                              gamma=gamma,
                              gwb_layers=gwb_layers,
                              ot_layers=ot_layers,
                              prior=data_sampler)
            model = FGWF.train_usl(model, graph_data,
                                   size_batch=size_batch,
                                   epochs=epochs,
                                   lr=lr,
                                   weight_decay=weight_decay,
                                   shuffle_data=shuffle_data,
                                   zeta=zeta,
                                   mode=mode,
                                   visualize_prefix=os.path.join(util.RESULT_DIR, name))
            model.eval()
            features = model.weights.cpu().data.numpy()
            embeddings = features.T
            kmeans = KMeans(init='k-means++', n_clusters=num_classes, n_init=10)
            pred = kmeans.fit_predict(embeddings)
            best_acc[i] = max([1 - np.sum(np.abs(pred - labels)) / len(graph_data),
                               1 - np.sum(np.abs((1 - pred) - labels)) / len(graph_data)])
            print(best_acc[i])
        else:
            model = FGWF.FGWF(num_samples=len(graph_data),
                              num_classes=num_classes,
                              size_atoms=size_atoms,
                              dim_embedding=dim_embedding,
                              ot_method=ot_method,
                              gamma=gamma,
                              gwb_layers=gwb_layers,
                              ot_layers=ot_layers,
                              prior=data_sampler)
            model, predictor, best_acc[i] = FGWF.train_ssl(model, graph_data,
                                                           size_batch=size_batch,
                                                           epochs=epochs,
                                                           lr=lr,
                                                           weight_decay=weight_decay,
                                                           shuffle_data=shuffle_data,
                                                           zeta=zeta,
                                                           mode=mode,
                                                           ssl=p,
                                                           visualize_prefix=os.path.join(util.RESULT_DIR, name))
            FGWF.save_model(predictor, os.path.join(util.MODEL_DIR, '{}_{}_{}_predictor.pkl'.format(name, mode, i)))
            print(best_acc[i])

        FGWF.save_model(model, os.path.join(util.MODEL_DIR, '{}_{}_{}_fgwf.pkl'.format(name, mode, i)))

    print(best_acc)
    filename_pkl = os.path.join(util.RESULT_DIR, 'classification_acc_{}_{}.pkl'.format(name, mode))
    with open(filename_pkl, 'wb') as f:
        pickle.dump(best_acc, f)
