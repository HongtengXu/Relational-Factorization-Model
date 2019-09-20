import dev.util as util
import numpy as np
import os
import pickle

name_datasets = util.navigate_child_dirs(util.DATA_POINT_DIR)

for name in name_datasets:
    filename_edges = os.path.join(util.DATA_POINT_DIR, name, '{}_A.txt'.format(name))
    filename_graphs = os.path.join(util.DATA_POINT_DIR, name, '{}_graph_indicator.txt'.format(name))
    filename_labels = os.path.join(util.DATA_POINT_DIR, name, '{}_graph_labels.txt'.format(name))
    filename_nodes = os.path.join(util.DATA_POINT_DIR, name, '{}_node_attributes.txt'.format(name))

    with open(filename_edges) as f:
        edges = f.readlines()
    for n in range(len(edges)):
        edge = edges[n].strip("\n")
        edges[n] = [float(node) for node in edge.split(',')]

    with open(filename_graphs) as f:
        graphs = f.readlines()
    for n in range(len(graphs)):
        graph = int(graphs[n].strip("\n"))
        graphs[n] = graph

    with open(filename_labels) as f:
        labels = f.readlines()
    for n in range(len(labels)):
        label = int(labels[n].strip("\n"))
        labels[n] = label

    label2idx = {}
    idx = 0
    for n in range(len(labels)):
        if labels[n] not in label2idx.keys():
            label2idx[labels[n]] = idx
            idx += 1
    num_class = len(label2idx)

    with open(filename_nodes) as f:
        nodes = f.readlines()

    node2feature = {}
    for i in range(len(nodes)):
        node_id = i + 1
        values = nodes[i].strip("\n")
        feature = np.array([float(value) for value in values.split(',')])
        node2feature[node_id] = feature
    dim_feature = node2feature[1].shape[0]

    graph2node = {}
    node2graph = {}
    for i in range(len(graphs)):
        node_id = i + 1
        graph_id = graphs[i]
        node2graph[node_id] = graph_id

        if graph_id not in graph2node.keys():
            graph2node[graph_id] = {}
            graph2node[graph_id][node_id] = 0
        else:
            idx = len(graph2node[graph_id])
            graph2node[graph_id][node_id] = idx

    graph2size = {}
    ave_node_size = 0
    for graph_id in graph2node.keys():
        graph2size[graph_id] = len(graph2node[graph_id])
        ave_node_size += len(graph2node[graph_id])
    ave_node_size /= len(graph2node)

    graph2feature = {}
    for graph_id in graph2node.keys():
        num_nodes = graph2size[graph_id]
        features = np.zeros((num_nodes, dim_feature))
        for node_id in graph2node[graph_id].keys():
            feature = node2feature[node_id]
            row_id = graph2node[graph_id][node_id]
            features[row_id, :] = feature
        graph2feature[graph_id] = features

    graph2edge = {}
    for m in range(len(edges)):
        src = edges[m][0]
        dst = edges[m][1]
        # print(edges[m], src, dst)
        graph_id1 = node2graph[src]
        graph_id = node2graph[dst]
        if graph_id != graph_id1:
            print('wrong')
        src_id = graph2node[graph_id][src]
        dst_id = graph2node[graph_id][dst]
        if graph_id not in graph2edge.keys():
            graph2edge[graph_id] = [[src_id, dst_id]]
        else:
            graph2edge[graph_id].append([src_id, dst_id])

    ave_edge_size = 0
    for graph_id in graph2edge.keys():
        ave_edge_size += len(graph2edge[graph_id])
    ave_edge_size /= len(graph2edge)

    graph2label = {}
    for n in range(len(labels)):
        graph_id = n + 1
        graph2label[graph_id] = label2idx[labels[n]]

    print('{}: {}/{}/{}/{} graphs, {} classes, {} features, {:.2f} nodes + {:.2f} edges per graph'.format(
        name, len(graph2edge), len(graph2label), len(graph2feature), len(graph2size),
        num_class, dim_feature, ave_node_size, ave_edge_size))

    graph_edges = []
    graph_labels = []
    graph_features = []
    graph_sizes = []
    for key in graph2edge.keys():
        graph_edges.append(graph2edge[key])
        graph_labels.append(graph2label[key])
        graph_features.append(graph2feature[key])
        graph_sizes.append(graph2size[key])

    filename_pkl = os.path.join(util.DATA_POINT_DIR, name, 'processed_data.pkl')
    with open(filename_pkl, 'wb') as f:
        pickle.dump([graph_edges, graph_sizes, graph_labels, graph_features, num_class], f)







