import dev.util as util
import os
import pickle

name_datasets = ['COLORS-3']  # util.navigate_child_dirs(util.DATA_GRAPH_DIR)

for name in name_datasets:
    filename_edges = os.path.join(util.DATA_GRAPH_DIR, name, '{}_A.txt'.format(name))
    filename_graphs = os.path.join(util.DATA_GRAPH_DIR, name, '{}_graph_indicator.txt'.format(name))
    filename_labels = os.path.join(util.DATA_GRAPH_DIR, name, '{}_graph_labels.txt'.format(name))
    # print(filename_graphs)
    # print(filename_labels)
    # print(filename_edges)

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

    graph2node = {}
    node2graph = {}
    for i in range(len(graphs)):
        node_id = i + 1
        graph_id = graphs[i]
        node2graph[node_id] = graph_id

        if graph_id not in graph2node.keys():
            graph2node[graph_id] = {}
            idx = len(graph2node[graph_id])
            graph2node[graph_id][node_id] = idx
        else:
            # graph2node[graph_id].append(node_id)
            idx = len(graph2node[graph_id])
            graph2node[graph_id][node_id] = idx

    graph2size = {}
    ave_node_size = 0
    for graph_id in graph2node.keys():
        graph2size[graph_id] = len(graph2node[graph_id])
        ave_node_size += len(graph2node[graph_id])
    ave_node_size /= len(graph2node)

    graph2edge = {}
    for m in range(len(edges)):
        src = edges[m][0]
        dst = edges[m][1]
        # print(edges[m], src, dst)
        graph_id = node2graph[src]
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

    print('{}: {}/{}/{} graphs, {} classes, {:.2f} nodes + {:.2f} edges per graph'.format(
        name, len(graph2edge), len(graph2label), len(graph2size), num_class, ave_node_size, ave_edge_size))

    graph_edges = []
    graph_labels = []
    graph_sizes = []
    for key in graph2edge.keys():
        graph_edges.append(graph2edge[key])
        graph_labels.append(graph2label[key])
        graph_sizes.append(graph2size[key])

    filename_pkl = os.path.join(util.DATA_GRAPH_DIR, name, 'processed_data.pkl')
    with open(filename_pkl, 'wb') as f:
        pickle.dump([graph_edges, graph_sizes, graph_labels, num_class], f)







