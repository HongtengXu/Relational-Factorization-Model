"""
Compare different optimization algorithms for calculating Gromov-Wasserstein discrepancy
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import time

import methods.DataIO as DataIO
import methods.GromovWassersteinFramework as GWL


for mode in ['directed', 'undirected']:
    NN = 10
    MM = 1000
    dgw_ppa = np.zeros((MM, NN))
    dgw_badmm = np.zeros((MM, NN))
    for nn in range(NN):

        num_nodes = 100
        maps = np.eye(num_nodes)
        maps = maps[::-1, :]

        num_edges_per_nodes = int(np.log(num_nodes))

        # cost_s = np.zeros((num_nodes, num_nodes))
        # for i in range(10):
        #     graph_ba = nx.barabasi_albert_graph(num_nodes, num_edges_per_nodes, seed=None)
        #     p_s, tmp, idx2node = DataIO.extract_graph_info(graph_ba)
        #     cost_s += tmp
        # cost_s = csr_matrix((cost_s + cost_s.T) / 10)
        #
        # cost_s = np.random.rand(num_nodes, num_nodes)
        # cost_s = cost_s @ cost_s.T
        # cost_s /= np.max(cost_s)

        if mode == 'undirected':
            graph_ba = nx.barabasi_albert_graph(num_nodes, int(num_edges_per_nodes/2), seed=None)
            p_s, cost_s, idx2node = DataIO.extract_graph_info(graph_ba)
            cost_s = cost_s + cost_s.T
        else:
            graph_ba = nx.barabasi_albert_graph(num_nodes, num_edges_per_nodes, seed=None)
            p_s, cost_s, idx2node = DataIO.extract_graph_info(graph_ba)


        # graph_ba = nx.barabasi_albert_graph(num_nodes, num_edges_per_nodes, seed=None)
        # p_t, cost_t, idx2node = DataIO.extract_graph_info(graph_ba)
        # cost_t = cost_t + cost_t.T

        p_t = maps @ p_s
        cost_t = maps @ cost_s @ maps.T

        # p_s = np.ones(p_s.shape) / num_nodes
        # p_t = np.ones(p_t.shape) / num_nodes
        p_s /= np.sum(p_s)
        p_t /= np.sum(p_t)

        # plt.imshow(np.asarray(cost_s.todense()))
        # plt.colorbar()
        # plt.savefig('cost_s.pdf')
        # plt.close('all')
        # #
        # plt.imshow(cost_t)
        # plt.colorbar()
        # plt.savefig('cost_t.pdf')
        # plt.close('all')

        ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
                   'ot_method': 'proximal',
                   'beta': 0.01,
                   'outer_iteration': 3000,  # outer, inner iteration, error bound of optimal transport
                   'iter_bound': 1e-30,
                   'inner_iteration': 1,
                   'sk_bound': 1e-30,
                   'max_iter': 1,  # iteration and error bound for calcuating barycenter
                   'cost_bound': 1e-16,
                   'update_p': False,  # optional updates of source distribution
                   'lr': 0.1,
                   'node_prior': None,
                   'alpha': 0,
                   'test_mode': True}

        cost_st = GWL.node_cost_st(cost_s, cost_t, p_s, p_t,
                                   loss_type=ot_dict['loss_type'], prior=ot_dict['node_prior'])
        cost = GWL.node_cost(cost_s, cost_t, maps / num_nodes, cost_st, ot_dict['loss_type'])
        d_gw0 = (cost * maps / num_nodes).sum()


        t0 = time.time()
        ot_dict['beta'] = 10
        ot_dict['outer_iteration'] = 1
        ot_dict['inner_iteration'] = MM
        ot_dict['ot_method'] = 'b-admm'
        trans1, d_gw1, _ = GWL.gromov_wasserstein_discrepancy(cost_s, cost_t, p_s, p_t, ot_dict)
        t1 = time.time()
        ot_dict['beta'] = 1e-2
        ot_dict['outer_iteration'] = MM
        ot_dict['inner_iteration'] = 10
        ot_dict['ot_method'] = 'proximal'
        trans2, d_gw2, _ = GWL.gromov_wasserstein_discrepancy(cost_s, cost_t, p_s, p_t, ot_dict)
        t2 = time.time()

        print(len(d_gw1), len(d_gw2))
        dgw_badmm[:, nn] = np.asarray(d_gw1)
        dgw_ppa[:, nn] = np.asarray(d_gw2)

        print('Sparsity:\n b-admm={:.1f} time={:.3f}sec,\n ippa={:.1f} time={:.3f}sec,\n'.format(
            np.sum(trans1 == 0) / trans1.size * 100, t1 - t0,
            np.sum(trans2 == 0) / trans2.size * 100, t2 - t1))

        # plt.imshow(trans1)
        # plt.colorbar()
        # plt.savefig('ot_b-admm.pdf')
        # plt.close('all')
        #
        # plt.imshow(trans2)
        # plt.colorbar()
        # plt.savefig('ot_ippa.pdf')
        # plt.close('all')

    dgw_ppa_mean = np.mean(dgw_ppa, axis=1)
    dgw_ppa_std = 0.25 * np.std(dgw_ppa, axis=1)
    dgw_badmm_mean = np.mean(dgw_badmm, axis=1)
    dgw_badmm_std = 0.5 * np.std(dgw_badmm, axis=1)

    plt.figure(figsize=(5, 5))
    plt.plot(range(MM), dgw_ppa_mean, label='PPA', color='blue')
    plt.fill_between(range(MM), dgw_ppa_mean - dgw_ppa_std, dgw_ppa_mean + dgw_ppa_std,
                     color='blue', alpha=0.2)
    plt.plot(range(MM), dgw_badmm_mean, label='BADMM', color='orange')
    plt.fill_between(range(MM), dgw_badmm_mean - dgw_badmm_std, dgw_badmm_mean + dgw_badmm_std,
                     color='orange', alpha=0.2)
    plt.legend()
    plt.xlabel('The number of iterations')
    plt.ylabel('GW discrepancy')
    plt.savefig('cmp_{}.pdf'.format(mode))
    plt.close('all')

