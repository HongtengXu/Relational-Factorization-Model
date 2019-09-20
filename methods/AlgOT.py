import torch

from typing import Tuple


def cost_mat(cost_s: torch.Tensor,
             cost_t: torch.Tensor,
             p_s: torch.Tensor,
             p_t: torch.Tensor,
             tran: torch.Tensor,
             emb_s: torch.Tensor = None,
             emb_t: torch.Tensor = None) -> torch.Tensor:
    """
    Implement cost_mat for Gromov-Wasserstein discrepancy (GWD)

    Suppose the loss function in GWD is |a-b|^2 = a^2 - 2ab + b^2. We have:

    f1(a) = a^2,
    f2(b) = b^2,
    h1(a) = a,
    h2(b) = 2b

    When the loss function can be represented in the following format: loss(a, b) = f1(a) + f2(b) - h1(a)h2(b), we have

    cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
    cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        p_s: (ns, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        p_t: (nt, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        tran: (ns, nt) matrix (torch tensor), representing the optimal transport from source to target domain.
        emb_s: (ns, d) matrix
        emb_t: (nt, d) matrix
    Returns:
        cost: (ns, nt) matrix (torch tensor), representing the cost matrix conditioned on current optimal transport
    """
    f1_st = ((cost_s ** 2) @ p_s).repeat(1, tran.size(1))
    f2_st = (torch.t(p_t) @ torch.t(cost_t ** 2)).repeat(tran.size(0), 1)
    cost_st = f1_st + f2_st
    cost = cost_st - 2 * cost_s @ tran @ torch.t(cost_t)

    if emb_s is not None and emb_t is not None:
        tmp1 = emb_s @ torch.t(emb_t)
        tmp2 = torch.sqrt((emb_s ** 2) @ torch.ones(emb_s.size(1), 1))
        tmp3 = torch.sqrt((emb_t ** 2) @ torch.ones(emb_t.size(1), 1))
        cost += 0.5 * (1 - tmp1 / (tmp2 @ torch.t(tmp3)))
        # tmp1 = 2 * emb_s @ torch.t(emb_t)
        # tmp2 = ((emb_s ** 2) @ torch.ones(emb_s.size(1), 1)).repeat(1, tran.size(1))
        # tmp3 = ((emb_t ** 2) @ torch.ones(emb_t.size(1), 1)).repeat(1, tran.size(0))
        # tmp = 0.1 * (tmp2 + torch.t(tmp3) - tmp1) / (emb_s.size(1) ** 2)
        # cost += tmp
    return cost


def ot_badmm(cost_s: torch.Tensor,
             cost_t: torch.Tensor,
             p_s: torch.Tensor,
             p_t: torch.Tensor,
             tran: torch.Tensor,
             dual: torch.Tensor,
             gamma: float,
             num_layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve min_{trans in Pi(p_s, p_t)} <cost_st - 2 * cost_s * tran * cost_t^T, tran> via Bregman ADMM (B-ADMM)

    Introducing an auxiliary variable "aux", we reformulate the problem as

    min <cost_st - 2 * cost_s * aux * cost_t^T, tran>
    s.t. tran in Pi(p_s, .), aux in Pi(., p_t), and tran = aux

    and solve it via B-ADMM.

    Specifically, further introducing a dual variable "dual",
    we repeat the following steps till converge:

    step 1:
    min_{tran in Pi(p_s, .)} <cost_st - 2 * cost_s * aux * cost_t^T, tran> + <dual, tran - aux> + gamma * KL(tran | aux)

    step 2:
    min_{aux in Pi(., p_t)} <-2 * cost_s^T * tran * cost_t, aux> + <dual, tran - aux> + gamma * KL(aux | tran)

    step 3:
    dual = dual + gamma * (tran - aux)

    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        p_s: (ns, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        p_t: (nt, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        tran: (ns, nt) matrix (torch tensor), representing initial optimal transport from source to target domain.
        dual: (ns, nt) matrix (torch tensor), representing dual variables
        gamma: the weight of Bergman divergence
        num_layer: the number of iterations (the number of layers in this computation module)

    Returns:
        tran: (ns, nt) matrix (torch tensor), representing updated optimal transport from source to target domain.
        dual: (ns, nt) matrix (torch tensor), representing updated dual variables

    """
    all1_s = torch.ones(p_s.size())
    all1_t = torch.ones(p_t.size())
    for m in range(num_layer):
        kernel_a = torch.exp((dual + 2 * torch.t(cost_s) @ tran @ cost_t) / gamma) * tran
        b = p_t / (torch.t(kernel_a) @ all1_s)
        aux = (all1_s @ torch.t(b)) * kernel_a

        dual = dual + gamma * (tran - aux)

        kernel_t = torch.exp(-(cost_mat(cost_s, cost_t, p_s, p_t, aux) + dual) / gamma) * aux
        a = p_s / (kernel_t @ all1_t)
        tran = (a @ torch.t(all1_t)) * kernel_t

    return tran, dual


def ot_ppa(cost_s: torch.Tensor,
           cost_t: torch.Tensor,
           p_s: torch.Tensor,
           p_t: torch.Tensor,
           tran: torch.Tensor,
           dual: torch.Tensor,
           gamma: float,
           num_layer: int,
           sinkhorn_iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve min_{trans in Pi(p_s, p_t)} <cost_st - 2 * cost_s * tran * cost_t^T, tran> via proximal point algorithm (ppa)

    Specifically, at each iteration, we solve the following problem:

    min_{tran in Pi(p_s, p_t)} <cost_st - 2 * cost_s * tran0 * cost_t^T, tran> + gamma * KL(tran | tran0)

    where "tran0" is previous estimation. This problem can be reformulated as

    min_{tran in Pi(p_s, p_t)} <cost_st - 2 * cost_s * tran0 * cost_t^T - gamma * log(tran0), tran> + gamma * H(tran)

    where "H(tran)" is the entropy of tran.
    This problem can be solved by Sinkhorn-Knopp iterations via introducing a dual variable "dual".

    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        p_s: (ns, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        p_t: (nt, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        tran: (ns, nt) matrix (torch tensor), representing initial optimal transport from source to target domain.
        dual: (ns, 1) vector (torch tensor), representing dual variables
        gamma: the weight of Bergman proximal term
        num_layer: the number of iterations (the number of layers in this computation module)
        sinkhorn_iters: the number of Sinkhorn-Knopp iterations

    Returns:
        tran: (ns, nt) matrix (torch tensor), representing updated optimal transport from source to target domain.
        dual: (ns, nt) matrix (torch tensor), representing updated dual variables
    """
    for m in range(num_layer):
        kernel = torch.exp(-cost_mat(cost_s, cost_t, p_s, p_t, tran) / gamma) * tran
        b = p_t / (torch.t(kernel) @ dual)
        for i in range(sinkhorn_iters):
            dual = p_s / (kernel @ b)
            b = p_t / (torch.t(kernel) @ dual)
        tran = (dual @ torch.t(b)) * kernel
    return tran, dual


def optimal_transport(cost_s: torch.Tensor,
                      cost_t: torch.Tensor,
                      p_s: torch.Tensor,
                      p_t: torch.Tensor,
                      ot_method: str,
                      gamma: float,
                      num_layer: int,
                      emb_s: torch.Tensor = None,
                      emb_t: torch.Tensor = None):
    tran = p_s @ torch.t(p_t)
    if ot_method == 'ppa':
        dual = torch.ones(p_s.size()) / p_s.size(0)
        for m in range(num_layer):
            kernel = torch.exp(-cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t) / gamma) * tran
            b = p_t / (torch.t(kernel) @ dual)
            for i in range(20):
                dual = p_s / (kernel @ b)
                b = p_t / (torch.t(kernel) @ dual)
            tran = (dual @ torch.t(b)) * kernel

    elif ot_method == 'b-admm':
        all1_s = torch.ones(p_s.size())
        all1_t = torch.ones(p_t.size())
        dual = torch.zeros(p_s.size(0), p_t.size(0))
        for m in range(num_layer):
            kernel_a = torch.exp((dual + 2 * torch.t(cost_s) @ tran @ cost_t) / gamma) * tran
            b = p_t / (torch.t(kernel_a) @ all1_s)
            aux = (all1_s @ torch.t(b)) * kernel_a

            dual = dual + gamma * (tran - aux)

            kernel_t = torch.exp(-(cost_mat(cost_s, cost_t, p_s, p_t, aux, emb_s, emb_t) + dual) / gamma) * aux
            a = p_s / (kernel_t @ all1_t)
            tran = (a @ torch.t(all1_t)) * kernel_t
    d_gw = (cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t) * tran).sum()
    return d_gw, tran


def ot_fgw(cost_s: torch.Tensor,
           cost_t: torch.Tensor,
           p_s: torch.Tensor,
           p_t: torch.Tensor,
           ot_method: str,
           gamma: float,
           num_layer: int,
           emb_s: torch.Tensor = None,
           emb_t: torch.Tensor = None):
    tran = p_s @ torch.t(p_t)
    if ot_method == 'ppa':
        dual = torch.ones(p_s.size()) / p_s.size(0)
        for m in range(num_layer):
            cost = cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t)
            # cost /= torch.max(cost)
            kernel = torch.exp(-cost / gamma) * tran
            b = p_t / (torch.t(kernel) @ dual)
            for i in range(5):
                dual = p_s / (kernel @ b)
                b = p_t / (torch.t(kernel) @ dual)
            tran = (dual @ torch.t(b)) * kernel

    elif ot_method == 'b-admm':
        all1_s = torch.ones(p_s.size())
        all1_t = torch.ones(p_t.size())
        dual = torch.zeros(p_s.size(0), p_t.size(0))
        for m in range(num_layer):
            kernel_a = torch.exp((dual + 2 * torch.t(cost_s) @ tran @ cost_t) / gamma) * tran
            b = p_t / (torch.t(kernel_a) @ all1_s)
            aux = (all1_s @ torch.t(b)) * kernel_a

            dual = dual + gamma * (tran - aux)

            cost = cost_mat(cost_s, cost_t, p_s, p_t, aux, emb_s, emb_t)
            # cost /= torch.max(cost)
            kernel_t = torch.exp(-(cost + dual) / gamma) * aux
            a = p_s / (kernel_t @ all1_t)
            tran = (a @ torch.t(all1_t)) * kernel_t
    d_gw = (cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t) * tran).sum()
    return d_gw, tran
