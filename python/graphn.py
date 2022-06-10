import cvxpy as cp
import numpy as np
import networkx as nx
from collections import deque


def generate_graph_code(graph: nx.Graph, getOrderedNeighbours) -> str:
    """
    @brief Calculates the code for G1 according to [Marked Subgraph Isomorphism of Ordered Graphs]
    (https://link.springer.com/content/pdf/10.1007%2FBFb0033230.pdf) by Xiaoyi and Bunke.
    @param graph Graph that is encoded
    @param getOrderedNeighbours Function (graph, node, start_node) that returns an ordered list of ids
    of neighbour nodes of a given node (starting at the given start_node).
    @return Code of the given graph
    """
    num_nodes = len(graph.nodes)
    q = deque()
    relabel_id = 1
    labels = np.zeros(num_nodes, dtype=int)
    s = []
    code = ""
    used_nodes = np.zeros(num_nodes)
    for a in graph.nodes:
        # pick the first inner edge we find and set the initial inner vertex to true
        if len(graph.edges(a)) > 1:
            edges = graph.edges(a)
            for e in edges:
                q.append(e)
                break
            used_nodes[a] = True
            labels[a] = 1
            break

    # first iteration:
    # Q = {vi, vj}, vi = initial (inner node)
    while len(q) > 0:
        v_i, v_j = q.pop()
        ordered_neighbours = getOrderedNeighbours(graph, v_i, v_j)
        for v_k in ordered_neighbours:
            s.append(v_k)
            if not used_nodes[v_k]:
                used_nodes[v_k] = True
                relabel_id += 1
                labels[v_k] = relabel_id
                if len(graph.edges(v_k)) > 1:
                    edges = graph.edges(v_k)
                    for e in edges:
                        q.append(e)
                        break

    for node in s:
        code += str(labels[node])

    return code


def marked_subgraph_isomorphism(G_a: nx.Graph, G_b: nx.Graph) -> bool:
    """
    @brief Implementation of the paper [Marked Subgraph Isomorphism of Ordered Graphs]
    (https://link.springer.com/content/pdf/10.1007%2FBFb0033230.pdf) by Xiaoyi and Bunke.
    @param G_a Graph that is checked if it is a subgraph of G_b
    @param G_b Graph
    @return True if G_a is an isomorphic subgraph of G_b
    """

    return True


def ggd(G_a: nx.Graph, G_b: nx.Graph, dist_func, c_n=1.0, c_e=1.0, verbose=False) -> float:
    """
    @brief Calculates the geometric graph distance
    between two (small) graphs a and b based on the paper [Measuring the Similarity of Geometric Graphs]
    (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.206.900&rep=rep1&type=pdf) by Cheong et al.
    @param G_a Graph A = G(V_A, E_A)
    @param G_b Graph B = G(V_A, E_B)
    @param dist_func Function for calculating the distance that takes two nodes of the input graphs
    @param c_e Cost for edge operations
    @param c_n Cost for vertex operations
    @param verbose Verbose option for solver
    @return Geometric Graph Distance, if any of the graphs is empty (no nodes or edges) returns inf TODO return full
    cost of moving a to b
    """
    an = len(G_a.nodes)
    bn = len(G_b.nodes)

    if an == 0 or bn == 0 or len(G_a.edges) == 0 or len(G_b.edges) == 0:
        return np.inf

    len_uv = []
    # Graph A = a, b, c; e1 e2
    # Graph B = x, y; e3 e4

    # Vuv -> ax, ay, bx, by, cx, cy
    for a, dA in G_a.nodes(data=True):
        for b, dB in G_b.nodes(data=True):
            len_uv.append(dist_func(dA, dB))

    len_e = [dist_func(G_a.nodes[u], G_a.nodes[v]) for u, v in G_a.edges]        # length of edges
    len_e_prime = [dist_func(G_b.nodes[u], G_b.nodes[v]) for u, v in G_b.edges]  # length of other edges

    L_i = [[u, v] for u, v in G_a.edges]   # indices of u,v for e
    L_ip = [[u, v] for u, v in G_b.edges]  # indices for u,v for e'

    L_UV = cp.Constant(len_uv)
    L_E = cp.Constant(sum(len_e))
    L_EP = cp.Constant(sum(len_e_prime))
    C_E = cp.Constant(c_e)
    C_V = cp.Constant(c_n)
    Vuv = cp.Variable((an * bn), value=np.zeros(an * bn), boolean=True)  # Binary Variable Vuv

    Eee = cp.Variable((len(G_a.edges) * len(G_b.edges)), value=np.zeros(len(G_a.edges) * len(G_b.edges)),
                      boolean=True, integer=True)  # Binary Variable Eee

    tmp = []
    # EEe -> e1e3, e1e4, e2e3, e2e4
    for e in len_e:
        for ep in len_e_prime:
            tmp.append(int(e + ep - abs(e - ep)))
    C_EEP = cp.Constant(tmp)  # cost modifier |e| + |e'| - ||e| - |e'|| in objective

    constraints = []

    # for each $$u \in V_A sum_{v \in V_B}{V_{uv} \leq 1}$$
    for i in range(an):
        constraints += [cp.sum(Vuv[i * bn::(i + 1) * bn]) <= 1]

    # for each $$v \in V_B sum_{u \in V_A}{V_{uv} \leq 1}$$
    for i in range(bn):
        constraints += [cp.sum(Vuv[i:an:]) <= 1]

    # c3 = []  # $$e = (u,v) and e' = (u', v'), E_{ee'} \leq 1/2 * (Vuu' + Vvv' + Vuv' + Vvu')$$
    for i in range(len(len_e)):
        ia = int(i / len(G_b.edges))
        ib = i % len(G_b.edges)
        constraints += [Eee[i] <= 0.5 * (
                Vuv[bn * L_i[ia][0]] + Vuv[L_ip[ib][0]] + Vuv[bn * L_i[ia][1]] + Vuv[L_ip[ib][1]])]

    objective = cp.Minimize(
        C_V * cp.sum(cp.multiply(L_UV, Vuv))
        + C_E * cp.sum(L_E)
        + C_E * cp.sum(L_EP)
        - C_E * (cp.sum(cp.multiply(C_EEP, Eee))))
    problem = cp.Problem(objective, constraints)
    result = problem.solve(verbose=verbose)

    return abs(result)
