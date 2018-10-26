"""ProtocopRank is the base class for ranking subgraphs for a given query.

The constructor takes a networkx multi-digraph.
The primary methods are rank(subgraphs) and score(subgraph)
The most typical use case is:
    # G is a networkX MultiDiGraph()
    # subgraphs is a list of networkX MultiDiGraph()
    pr = ProtocopRank(G)
    score_struct = pr.report_scores_dict(subgraphs)
"""

import logging
import time
import heapq
import operator
from collections import defaultdict
from itertools import combinations

import numpy as np
import numpy.linalg

logger = logging.getLogger(__name__)


def argsort(x, reverse=False):
    """Return the indices that would sort the array."""
    return [p[0] for p in sorted(enumerate(x), key=lambda elem: elem[1], reverse=reverse)]


def flatten_semilist(x):
    lists = [n if isinstance(n, list) else [n] for n in x]
    return [e for el in lists for e in el]


class Ranker:
    """Ranker."""

    graphInfo = {}  # will be populated on construction
    naga_parameters = {'alpha': .9, 'beta': .9}
    prescreen_count = 2000  # only look at this many graphs more in depth
    teleport_weight = 0.001  # probability to teleport along graph (make random inference) in hitting time calculation

    def __init__(self, graph=None, question=None):
        """Create ranker."""
        logger.info("QUESTION")
        logger.info(question)
        logger.info("WHOLE GRAPH")
        logger.info(graph)
        self.graph = graph
        self.question = question
        self._evaluated_templates = {}
        self._result_count = -1

    def set_weights(self, method='ngd'):
        """Initialize weights on the graph based on metadata.

        * 'ngd' uses the omnicorp_article_count on the edges to generate
        normalized google distance weights.
        """

        if method != 'ngd':
            raise RuntimeError('You must use method "ngd".')

        node_pubs = {n['id']: n['omnicorp_article_count'] for n in self.graph['nodes']}

        N = 1e8  # approximate number of articles in corpus * typical number of keywords
        minimum_article_node_count = 1000  # assume every concept actually has at least this many publications we may or may not know about

        mean_ngd = 1e-8

        # notation: edge contains metadata, e is edge without metadata
        edges = self.graph['edges']
        for edge in edges:
            pub_count = len(edge['publications']) if 'publications' in edge else 0

            # initialize scoring info - ngd initializes to big number (np.inf not
            # well-liked) so that weight becomes zero
            scoring_info = {'num_pubs': pub_count, 'ngd': 1e6}
            edge['scoring'] = scoring_info

            source_pubs = max(int(node_pubs[edge['source_id']]), minimum_article_node_count)
            target_pubs = max(int(node_pubs[edge['target_id']]), minimum_article_node_count)

            edge_count = edge['scoring']['num_pubs'] + 1  # avoid log(0) problem

            # formula for normalized google distance
            ngd = (np.log(max(source_pubs, target_pubs)) - np.log(edge_count)) / \
                (np.log(N) - np.log(min(source_pubs, target_pubs)))

            # this shouldn't happen but theoretically could
            if ngd < 0:
                ngd = 0

            edge['scoring']['ngd'] = ngd
            mean_ngd += ngd

        mean_ngd = mean_ngd / len(edges)

        for edge in edges:
            # ngd is like a metric, we need a similarity
            # common way to do this is with a gaussian kernel
            weight = np.exp(-(edge['scoring']['ngd'] / mean_ngd)**2 / 2)

            # make sure weights on edges are not nan valued - set them to zero otherwise
            edge['weight'] = weight if np.isfinite(weight) else 0

    def sum_edge_weights(self, subgraph):
        """Add edge weights."""
        edge_ids = flatten_semilist(subgraph['edges'].values())
        weights = [e['weight'] for e in self.graph['edges'] if e['id'] in edge_ids]
        return sum(weights)

    def prescreen(self, subgraph_list, max_results=None):
        """Prescreen subgraphs.

        Keep the top max_results or self.prescreen_count, by their total edge weight.
        """
        if max_results is None:
            max_results = self.prescreen_count

        logger.debug(f'  Getting {len(subgraph_list)} prescreen scores...')
        prescreen_scores = [self.sum_edge_weights(sg) for sg in subgraph_list]

        logger.debug(f'  Getting top {max_results}...')
        prescreen_sorting = [x[0] for x in heapq.nlargest(max_results, enumerate(prescreen_scores), key=operator.itemgetter(1))]

        logger.debug('  Returning sorted results...')
        return [subgraph_list[i] for i in prescreen_sorting]

    def rank(self, subgraph_list, max_results=250):
        """Generate a sorted list and scores for a set of subgraphs."""

        if not subgraph_list:
            return ([], [])

        # add weights to edges
        logger.debug('Setting weights... ')
        start = time.time()
        self.set_weights(method='ngd')
        logger.debug(f"{time.time()-start} seconds elapsed.")

        # prescreen
        if max_results is not None:
            logger.debug("Prescreening subgraph_list... ")
            start = time.time()
            subgraph_list = self.prescreen(subgraph_list, max_results=max([self.prescreen_count, max_results * 2]))
            logger.debug(f"{time.time()-start} seconds elapsed.")

        # get subgraph statistics
        logger.debug("Calculating subgraph statistics... ")
        start = time.time()
        logger.info("subgraph_statistic on ")
        graph_stat = [self.subgraph_statistic(sg, metric_type='volt') for sg in subgraph_list]
        logger.debug(f"{time.time()-start} seconds elapsed.")

        # Fail safe to nuke nans
        ranking_scores = [r if np.isfinite(r) and r >= 0 else -1 for r in graph_stat]

        # sort by scores
        ranking_sorting = argsort(ranking_scores, reverse=True)
        subgraph_list = [subgraph_list[i] for i in ranking_sorting]
        subgraph_scores = [ranking_scores[i] for i in ranking_sorting]

        # trim output
        if max_results is not None:
            logger.debug('Keeping top %d...', max_results)
            subgraph_list = subgraph_list[:max_results]
            subgraph_scores = subgraph_scores[:max_results]

        return (subgraph_scores, subgraph_list)

    def report_ranking(self, subgraph_list, max_results=250):
        """Report ranking."""
        # construct the output that question.py expects

        (subgraph_scores, subgraph_list) = self.rank(subgraph_list, max_results=max_results)

        # add extra computed metadata in self.graph to subgraph for display
        logger.debug("Extracting subgraphs... ")
        start = time.time()
        report = []
        for i, subgraph in enumerate(subgraph_list):
            node_ids = flatten_semilist(subgraph['nodes'].values())
            nodes = [n for n in self.graph['nodes'] if n['id'] in node_ids]
            edge_ids = flatten_semilist(subgraph['edges'].values())
            edges = [e for e in self.graph['edges'] if e['id'] in edge_ids]
            sgr = {
                'nodes': nodes,
                'edges': edges,
                'score': subgraph_scores[i]
            }
            report.append(sgr)
        logger.debug(f"{time.time()-start} seconds elapsed.")

        return (report, subgraph_list)

    def get_rgraph(self, subgraph):
        """Get "ranker" subgraph."""

        # get list of nodes
        nodes = []
        knode_map = defaultdict(set)
        for qnode_id in subgraph['nodes']:
            knode_ids = subgraph['nodes'][qnode_id]
            if not isinstance(knode_ids, list):
                knode_ids = [knode_ids]
            for knode_id in knode_ids:
                rnode_id = f"{qnode_id}/{knode_id}"
                nodes.append(rnode_id)
                knode_map[knode_id].add(rnode_id)

        # get "result" edges
        edges = []
        for qedge_id in subgraph['edges']:
            if qedge_id[0] == 's':
                continue
            qedge = next(e for e in self.question['edges'] if e['id'] == qedge_id)
            kedge_ids = subgraph['edges'][qedge_id]
            if not isinstance(kedge_ids, list):
                kedge_ids = [kedge_ids]
            for kedge_id in kedge_ids:
                kedge = next(e for e in self.graph['edges'] if e['id'] == kedge_id)
                # find source and target
                candidate_source_ids = [f"{qedge['source_id']}/{kedge['source_id']}", f"{qedge['source_id']}/{kedge['target_id']}"]
                candidate_target_ids = [f"{qedge['target_id']}/{kedge['source_id']}", f"{qedge['target_id']}/{kedge['target_id']}"]
                source_id = next(node_id for node_id in nodes if node_id in candidate_source_ids)
                target_id = next(node_id for node_id in nodes if node_id in candidate_target_ids)
                edge = {
                    'weight': kedge['weight'],
                    'source_id': source_id,
                    'target_id': target_id
                }
                edges.append(edge)

        # get "support" edges
        for kedge in self.graph['edges']:
            if not (kedge['type'] == 'literature_co-occurrence' and kedge['source_id'] in knode_map and kedge['target_id'] in knode_map):
                continue
            for source_id in knode_map[kedge['source_id']]:
                for target_id in knode_map[kedge['target_id']]:
                    edge = {
                        'weight': kedge['weight'],
                        'source_id': source_id,
                        'target_id': target_id
                    }
                    edges.append(edge)

        return nodes, edges

    def subgraph_statistic(self, subgraph, metric_type='hit'):
        """Compute subgraph score."""
        terminals = terminal_nodes(self.question)
        laplacian, node_ids = self.graph_laplacian(subgraph)
        terminals = [n for n in node_ids if any([n.startswith(t) for t in terminals])]
        if metric_type == 'mix':
            return 1 / mixing_time_from_laplacian(laplacian)
        elif metric_type == 'hit':
            htimes = []
            node_id_set = set(node_ids)
            for from_id, to_id in combinations(terminals, 2):
                node_ids_sorted = [from_id] + list(node_id_set - {from_id, to_id}) + [to_id]
                idx = [node_ids.index(node_id) for node_id in node_ids_sorted]
                idx = np.expand_dims(np.array(idx), axis=1)
                idy = np.transpose(idx)
                Q = -laplacian[idx, idy]
                htimes.append(1 / hitting_times_miles(Q)[0][0])
            return sum(htimes)
        elif metric_type == 'volt':
            voltages = []
            node_id_set = set(node_ids)
            for from_id, to_id in combinations(terminals, 2):
                node_ids_sorted = [from_id] + list(node_id_set - {from_id, to_id}) + [to_id]
                idx = [node_ids.index(node_id) for node_id in node_ids_sorted]
                idx = np.expand_dims(np.array(idx), axis=1)
                idy = np.transpose(idx)
                L = laplacian[idx, idy]
                voltages.append(1 / voltage_from_laplacian(L))
            return sum(voltages)
        else:
            raise ValueError(f'Unknown metric type "{metric_type}"')

    def graph_laplacian(self, subgraph):
        """Generate graph Laplacian."""
        # subgraph is a list of dicts with fields 'id' and 'bound'

        node_ids, edges = self.get_rgraph(subgraph)

        # compute graph laplacian for this case with potentially duplicated nodes
        num_nodes = len(node_ids)
        laplacian = np.zeros((num_nodes, num_nodes))
        index = {node_id: node_ids.index(node_id) for node_id in node_ids}
        for edge in edges:
            source_id, target_id, weight = edge['source_id'], edge['target_id'], edge['weight']
            i, j = index[source_id], index[target_id]
            laplacian[i, j] += -weight
            laplacian[j, i] += -weight
            laplacian[i, i] += weight
            laplacian[j, j] += weight

        # add teleportation to allow leaps of faith
        laplacian = laplacian + self.teleport_weight * (num_nodes * np.eye(num_nodes) - np.ones((num_nodes, num_nodes)))
        return laplacian, node_ids


def terminal_nodes(question):
    """Return indices of terminal question nodes.

    Terminal nodes are those that have degree 1.
    """
    degree = defaultdict(int)
    for edge in question['edges']:
        degree[edge['source_id']] += 1
        degree[edge['target_id']] += 1
    return [f"{key}" for i, key in enumerate(degree) if degree[key] == 1]


def mixing_time_from_laplacian(laplacian):
    """Compute mixing time from Laplacian.

    mixing time = 1/log(1/μ)
                = 1/(1 - μ) if μ ~= 1
    μ = second-largest eigenvalue modulus (absolute eigenvalue)

    Boyd et al. 2004. Fastest Mixing Markov Chain on a Graph. SIAM Review Vol 46, No 4, pp 667-689
    https://web.stanford.edu/~boyd/papers/pdf/fmmc.pdf
    """
    # invent discrete-time transition probability matrix
    g = max(np.diag(laplacian))
    if g < 1e-8 or not np.isfinite(g):
        return -1
    P = np.eye(laplacian.shape[0]) - laplacian / g / 2

    try:  # always put other people's code inside a try catch loop
        eigvals = numpy.linalg.eigvals(P)
    except Exception:
        # this should never happen for nonzero teleportation
        logger.debug(f"Eigenvalue computation failed for P:\n{P}")
        return -1

    eigvals = np.abs(eigvals)
    eigvals.sort()  # sorts ascending

    return 1 / (1 - eigvals[-2]) / g


def voltage_from_laplacian(L):
    iv = np.zeros(L.shape[0])
    iv[0] = -1
    iv[-1] = 1
    results = np.linalg.lstsq(L, iv)
    potentials = results[0]
    return potentials[-1] - potentials[0]


def hitting_times(Q):
    """Compute hitting time AKA mean first-passage time.

    Q is the "transition rate matrix" or "infinitesimal generator matrix"

    https://cims.nyu.edu/~holmes/teaching/asa17/handout-Lecture4_2017.pdf
    http://www.seas.ucla.edu/~vandenbe/133A/lectures/cls.pdf
    """
    N = Q.shape[0]
    b = -np.ones((N, 1))

    C = np.zeros((1, N))
    C[-1, -1] = 1
    d = np.zeros((1, 1))
    Z = np.zeros((1, 1))
    A = np.concatenate((
        np.concatenate((np.matmul(np.transpose(Q), Q), np.transpose(C)), axis=1),
        np.concatenate((C, Z), axis=1)
    ), axis=0)
    b = np.concatenate((np.matmul(np.transpose(Q), b), d), axis=0)

    results = np.linalg.lstsq(A, b)
    tau = results[0]
    # print("b: ", b)
    # print("A * tau: ", np.matmul(A, tau))
    return results[0]


def hitting_times_miles(Q):
    """Compute hitting time AKA mean first-passage time.

    Similar to above, but actually works. Since transitions from the absorbing states
    shouldn't matter, use those rows of the matrix to enforce that the hitting times
    starting in the absorbing states are zero.

    Assume that the last state is the only absorbing state, i.e. the queried-for node.
    To score a branching answer graph, sum the hitting times from all other ternminal
    nodes to the queried-for node.
    """
    N = Q.shape[0]
    b = -np.ones((N, 1))
    b[-1] = 0

    Qtilde = Q
    Qtilde[-1, :] = 0
    Qtilde[-1, -1] = 1

    results = np.linalg.lstsq(Qtilde, b)
    tau = results[0]
    return tau
