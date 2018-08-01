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
import networkx as nx
import numpy as np
import numpy.linalg

logger = logging.getLogger(__name__)


class Ranker:
    """Ranker."""

    G = nx.MultiDiGraph()  # a networkx MultiDiGraph() with weights
    graphInfo = {}  # will be populated on construction
    naga_parameters = {'alpha': .9, 'beta': .9}
    prescreen_count = 2000  # only look at this many graphs more in depth
    teleport_weight = 0.001  # probability to teleport along graph (make random inference) in hitting time calculation
    output_count = 250

    def __init__(self, graph=nx.MultiDiGraph()):
        """Create ranker."""
        self.graph = graph
        self._evaluated_templates = {}
        self._result_count = -1

    def set_weights(self, method='ngd'):
        """Initialize weights on the graph based on metadata.

        * 'logistic' just counts # of publications and applies a hand tuned
        logistic.
        * 'ngd' uses the omnicorp_article_count on the edges to generate
        normalized google distance weights.
        """
        # notation: edge contains metadata, e is edge without metadata
        edges = self.graph.edges(data=True, keys=True)
        pub_counts = {edge[:-1]: len(edge[-1]['publications']) if 'publications' in edge[-1] else 0 for edge in edges}

        # initialize scoring info - ngd initializes to big number (np.inf not
        # well-liked) so that weight becomes zero
        scoring_info = {e: {'num_pubs': pub_counts[e], 'ngd': 1e6} for e in pub_counts}
        nx.set_edge_attributes(self.graph, values=scoring_info, name='scoring')

        if method == 'logistic':
            # apply logistic function to publications to get weights
            weights = {e: 1 / (1 + np.exp((5 - pub_counts[e]) / 2)) for e in pub_counts}

        elif method == 'ngd':
            # this method tries to use omnicorp's article counts to normalize
            # probabilities in a meaningful way
            nodes = self.graph.nodes()
            node_pub_sum = {n: sum([edge[-1]['scoring']['num_pubs'] for edge in self.graph.edges(n, data=True)]) for n in nodes}

            N = 1e8  # approximate number of articles in corpus * typical number of keywords
            default_article_node_count = 25000  # large but typical number of article counts for a node
            minimum_article_node_count = 1000  # assume every concept actually has at least this many publications we may or may not know about

            mean_ngd = 1e-8
            node_count = [minimum_article_node_count] * 2
            for edge in edges:
                for i in range(2):
                    if 'omnicorp_article_count' in self.graph.node[edge[i]]:
                        node_count[i] = int(self.graph.node[edge[i]]['omnicorp_article_count'])
                    else:
                        node_count[i] = default_article_node_count

                    # make sure the node counts are at least as great as the sum of pubs along the edges we have
                    node_count[i] = max(node_count[i], node_pub_sum[edge[i]], minimum_article_node_count)

                edge_count = edge[-1]['scoring']['num_pubs'] + 1  # avoid log(0) problem

                # formula for normalized google distance
                ngd = (np.log(min(node_count)) - np.log(edge_count)) / \
                    (np.log(N) - np.log(max(node_count)))

                # this shouldn't happen but theoretically could
                if ngd < 0:
                    ngd = 0

                edge[-1]['scoring']['ngd'] = ngd
                mean_ngd = mean_ngd + ngd

            mean_ngd = mean_ngd / len(edges)
            logger.debug(f"Mean ngd: {mean_ngd}")

            # ngd is like a metric, we need a similarity
            # common way to do this is with a gaussian kernel
            weights = {edge[:-1]: np.exp(-(edge[-1]['scoring']['ngd'] / mean_ngd)**2 / 2) for edge in edges}

        else:
            raise Exception('Method ' + method + ' has not been implemented.')

        # make sure weights on edges are not nan valued - set them to zero otherwise
        weights = {e: (weights[e] if np.isfinite(weights[e]) else 0) for e in weights}

        # set the weights on the edges
        nx.set_edge_attributes(self.graph, values=weights, name='weight')

    def get_edges_by_id(self, edge_ids, data=False):
        """Get edges by id."""
        edges = [e for e in self.graph.edges(data=True) if e[-1]['id'] in edge_ids]
        if data:
            return edges
        else:
            return [e[:2] for e in edges]

    def sum_edge_weights(self, subgraph):
        """Add edge weights."""
        # choose edges with one of the appropriate ids
        edges = self.get_edges_by_id(subgraph['edges'].values(), data=True)
        # sum their weights
        return sum([edge[-1]['weight'] for edge in edges])

    def rank(self, subgraph_list):
        """Generate a sorted list and scores for a set of subgraphs."""
        # subgraph_list is a list of lists of dicts with fields
        # 'id' and 'bound'

        if not subgraph_list:
            return ([], [])

        logger.debug('set_weights()... ')
        start = time.time()
        self.set_weights(method='ngd')
        logger.debug(f"{time.time()-start} seconds elapsed.")

        # add the none node to G
        if 'None' in self.graph:
            logger.error("Node 'None' already exists in G.\n" +
                         "This could cause incorrect ranking results.")
        else:
            self.graph.add_node('None')   # must add the none node to correspond to None id's

        logger.debug("Prescreening subgraph_list... ")
        start = time.time()
        prescreen_scores = [self.sum_edge_weights(sg) for sg in subgraph_list]
        prescreen_sorting, prescreen_scores_sorted = zip(*sorted(enumerate(prescreen_scores), key=lambda elem: elem[1], reverse=True))

        if len(prescreen_sorting) > self.prescreen_count:
            prescreen_sorting = prescreen_sorting[0:self.prescreen_count]
            prescreen_scores_sorted = prescreen_scores_sorted[0:self.prescreen_count]

        subgraph_list = [subgraph_list[i] for i in prescreen_sorting]
        logger.debug(f"{time.time()-start} seconds elapsed.")

        logger.debug("Calculating subgraph statistics()... ")
        start = time.time()

        graph_stat = [self.subgraph_statistic(sg, metric_type='mix') for sg in subgraph_list]
        logger.debug(f"{time.time()-start} seconds elapsed.")

        graph_comparison = self.comparison_statistic(subgraph_list, metric_type='mix')
        logger.debug(f"Comparison graph statistic: {graph_comparison}")

        # larger hitting/mixing times are worse
        ranking_scores = [graph_comparison / s if s > 0 else -1 for s in graph_stat]

        # Fail safe to nuke nans
        ranking_scores = [r if np.isfinite(r) else -1 for r in ranking_scores]

        ranking_sorting, _ = zip(*sorted(enumerate(ranking_scores), key=lambda elem: elem[1], reverse=True))
        subgraph_list = [subgraph_list[i] for i in ranking_sorting]

        subgraph_scores = [ranking_scores[i] for i in ranking_sorting]

        # trim output
        if len(subgraph_list) > self.output_count:
            subgraph_list = subgraph_list[:self.output_count]
            subgraph_scores = subgraph_scores[:self.output_count]

        return (subgraph_scores, subgraph_list)

    def report_ranking(self, subgraph_list):
        """Report ranking."""
        # construct the output that question.py expects

        (subgraph_scores, subgraph_list) = self.rank(subgraph_list)

        # add extra computed metadata in self.graph to subgraph for display
        logger.debug("Extracting subgraphs... ")
        start = time.time()
        subgraphs_meta = [self.graph.subgraph([s for s in subgraph['nodes'].values()]) for subgraph in subgraph_list]
        logger.debug(f"{time.time()-start} seconds elapsed.")

        report = []
        for i, subgraph in enumerate(subgraphs_meta):

            # re-sort the nodes in the sub-graph according to the node_list
            # and remove None nodes
            node_list = subgraph_list[i]['nodes'].values()
            nodes = list(subgraph.nodes(data=True))
            ids = [n[0] for n in nodes]
            nodes = [nodes[ids.index(n)][-1] for n in node_list if n != 'None']

            edges = list(subgraph.edges(data=True))
            edges = [e[-1] for e in edges]

            sgr = dict()
            sgr['score'] = subgraph_scores[i]
            sgr['nodes'] = nodes
            sgr['edges'] = edges

            report.append(sgr)

        return (report, subgraph_list)

    def subgraph_statistic(self, subgraph, metric_type='hit'):
        """Compute subgraph statistic?"""
        laplacian = self.graph_laplacian(subgraph)
        if metric_type == 'hit':
            return hitting_time_from_laplacian(laplacian)
        elif metric_type == 'mix':
            return mixing_time_from_laplacian(laplacian)
        else:
            raise ValueError(f'Unknown metric type "{metric_type}"')

    def comparison_statistic(self, subgraph_list, metric_type='hit'):
        """Create a prototypical graph to compare hitting times against.

        For now, we use a fully connected graph with weights 1/2
        """
        weight = 0.5
        num_nodes = int(np.round(np.mean([len(s) for s in subgraph_list])))
        num_nodes = max(num_nodes, 2)
        laplacian = weight * (num_nodes * np.eye(num_nodes) - np.ones((num_nodes, num_nodes)))

        if metric_type == 'hit':
            return hitting_time_from_laplacian(laplacian)
        elif metric_type == 'mix':
            return mixing_time_from_laplacian(laplacian)
        else:
            raise ValueError(f'Unknown metric type "{metric_type}"')

    def graph_laplacian(self, subgraph):
        """Generate graph Laplacian."""
        # subgraph is a list of dicts with fields 'id' and 'bound'

        # get updated weights
        edges = self.get_edges_by_id(subgraph['edges'].values(), data=True)
        # edge_subgraph is busted, this avoids the issue:
        edges = [(e[0], e[1], f"{e[2]['source_database']}:{e[2]['type']}") for e in edges]
        subgraph_update = self.graph.edge_subgraph(edges)
        node_ids = list(self.graph.nodes)

        nodes = subgraph_update.nodes(data=True)

        # get updated weights
        subgraph_update = subgraph_update.to_undirected()

        # compute graph laplacian for this case with potentially duplicated
        # nodes (None may be duplicated)
        num_nodes = len(nodes)
        laplacian = np.zeros((num_nodes, num_nodes))
        index = {id: node_ids.index(id) for id in node_ids}
        for source_id, target_id, weight in subgraph_update.edges(data='weight'):
            if source_id is not target_id and (source_id in node_ids) and (target_id in node_ids):
                i, j = index[source_id], index[target_id]
                laplacian[i, j] = -weight
                laplacian[j, i] = -weight
                laplacian[i, i] = laplacian[i, i] + weight
                laplacian[j, j] = laplacian[j, j] + weight

        # add teleportation to allow leaps of faith
        laplacian = laplacian + self.teleport_weight * (num_nodes * np.eye(num_nodes) - np.ones((num_nodes, num_nodes)))
        return laplacian


def hitting_time_from_laplacian(laplacian):
    """Compute hitting time from Laplacian."""
    # assume L is square
    laplacian[-1, :] = 0
    laplacian[-1, -1] = 1
    b = np.zeros(laplacian.shape[0])
    b[0] = 1
    return np.linalg.solve(laplacian, b)[1]


def mixing_time_from_laplacian(laplacian):
    """Compute mixing time from Laplacian."""
    # assume L is square
    n = laplacian.shape[0]
    g = max(np.diag(laplacian))
    if g < 1e-8 or not np.isfinite(g):
        return -1

    # uniformitization into discrete time chain
    P = np.eye(n) - laplacian / g / 2
    try:  # always put other people's code inside a try catch loop
        eigvals = numpy.linalg.eigvals(P)
    except Exception:
        # this should never happen for nonzero teleportation
        logger.debug("Eigenvalue computation failed for P:")
        for i in range(n):
            logger.debug(P[i, :])
        return -1

    eigvals = np.abs(eigvals)
    eigvals.sort()  # sorts ascending

    return 1 / (1 - eigvals[-2]) / g
