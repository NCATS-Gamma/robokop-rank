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
    output_count = 250

    def __init__(self, graph=None):
        """Create ranker."""
        self.graph = graph
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

    def get_edges_by_id(self, edge_ids):
        """Get edges by id."""
        edges = [e for e in self.graph['edges'] if e['id'] in flatten_semilist(edge_ids)]
        return edges

    def sum_edge_weights(self, subgraph):
        """Add edge weights."""
        # choose edges with one of the appropriate ids
        edges = self.get_edges_by_id(subgraph['edges'].values())
        # sum their weights
        return sum([edge['weight'] for edge in edges])

    def prescreen(self, subgraph_list):
        """Prescreen subgraphs.
        
        Keep the top self.prescreen_count, by their total edge weight.
        """
        logger.debug(f'Getting {len(subgraph_list)} prescreen scores...')
        prescreen_scores = [self.sum_edge_weights(sg) for sg in subgraph_list]

        logger.debug('Getting top N...')
        prescreen_sorting = [x[0] for x in heapq.nlargest(self.prescreen_count, enumerate(prescreen_scores), key=operator.itemgetter(1))]

        logger.debug('Returning sorted results...')
        return [subgraph_list[i] for i in prescreen_sorting]

    def rank(self, subgraph_list):
        """Generate a sorted list and scores for a set of subgraphs."""

        if not subgraph_list:
            return ([], [])

        # add weights to edges
        logger.debug('Setting weights... ')
        start = time.time()
        self.set_weights(method='ngd')
        logger.debug(f"{time.time()-start} seconds elapsed.")

        # prescreen
        logger.debug("Prescreening subgraph_list... ")
        start = time.time()
        subgraph_list = self.prescreen(subgraph_list)
        logger.debug(f"{time.time()-start} seconds elapsed.")

        # get subgraph statistics
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

        # sort by scores
        ranking_sorting = argsort(ranking_scores, reverse=True)
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
        logger.debug(f"{time.time()-start} seconds elapsed.")

        report = []
        for i, subgraph in enumerate(subgraph_list):
            nodes = [n for n in self.graph['nodes'] if n['id'] in flatten_semilist(subgraph['nodes'].values())]
            edges = [e for e in self.graph['edges'] if e['id'] in flatten_semilist(subgraph['edges'].values())]

            sgr = {
                'nodes': nodes,
                'edges': edges,
                'score': subgraph_scores[i]
            }

            report.append(sgr)

        return (report, subgraph_list)

    def subgraph_statistic(self, subgraph, metric_type='mix'):
        """Compute subgraph statistic?"""
        laplacian = self.graph_laplacian(subgraph)
        if metric_type == 'mix':
            return mixing_time_from_laplacian(laplacian)
        else:
            raise ValueError(f'Unknown metric type "{metric_type}"')

    def comparison_statistic(self, subgraph_list, metric_type='mix'):
        """Create a prototypical graph to compare mixing times against.

        For now, we use a fully connected graph with weights 1/2
        """
        weight = 0.5
        num_nodes = int(np.round(np.mean([len(s) for s in subgraph_list])))
        num_nodes = max(num_nodes, 2)
        laplacian = weight * (num_nodes * np.eye(num_nodes) - np.ones((num_nodes, num_nodes)))

        if metric_type == 'mix':
            return mixing_time_from_laplacian(laplacian)
        else:
            raise ValueError(f'Unknown metric type "{metric_type}"')

    def graph_laplacian(self, subgraph):
        """Generate graph Laplacian."""
        # subgraph is a list of dicts with fields 'id' and 'bound'

        node_map = {}
        for key in subgraph['nodes']:
            value = subgraph['nodes'][key]
            if isinstance(value, str):
                node_map[value] = key
            else:
                for v in value:
                    node_map[v] = key

        # get updated weights
        edges = self.get_edges_by_id(subgraph['edges'].values())
        edge_map = {e['id']: e for e in edges}
        edges = []
        for edge_thing in subgraph['edges'].values():
            if isinstance(edge_thing, int) or isinstance(edge_thing, str):
                first_edge = edge_map[edge_thing]
                edge = {
                    'weight': first_edge['weight'],
                    'source_id': node_map[first_edge['source_id']],
                    'target_id': node_map[first_edge['target_id']]
                }
            else:
                edge_thing.sort()
                first_edge = edge_map[edge_thing[0]]
                edge = {
                    'weight': sum([edge_map[e]['weight'] for e in edge_thing]),
                    'source_id': node_map[first_edge['source_id']],
                    'target_id': node_map[first_edge['target_id']]
                }
            edges.append(edge)

        node_ids = list({e['source_id'] for e in edges}.union({e['target_id'] for e in edges}))

        # compute graph laplacian for this case with potentially duplicated nodes
        num_nodes = len(node_ids)
        laplacian = np.zeros((num_nodes, num_nodes))
        index = {id: node_ids.index(id) for id in node_ids}
        for e in edges:
            source_id, target_id, weight = e['source_id'], e['target_id'], e['weight']
            if source_id is not target_id and (source_id in node_ids) and (target_id in node_ids):
                i, j = index[source_id], index[target_id]
                laplacian[i, j] = -weight
                laplacian[j, i] = -weight
                laplacian[i, i] = laplacian[i, i] + weight
                laplacian[j, j] = laplacian[j, j] + weight

        # add teleportation to allow leaps of faith
        laplacian = laplacian + self.teleport_weight * (num_nodes * np.eye(num_nodes) - np.ones((num_nodes, num_nodes)))
        return laplacian


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
