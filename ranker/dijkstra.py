"""Dijkstra's algorithm.

Adapted from:
https://gist.github.com/kachayev/5990802
"""

from collections import defaultdict
from heapq import heappop, heappush


def dijkstra(edges, source):
    """Evaluate minumum distance from source node using Dijkstra's algorithm.

    `edges` are (n1, n2, cost), undirected
    """
    g = defaultdict(list)
    for left, right, cost in edges:
        g[left].append((cost, right))
        g[right].append((cost, left))

    # Mark all nodes unvisited. Create a set of all the unvisited nodes called the unvisited set.
    seen = set()
    # Assign to every node a tentative distance value: set it to zero for our initial node and to infinity for all other nodes.
    if not isinstance(source, list):
        source = [source]
    mins = {s: 0 for s in source}
    # Set the initial node as current.
    q = [(0, s) for s in source]  # (distance, node)

    while q:
        # Select the unvisited node that is marked with the smallest tentative distance, set it as the new "current node", and go back to step 3.
        (cost, v1) = heappop(q)
        if v1 not in seen:
            # When we are done considering all of the unvisited neighbors of the current node, mark the current node as visited and remove it from the unvisited set.
            seen.add(v1)

            # For the current node, consider all of its unvisited neighbors.
            for c, v2 in g.get(v1, ()):
                if v2 in seen:
                    continue
                # Calculate their tentative distances through the current node.
                candidate = cost + c
                # Compare the newly calculated tentative distance to the current assigned value and assign the smaller one.
                incumbent = mins.get(v2, None)
                if incumbent is None or candidate < incumbent:
                    mins[v2] = candidate
                    heappush(q, (candidate, v2))

    return mins
