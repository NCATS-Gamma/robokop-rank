# ROBOKOP Rank - The reasoning engine for ROBOKOP

[ROBOKOP](http://github.com/NCATS-Gamma/robokop) is a system for reasoning over knowledge oriented pathways. This is the reasoning system responsible for returning a ranked list of graphs that are relevant to a user defined question. 

Generating and ranking answers is composed of three parts: Generating answers, converting answers into weighted graphs, and then ranking the weighted graphs based upon their connectivity.

Installation instructions can be found here http://github.com/NCATS-Gamma/robokop

For a public instance of the API see

* http://robokop.renci.org:6011/apidocs/

## Answer Generation
A list of potential answers are generated using a cypher query on a neo4j database. The cypher query is constructed based on the user defined input question specification. Each answer corresponds to a subgraph within the larger graph contained in the NEO4j instance. Each potential answer graph is then converted into a networkx 1.11 graph and the nodes and edges are formated to contain metadata such as relevant publications.

## Weight Generation
Each answer is comprised of a series of nodes and edges. The goal of this step is convert meta-data into numbers on edges corresponding to our confidence there is a good link between the two concepts that edge links.

Each edge and node contains metadata returned by the knowledge sources. We have two methods for converting edge metadata into weights: logistic, and ngd. The default is ngd.

### Logistic weights
The logistic weights are computed from the number of publications along each edge. The pub_counts are then scaled by the following logistic

    1/(1 + exp((5 - pub_counts)/2))

This logistic function returns a nonzero value for zero pub_counts, as some edges which have come from some knowledge sources do not contain lists of publications.

### Normalized Google Distance (NGD)
Recently, the omnicorp knowledge source has returned us publication counts for each node. It is important to take this into consideration, as edges whose nodes have a large number of publications may by chance have a large number of publications on the edge, even when the two concepts are weakly related.

We use the formula described on the wikipedia page for normalized google distance.
<https://en.wikipedia.org/wiki/Normalized_Google_distance>

The omni-corp node information does not always return a non-zero number of article counts for each node. For this reason, we set a minimum article node count of 1000, and assume that any concept in the knowledge sources have at least some minimum number of associated publications. If an article count is not given for a node, we assume it is a relatively large number, 25000.

The publications along each edge may also be zero, which does not agree with the log function. For this reason, we add 1 to the publication counts.

We use N = 1e8, which roughly corresponds to the order of magnitude of the number of articles * 10 keywords/article.

Once the normalized google distances are computed, we wish to convert them into a similarity. Distances return small values for similar objects, and we wish to have large values corresponding to similar objects. For this reason, we use a gaussian kernel with mean zero, and std. corresponding to the average google distance over all the returned edges. This results in a similarity as requested.

## Answer Ranking
The answer graphs sent to the ranker have 'support' edges which link any two concepts in the graph. These support edges must be taken into account in the ranking, as graphs with many support edges are likely to be better answers. 

### Prescreening
In order to save computational time, the graphs are first sent through a prescreening stage, and culled to the top 2000 answers for further evaluation. Sometimes for larger graphs, tens of thousands of potential answers are returned by the cypher query.

### Graph Statistics
In order to rank the graphs, we wish to analyze the graph adjacency matrix as a whole, and return a single number dictating the graph's connectivity as a whole. We have code which can compute two different quantities of interest: hitting times, and mixing times. The default graph statistic to rank on is the mixing time.

In order to make the graphs always well behaved, we allow for a small probability of teleporting between any two nodes. This is a standard procedure to allow for regularizing graphs with little impact upon the final result, and does a good job of handling graphs with disconnected components. Choosing the teleportation probability very small will force a low score on disconnected graphs. It also has the benefit of making all graphs connected and acyclic.

#### Hitting times
The hitting time is the average time a random walker on the graph spends to get from one node to another node. The graphs we have evaluated thus far have a clear 'start' and 'stop' node, and so we compute the hitting time from the first node to the last node - this assumption is a major drawback to the hitting time method for more general graphs.

Computing the hitting time requires solving a linear system, which is O(n^2) for the small matrices we typically see.

#### Mixing times
The mixing time is the amount of time before any initial distribution on the graph is 'close' to the stationary distribution. Random walks on connected, acyclic graphs have probability distributions which converge exponentially fast to the stationary distribution. The rate at which this exponential convergence is directly related to the mixing time. 

The mixing time can be computed from the eigenvalues of the transition matrix. The first eigenvalue of the transition matrix is always 1, and corresponds to the stationary distribution. The second eigenvalue controls the rate at which the markov chain converges to the stationary distribution.

Computing the full eigenvalue decomposition is O(n^3), although since we only care about the top two eigenvalues, it could one day be reduced to roughly O(n^2).

### Ranking scores
Once the graph statistics (hitting or mixing times) are computed for each graph, we wish to come up with score for each graph which does not depend upon the number of nodes in the answer graphs. We generate a graph to compare to, which is a fully connected graph of size n, where n is the average number of nodes in the answers. The transition weights on the graph are set to 1/2. The final score for each answer is the comparison graph statistic divided by the graph statistic. This results in larger ranking scores for graphs with smaller hitting/mixing times - meaning they are better connected. 
