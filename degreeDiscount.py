''' Implementation of degree discount heuristic [1] for Independent Cascade model
of influence propagation in graph G

[1] -- Wei Chen et al. Efficient influence maximization in Social Networks (algorithm 4)
'''
__author__ = 'ivanovsergey'
from priorityQueue import PriorityQueue as PQ # priority queue

def degreeDiscount(G, k, crowd_sourced, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (with priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    S = []
    dd = PQ() # degree discount
    t = dict() # number of adjacent vertices that are in S
    d = dict() # degree of each vertex

    # initialize degree discount
    for u in G.nodes():
        d[u] = sum([G[u][v]['weight'] for v in G[u]]) # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd.add_task(u, -d[u]) # add degree of each node
        t[u] = 0

    # add vertices to S greedily
    #for i in tqdm(range(k)):
    while (len(S)<k):
        u, priority = dd.pop_item() # extract node with maximal degree discount
        if u in crowd_sourced:
            S.append(u)

            for v in G[u]:

                if v not in S:

                    t[v] += G[u][v]['weight'] # increase number of selected neighbors

                    priority = d[v] - 2*t[v] - (d[v] - t[v])*t[v]*p # discount of degree
                    dd.add_task(v, -priority)
    return S