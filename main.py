import networkx as nx
import pandas as pd
from tqdm import tqdm
import time
import numpy as np

# Reade re-tweet graph 
data=pd.read_csv('/home/parisa/retweet_subgraph.csv', header=None)


G=nx.DiGraph()
    
vertics=set(data[0]).union(set(data[1]))

l=len(data)
for i in tqdm(range(l)):
    G.add_edge(data[1][i], data[0][i], weight=int(data[2][i]))
    
    
    
#read isis-users 
isis_user_set=set()
f=open('/home/parisa/isis_seed.csv')
for line in f:
    isis_user_set.add(int(line.strip()))
f.close()
        
isis_user_active=set()
for user in tqdm(isis_user_set):
    if user in vertics:
        isis_user_active.add(user)
        
    
def IC(g,S,p=0.5,mc=10, steps=3):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    
    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in (range(mc)):
        
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        #while new_active:
        for step in range(steps):
            

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in (new_active):
                
                # Determine neighbors that become infected
                np.random.seed(i)
                success = np.random.uniform(0,1,len(g[node])) > p
                
                if len(g[node]) > 1:
                    new_ones += list(np.extract(success, g[node]))
                

            new_active = list(set(new_ones) - set(A))
            
            # Add newly activated nodes to the set of activated nodes
            A += new_active
            
        spread.append(len(A))
        
    return(np.mean(spread))

def celf(g,k, crowd_sourced, p=0.1,mc=1000, steps=3):  
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """
      
    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    
    # Calculate the first iteration sorted list
    start_time = time.time() 
    #marg_gain = [IC(g,[node],p,mc) for node in range(g.vcount())]
    marg_gain = [IC(g,[node],p,mc) for node in crowd_sourced]
    

    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(crowd_sourced, marg_gain), key=lambda x: x[1],reverse=True)

    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    #Q, LOOKUPS, timelapse = Q[1:], [g.vcount()], [time.time()-start_time]
    Q, timelapse = Q[1:], [time.time()-start_time]
    
    # --------------------
    # Find the next k-1 nodes using the list-sorting procedure
    # --------------------
    
    for _ in tqdm(range(k-1)):    

        check, node_lookup = False, 0
        
        while not check:
            
            # Count the number of times the spread is computed
            node_lookup += 1
            
            # Recalculate spread of top node
            current = Q[0][0]
            
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current,IC(g,S+[current],p,mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse = True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        #LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)

        # Remove the selected node from the list
        Q = Q[1:]

    #return(S,SPREAD,timelapse,LOOKUPS)
    return(S,SPREAD,timelapse)

def greedy(g,k, crowd_sourced, p=0.1,mc=1000, steps=3):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    S, spread, timelapse, start_time = [], [], [], time.time()
    
    # Find k nodes with largest marginal gain
    for _ in tqdm(range(k)):

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        #for j in set(range(g.vcount()))-set(S):
        for j in set(crowd_sourced)-set(S):

            # Get the spread
            s = IC(g,S + [j],p,mc, steps)

            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j

        # Add the selected node to the seed set
        S.append(node)
        
        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)

    return(S,spread,timelapse)

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

S_celf,spread,timelapse=celf(G,50, isis_user_active, p=0.1,mc=10, steps=3)
S_greedy,spread,timelapse=greedy(G,50, isis_user_active, p=0.1,mc=10, steps=3)
S_degreeDiscount=degreeDiscount(G, 50, isis_user_active, p=.01)


SEED=list(isis_user_active)
n=IC(G, SEED, mc=10, steps=3)
print ("influence by all isis-users: ")
print (n)


SEED=S_greedy
n=IC(G, SEED, mc=10, steps=3)
print ("influence by top 50 nodes obtained from greedy algorithm: ")
print (n)


SEED=S_celf
n=IC(G, SEED, mc=10, steps=3)
print ("influence by top 50 nodes obtained from CELF algorithm: ")
print (n)


SEED=S_degreeDiscount
n=IC(G, SEED, mc=10, steps=3)
print ("influence by top 50 nodes obtained from degreeDiscount algorithm:")
print (n)


SEED=list(isis_user_active-set(S_degreeDiscount))
n=IC(G, SEED, mc=10, steps=3)
print ("influence by all isis-users but top 50 nodes")
print (n)


#influence by top 10, 20, 30, 40
top_nodes=[10,20,30,40]
for num in top_nodes:
        
    SEED=S_degreeDiscount[:num]
    n=IC(G, SEED, mc=10, steps=3)
    print ("influence by top "+ str(num) + " nodes obtained from degreeDiscount algorithm:")
    print (n)

    SEED=list(isis_user_active-set(S_degreeDiscount[:num]))
    n=IC(G, SEED, mc=10, steps=3)
    print ("influence by all isis-users but top "+ str(num) +" nodes")
    print (n)


