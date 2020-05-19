import time
import networkx as nx
from tqdm import tqdm
from IC import IC

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