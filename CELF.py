import time
import networkx as nx
from tqdm import tqdm
from IC import IC

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