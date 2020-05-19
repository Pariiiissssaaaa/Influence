import numpy as np
import networkx as nx
from tqdm import tqdm


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
