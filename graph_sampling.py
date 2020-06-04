import networkx as nx
import pandas as pd
from tqdm import tqdm
import numpy as np

import glob
import csv


edge_files_10=glob.glob("/data/2/daily-edges/201510*")

isis_user=spark.read.csv('/data/reported_isis_ids_status20190828.txt', sep='\t').toDF( "id", "status").filter("status='suspended'").drop("status")
isis_user_set=set(isis_user.select('id').rdd.map(lambda x: int(x[0])).take(23887))

def sample_sub_graph(g,S, steps=3):
    """
    Input:  graph object, set of seed nodes
    Output: nodes in sampled graph
    """
    
        
    # Simulate propagation process      
    new_active, A = S[:], S[:]
 
    for step in range(steps):


        # For each newly active node, find its neighbors that become activated
        new_ones = []
        for node in (new_active):

            # Determine neighbors
            new_ones += list(g[node])

        new_active = list(set(new_ones) - set(A))

        # Add newly activated nodes to the set of activated nodes
        A += new_active
            
 
    return A

for file in edge_files_10:
    data=pd.read_csv(file, header=None)
    G=nx.DiGraph()
    vertics=set(data[0]).union(set(data[1]))
    l=len(data)
    for i in tqdm(range(1, l)):
        G.add_edge(data[1][i], data[0][i], weight=int(data[2][i]))
    
    
    isis_active=[user for user in isis_user_set if user in vertics]
    Seed=list(isis_active)
    
    sampled_nodes=sample_sub_graph(G, Seed, 2)
    data_sampled = data.loc[(data[0].isin(sampled_nodes))| data[1].isin(sampled_nodes)] 
    data_sampled.to_csv('daily_subgraphs/subGraph_'+file.split('/')[4], header=None, index=False)

    
    