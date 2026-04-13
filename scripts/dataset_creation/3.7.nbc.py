#The purpose of this code is to generate the network centrality metrics fo the bipartite networks for each year
from pyprojroot import here
import pandas as pd
import numpy as np
import igraph as ig
import sys


processed_data = here('data/processed_data')
bipartite_data = here('data/processed_data/net_bipartite_data')


def create_bipartite(edge_list_df):
    # Unique nodes for each set
    pu_nodes = edge_list_df['purchasing_unit_id'].unique()
    supplier_nodes = edge_list_df['supplier_name_clean'].unique()

    # Combine into one list of all nodes
    all_nodes = list(pu_nodes) + list(supplier_nodes)

    # Create mapping from name to id and vice versa
    name_to_id = {name: idx for idx, name in enumerate(all_nodes)}

    # Create edge list with weights
    edges = [(name_to_id[pu], name_to_id[supplier]) for pu, supplier in zip(edge_list_df['purchasing_unit_id'], edge_list_df['supplier_name_clean'])]
    weights = edge_list_df['weight'].tolist()

    # Build graph
    g = ig.Graph(edges=edges)
    g.vs["name"] = all_nodes
    g.es["weight"] = weights

    # Optionally: Set bipartite attribute (True for purchasing units, False for suppliers)
    g.vs["type"] = [True if name in pu_nodes else False for name in all_nodes]

    return g, name_to_id


def weighted_degree(graph: ig.Graph, weights='weight'):
    """
    Computes the weighted degree k'_i = sqrt(degree * strength)
    for each node, assuming alpha = beta = 1.

    Parameters:
        graph (igraph.Graph): The input graph.
        weight_attr (str): Edge attribute to use as weight.
        mode (str): 'ALL', 'IN', or 'OUT' (for directed graphs).

    Returns:
        list: Weighted degrees for all nodes.
    """
    degrees = graph.degree()
    strengths = graph.strength(weights=weights)
    wd = [np.sqrt(k * s) for k, s in zip(degrees, strengths)]
    wd = list(np.round(wd, 0).astype(int))
    return wd

def average_degree(graph: ig.Graph, weights='weight'):
    """
    Computes the weighted degree k'_i = sqrt(degree * strength)
    for each node, assuming alpha = beta = 1.

    Parameters:
        graph (igraph.Graph): The input graph.
        weight_attr (str): Edge attribute to use as weight.
        mode (str): 'ALL', 'IN', or 'OUT' (for directed graphs).

    Returns:
        list: Weighted degrees for all nodes.
    """
    degrees = graph.degree()
    strengths = graph.strength(weights=weights)
    ad = [(s / k) if k != 0 else 0 for k, s in zip(degrees, strengths)]
    ad = list(np.round(ad, 0).astype(int))
    return ad

def custom_k_shell_decomposition(graph, name_func, func):
    """
    Performs k-shell decomposition based on a weighted degree function.

    Parameters:
        graph (igraph.Graph): The input graph.
        weighted_degree_func (callable): A function that takes a graph and returns
                                         a list of weighted degrees (one per vertex index).

    Returns:
        list: k-shell/core number for each original node (by index)
    """
    #initialize variables
    g = graph.copy()
    coreness = {}

    if name_func in ['wdeg', 'ad']:
        g.vs['attibute'] = func(g, weights = 'weight')
    elif name_func in ['strength']:
        g.vs['attibute'] = g.strength(weights = 'weight')
    else:
        print("Invalid function name. Please use 'wdeg', 'ad', or 'strength'.")
        return None

    current_shell = 1

    while g.vcount() > 0:
        to_remove_names = [v['name'] for v in g.vs if v['attibute'] <= current_shell]
        to_remove_indexes = [v.index for v in g.vs if g.vs['attibute'][v.index] <= current_shell]
        prov_dict = {key: current_shell for key in to_remove_names}
        coreness = coreness | prov_dict
        g.delete_vertices(to_remove_indexes)
        
        if name_func in ['wdeg', 'ad']:
            g.vs['attibute'] = func(g, weights = 'weight')
        elif name_func in ['strength']:
            g.vs['attibute'] = g.strength(weights = 'weight')

        try:
            if min(g.vs['attibute']) > current_shell:
                current_shell += 1
        except:
            break
    
    return coreness

############################################ LOAD DATA

contracts = pd.read_feather(processed_data / 'mxc11to22_base.feather', columns=['purchasing_unit_id', 'supplier_name_clean', 'contract_year'])

print('shape of contracts:', contracts.shape)

#establish year
year = int(sys.argv[1])
print('########################### year', year)

edge_list_y = contracts[contracts['contract_year'] == year]
print(edge_list_y.shape)

edge_list_y = edge_list_y.groupby(['purchasing_unit_id', 'supplier_name_clean']).size().reset_index(name='weight')
edge_list_y['purchasing_unit_id'] = edge_list_y['purchasing_unit_id'].astype(str)
edge_list_y['supplier_name_clean'] = edge_list_y['supplier_name_clean'].astype(str)
edge_list_y['weight'] = edge_list_y['weight'].astype(int)

nbuyers = edge_list_y['purchasing_unit_id'].nunique()
nsuppliers = edge_list_y['supplier_name_clean'].nunique()
ntotal = nbuyers + nsuppliers

print(
    'n buyers: ', nbuyers,
    '; n suppliers: ', nsuppliers,
    '; n total: ', ntotal
    )

#create the bipartite graph
b_y, name_to_id_y = create_bipartite(edge_list_df = edge_list_y)
edge_list_y['buyer_id'] = edge_list_y['purchasing_unit_id'].map(name_to_id_y)
edge_list_y['supplier_id'] = edge_list_y['supplier_name_clean'].map(name_to_id_y)

#################################################################### edge betweenness
edge_betweenness = b_y.edge_betweenness(directed= False, weights = 'weight')
edge_list_y['edge_betweenness'] = edge_betweenness


#################################################################### coreness of the bipartite graph
coreness_wdeg = custom_k_shell_decomposition(b_y, name_func= 'wdeg', func = weighted_degree)
coreness_ad = custom_k_shell_decomposition(graph = b_y, name_func= 'ad', func = average_degree)
coreness_strength = custom_k_shell_decomposition(graph = b_y, name_func= 'strength', func = None)
edge_list_y['corewdeg_b'] = edge_list_y['purchasing_unit_id'].map(coreness_wdeg)
edge_list_y['corewdeg_s'] = edge_list_y['supplier_name_clean'].map(coreness_wdeg)
edge_list_y['coread_b'] = edge_list_y['purchasing_unit_id'].map(coreness_ad)
edge_list_y['coread_s'] = edge_list_y['supplier_name_clean'].map(coreness_ad)
edge_list_y['corestrength_b'] = edge_list_y['purchasing_unit_id'].map(coreness_strength)
edge_list_y['corestrength_s'] = edge_list_y['supplier_name_clean'].map(coreness_strength)

edge_list_y['c_id_lg'] = edge_list_y.index
edge_list_y['contract_year'] = year

#save the bipartite data
filename_e = 'bipartite_' + str(year) + '.feather'
edge_list_y.to_feather(bipartite_data / filename_e)

