#The purpose of this code is to generate the network centrality metrics fo the projections of the networks for each year
from pyprojroot import here
import pandas as pd
import numpy as np
import igraph as ig
import sys

processed_data = here('data/processed_data')
network_data = here('data/network_data')

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

def get_projection_centralities(projection_graph, year):
    projection_size = len(projection_graph.vs)
    #get the main component for closeness
    components = projection_graph.connected_components(mode='weak')
    component_sizes = pd.Series([len(component) for component in components])
    print(component_sizes.value_counts())
    largest_component = components[np.argmax([len(component) for component in components])]

    #get centrality measures
    degree_centrality = projection_graph.degree()
    strength = projection_graph.strength(weights='weight')
    closeness = projection_graph.closeness(vertices = largest_component, normalized=True)
    closeness_d = dict(zip(largest_component, closeness))
    betweenness = projection_graph.betweenness()
    norm_betweenness = np.array(betweenness) / (((projection_size - 1) * (projection_size - 2)) / 2)
    eigen = projection_graph.eigenvector_centrality(directed=False, weights='weight', scale=False)

    #create the dataframe
    vertex_data = [(v.index, v["name"]) for v in projection_graph.vs]
    df = pd.DataFrame(vertex_data, columns=["id", "name"])
    df['degree'] = degree_centrality
    df['strength'] = strength
    df['betweenness'] = betweenness
    df['norm_betweenness'] = norm_betweenness
    df['closeness'] = df['id'].map(closeness_d)
    df['closeness'] = df['closeness'].fillna(0)
    df['eigen'] = eigen
    df['contract_year'] = year

    return df

#import dataset
contracts = pd.read_feather(processed_data / 'mxc11to22_base.feather', columns=['purchasing_unit_id', 'supplier_name_clean', 'contract_year'])

#establish year
#year = map(int, sys.argv[1])
year = int(sys.argv[1])
print('########################### year', year)

#data for that year
edge_list_y = contracts[contracts['contract_year'] == year]

#create edgelist
edge_list_y = edge_list_y.groupby(['purchasing_unit_id', 'supplier_name_clean']).size().reset_index(name='weight')
edge_list_y['purchasing_unit_id'] = edge_list_y['purchasing_unit_id'].astype(str)
edge_list_y['supplier_name_clean'] = edge_list_y['supplier_name_clean'].astype(str)
edge_list_y['weight'] = edge_list_y['weight'].astype(int)
#check the number of buyers and suppliers
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
#create the projections
g_supplier, g_buyer = b_y.bipartite_projection()

#checks
assert g_supplier.is_directed() == False, 'g_supplier is directed'
assert g_buyer.is_directed() == False, 'g_buyer is directed'
assert len(g_supplier.vs) == nsuppliers, 'g_supplier has wrong number of vertices'
assert len(g_buyer.vs) == nbuyers, 'g_buyer has wrong number of vertices'

#get the centralities
df_supplier = get_projection_centralities(g_supplier, year)
df_buyer = get_projection_centralities(g_buyer, year)

#save the data
filenames = 'centralities_supplier_' + str(year) + '.feather'
df_supplier.to_feather(network_data / filenames)
filenameb = 'centralities_buyer_' + str(year) + '.feather'
df_buyer.to_feather(network_data / filenameb)
print('done with year ', year)