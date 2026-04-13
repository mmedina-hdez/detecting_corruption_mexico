import pandas as pd
import numpy as np
from pyprojroot import here
import networkx as nx
from networkx.algorithms import bipartite
import math
from tqdm import tqdm
import itertools

processed_data = here('data/processed_data')

var2keep = ['contract_year', 'purchasing_unit_id', 'contract_price_mx', 'supplier_name_clean' ]

#read feather
contracts = pd.read_feather(processed_data / 'mxc11to22_base.feather', columns=var2keep)
print(contracts.shape)

########## Calculate buyer dependence ##################
#budget per buyer per year
contracts['spenditure_buyer_per_year'] = contracts.groupby(['contract_year', 'purchasing_unit_id' ])['contract_price_mx'].transform('sum')
#proportion contract per buyer per year
contracts['proportion_contract_buyer_year'] = contracts['contract_price_mx'] / contracts['spenditure_buyer_per_year']
#proportion of share of supplier in annual budget of buyer
contracts['buyer_dependence'] = contracts.groupby(['contract_year', 'supplier_name_clean',  'purchasing_unit_id'])['proportion_contract_buyer_year'].transform('sum')
contracts['buyer_dependence'] = contracts['buyer_dependence'].astype(float)
#remove columns we don't need
contracts = contracts.drop(columns=['proportion_contract_buyer_year', 'spenditure_buyer_per_year'])

################################################################################
########################## Buyer dependence entropy ############################
################################################################################
print(contracts.shape)
entropy_df = contracts.copy().drop(columns=['contract_price_mx'])
entropy_df = entropy_df.drop_duplicates(subset=['contract_year', 'purchasing_unit_id', 'supplier_name_clean'], keep='first')
print(entropy_df.shape)
#calculate log of buyer dependence
entropy_df['log_buyer_dependence'] = np.log(entropy_df['buyer_dependence'])
#product buyer dependence and log buyer dependence
entropy_df['product_buyer_dependence'] = entropy_df['buyer_dependence'] * entropy_df['log_buyer_dependence']
#numerator
entropy_df['entropy_buyer_dependence_numerator'] = entropy_df.groupby(['contract_year', 'purchasing_unit_id'])['product_buyer_dependence'].transform('sum')
#number of suppliers
entropy_df['number_of_suppliers'] = entropy_df.groupby(['contract_year', 'purchasing_unit_id'])['supplier_name_clean'].transform('count')
#log number of suppliers
entropy_df['log_number_of_suppliers'] = np.log(entropy_df['number_of_suppliers'])
#get entropy
entropy_df['entropy_buyer_dependence'] = (entropy_df['entropy_buyer_dependence_numerator'] / entropy_df['log_number_of_suppliers'])*-1
#fix inf and nan
entropy_df['entropy_buyer_dependence'] = np.where(entropy_df['number_of_suppliers'] == 1, 0, entropy_df['entropy_buyer_dependence'])
#drop columns we don't need
entropy_df = entropy_df.drop(columns=['log_buyer_dependence', 'product_buyer_dependence', 'entropy_buyer_dependence_numerator', 'log_number_of_suppliers', 'number_of_suppliers', 'buyer_dependence', 'supplier_name_clean'])
#keep only purchasing unit and year levels
entropy_df= entropy_df.drop_duplicates(subset=['contract_year', 'purchasing_unit_id'], keep='first').reset_index(drop=True)


####################################################################################################
########################## Unweighted Competitive Clustering #######################################
####################################################################################################

ucc_buyers = contracts.copy().drop(columns=['buyer_dependence', "contract_price_mx"])
print(ucc_buyers.shape)
# get number of contracts per buyer per supplier
ucc_buyers = ucc_buyers.groupby(['contract_year', 'purchasing_unit_id', 'supplier_name_clean']).size().reset_index(name = 'ncontracts')
print(ucc_buyers.shape)


#function to calculate paths length 3

def count_paths(G, nodelist):

    p3 = []

    for node in tqdm(nodelist, desc='buyer nodes'):
        subgraph = nx.ego_graph(G, node, radius=3)
        all_nodes = list(subgraph.nodes())
        all_nodes.remove(node)

        simple_paths = nx.all_simple_paths(subgraph, source=node, target=all_nodes, cutoff=3)

        count = 0

        for path in simple_paths:
            if len(path) == 4:
                count += 1
        
        p3.append(count)

    return nodelist, p3

def count_cycles4(G, nodelist):
    
    #get projection of the network
    G = nx.bipartite.weighted_projected_graph(G, nodelist)
    #get adjacency matrix
    G_m = nx.to_numpy_array(G, nodelist = nodelist)
    G_m[G_m == 1] = 0
    vectorized_comb = np.vectorize(lambda n: math.comb(int(n), 2))
    G_m = vectorized_comb(G_m)
    #multiply entries by 2
    G_m = G_m * 2
    #geometric mean
    c4 = G_m.sum(axis=1).tolist()

    return nodelist, c4

year_list = []
purchasing_unit_id_list_p3 = []
purchasing_unit_id_list_c4 = []
p3_list = []
c4_list = []

for i in tqdm(ucc_buyers['contract_year'].unique(), desc='Years'):
    df = ucc_buyers[ucc_buyers['contract_year'] == i].reset_index(drop=True)
    #create graph
    G = nx.Graph()
    G = nx.from_pandas_edgelist(df, 'supplier_name_clean', 'purchasing_unit_id', edge_attr= ['ncontracts'])
    # Add bipartite attribute to nodes
    G.add_nodes_from(df['purchasing_unit_id'], bipartite=0)
    G.add_nodes_from(df['supplier_name_clean'], bipartite=1)
    #undirected
    G = G.to_undirected()
    #buyers node list
    buyer_nodelist = list(df['purchasing_unit_id'].unique())


    #Calculate paths of length 3
    print('Calculating paths of length 3')
    nodelist_p3, p3 = count_paths(G, buyer_nodelist)
    #cycles 4
    print('Calculating cycles of length 4')
    nodelist_c4, c4 = count_cycles4(G, buyer_nodelist)

    
    #save information
    len_list = len(buyer_nodelist)
    #repeat year
    year_list = year_list + ([i] * len_list)
    #save purchasing unit id - p3
    purchasing_unit_id_list_p3 = purchasing_unit_id_list_p3 + nodelist_p3
    #save purchasing unit id - c4
    purchasing_unit_id_list_c4 = purchasing_unit_id_list_c4 + nodelist_c4
    #save p3
    p3_list = p3_list + p3
    #save c4
    c4_list = c4_list + c4

ucc = pd.DataFrame({'contract_year': year_list, 
                    'purchasing_unit_id_p3': purchasing_unit_id_list_p3, 
                    'purchasing_unit_id_c4' :  purchasing_unit_id_list_c4 , 
                    'p3': p3_list, 
                    'c4': c4_list})
ucc['p3'] = ucc['p3'].astype(float)
ucc['c4'] = ucc['c4'].astype(float)
ucc['contract_year'] = ucc['contract_year'].astype(int)
#calculate competitive clustering
ucc['unweighted_competitive_clustering'] = ucc['c4'] / ucc['p3']
ucc = ucc.drop(columns=['purchasing_unit_id_p3'])
ucc.rename(columns={'purchasing_unit_id_c4': 'purchasing_unit_id'}, inplace=True)
ucc.to_feather(processed_data / 'unweighted_competitive_clustering.feather')

 

####################################################################################################
########################## Weighted Competitive Clustering #######################################
####################################################################################################

wcc_buyers = contracts.copy().drop(columns=['buyer_dependence'])
print(wcc_buyers.shape)
wcc_buyers = wcc_buyers.groupby(['contract_year', 'purchasing_unit_id', 'supplier_name_clean']).agg(contracts_total_value = ('contract_price_mx', 'sum') ).reset_index()
print(wcc_buyers.shape)

year_list = []
pu_id1_list = []
pu_id2_list = []
weights_pairwise_list = []
geo_mean_pairwise_list = []


for i in tqdm(wcc_buyers['contract_year'].unique(), desc='Years'):
    df = wcc_buyers[wcc_buyers['contract_year'] == i].reset_index(drop=True)

    ###################### Edge list with common nodes
    #get projection of the network
    #create graph
    G = nx.Graph()
    G = nx.from_pandas_edgelist(df, 'supplier_name_clean', 'purchasing_unit_id')
    # Add bipartite attribute to nodes
    G.add_nodes_from(df['purchasing_unit_id'], bipartite=0)
    G.add_nodes_from(df['supplier_name_clean'], bipartite=1)
    #undirected
    G = G.to_undirected()
    buyer_nodelist = list(df['purchasing_unit_id'].unique())
    G = nx.bipartite.weighted_projected_graph(G, buyer_nodelist)
    G = pd.DataFrame(G.edges(data=True), columns=['purchasing_unit_id', 'purchasing_unit_id2', 'weight'])
    G['weight'] = G['weight'].apply(lambda x: x['weight'])
    G = G[G['weight'] > 1].reset_index(drop=True)
    print('The shape of edges list of projection graph in year', i, ' is ', G.shape)

    pu_id1_list = pu_id1_list + G['purchasing_unit_id'].tolist()
    pu_id2_list = pu_id2_list + G['purchasing_unit_id2'].tolist()

    ###################### Matrix with buyer dependence
    matrix_i = df.pivot(index='supplier_name_clean', columns='purchasing_unit_id', values='contracts_total_value')

    ###################### Calculate geometric mean between two buyers

    gm_list = []
    for j in tqdm(range(len(G)), desc='Buyer pairs'):
        year_list.append(i)
        nodes2search = [G['purchasing_unit_id'][j], G['purchasing_unit_id2'][j]]
        matrix_j = matrix_i[nodes2search].dropna()
        matrix_j['product'] = matrix_j.prod(axis=1)
        matrix_j['max'] = matrix_j.max(axis=1)
        
        product_edges = matrix_j['product'].tolist() # list of the product of edges that connect the same supplier
        max_edges = matrix_j['max'].tolist()
        
        combinations_products = itertools.combinations(product_edges, 2)
        combinations_max = itertools.combinations(max_edges, 2)
        products = [a * b for a, b in combinations_products] #products of cycles edges
        maxs = [max(a, b) for a, b in combinations_max]
        maxs = np.array(maxs)**(4)
        weighted_products = np.array(products) / maxs
        geo_mean = np.array(weighted_products)**(1/4)
        geo_mean = geo_mean*2 #multiply by 2 because the cycles are reciprocal
        geo_mean = geo_mean.sum()
        geo_mean
        gm_list.append(geo_mean)

    geo_mean_pairwise_list = geo_mean_pairwise_list + gm_list

    print('I finished year:', i)    


geometrical_mean_pairwise_df = pd.DataFrame({'contract_year': year_list, 'purchasing_unit_id': pu_id1_list, 'purchasing_unit_id2': pu_id2_list, 'geometrical_mean_pairwise': geo_mean_pairwise_list})
#save as feather 
geometrical_mean_pairwise_df.to_feather(processed_data / 'geometrical_mean_pairwise.feather')

#i'm not sure if the order of the edges is unique, so for example if A - B is the same as B - A
#if i calculate the geometric mean based on one column, i would be maybe be losing information
#therefore i will add inverted edges to the dataframe
geometrical_mean_pairwise_df_inverted = geometrical_mean_pairwise_df.copy()
geometrical_mean_pairwise_df_inverted.columns = ['contract_year', 'purchasing_unit_id2', 'purchasing_unit_id', 'geometrical_mean_pairwise']
geometrical_mean_pairwise_df_inverted.head()

geometrical_mean_per_buyer = pd.concat([geometrical_mean_pairwise_df, geometrical_mean_pairwise_df_inverted]).reset_index(drop=True)
print(geometrical_mean_per_buyer.shape)
print(geometrical_mean_per_buyer.drop_duplicates(subset=['contract_year', 'purchasing_unit_id', 'purchasing_unit_id2'], keep='first').shape)
print(geometrical_mean_per_buyer.drop_duplicates(subset=['contract_year', 'purchasing_unit_id', 'purchasing_unit_id2', 'geometrical_mean_pairwise'], keep='first').shape)

geometrical_mean_per_buyer = geometrical_mean_per_buyer.drop_duplicates(subset=['contract_year', 'purchasing_unit_id', 'purchasing_unit_id2'], keep='first').reset_index(drop=True)
geometrical_mean_per_buyer.shape

geometrical_mean_per_buyer = geometrical_mean_per_buyer.groupby(['contract_year', 'purchasing_unit_id']).agg(allcycles_summed_geomeans = ('geometrical_mean_pairwise', 'sum')).reset_index()
geometrical_mean_per_buyer['contract_year'] = geometrical_mean_per_buyer['contract_year'].astype(int)
#save as feather
geometrical_mean_per_buyer.to_feather(processed_data / 'geometrical_mean_per_buyer.feather')

print(ucc.shape)
print(geometrical_mean_per_buyer.shape)
ucc_wcc = pd.merge(ucc, geometrical_mean_per_buyer, on=['contract_year', 'purchasing_unit_id'], how='left')
ucc_wcc['unbounded_weighted_competitive_clustering'] = ucc_wcc['allcycles_summed_geomeans'] * ucc_wcc['unweighted_competitive_clustering']
ucc_wcc['bounded_weighted_competitive_clustering'] = ucc_wcc['allcycles_summed_geomeans'] / ucc_wcc['p3']
#save as feather
ucc_wcc.to_feather(processed_data / 'weighted_and_unweighted_competitive_clustering.feather')
print(ucc_wcc.shape)

####################merge

fazekaswachs2020 = pd.merge(entropy_df, ucc_wcc, on=['contract_year', 'purchasing_unit_id'], how='left')
#save as feather
fazekaswachs2020.to_feather(processed_data / 'fazekaswachs2020.feather')