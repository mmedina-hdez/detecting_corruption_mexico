import sys
from pyprojroot import here
sys.path.append(str(here() / 'methods')) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import json

processed_data = here('data/processed_data')
supplementary_data = here('data/processed_data/supplementary_data/')

identifiers = [
    'file_code',
    'contract_code',
    'supplier_name_clean',
    'purchasing_unit_id',
    'contract_year',
]

# Load the contracts data
df = pd.read_feather(processed_data / 'contracts2ml.feather')#, columns=contract_level_cols)
# Remove the sanctioned hypothesis columns
keywords = ['sanctioned']
sanctioned_cols = [col for col in df.columns if any(kw in col for kw in keywords) and 'B_I_all' not in col]
df = df.drop(columns=sanctioned_cols)

# Sanctioned contracts
df = df[df['sanctionedB_I_all'] == 1].reset_index(drop=True).drop(columns=['sanctionedB_I_all'])

#Suppliers with contracts in more than one year
multiyear_suppliers = df.groupby(['supplier_name_clean', 'contract_year']).size().reset_index(name='ncontracts').groupby('supplier_name_clean').size().reset_index(name='nyears')
multiyear_suppliers = multiyear_suppliers[multiyear_suppliers['nyears'] > 1].reset_index(drop=True)
multiyear_suppliers = multiyear_suppliers['supplier_name_clean'].unique()

df = df[df['supplier_name_clean'].isin(multiyear_suppliers)].reset_index(drop=True)

#order the dataframe by year
df = df.sort_values(by=['contract_year'], ascending=True).reset_index(drop=True)

results = []

for idxs, supplier in tqdm(enumerate(multiyear_suppliers), desc='Processing suppliers'):
    df4supplier = df.copy()
    df4supplier = df4supplier[df4supplier['supplier_name_clean'] == supplier].reset_index(drop=True)

    #indices of the years stored in a dictionary // This indexes are for filtering the simmilarity matrix
    years_idx_dict = {}
    for i, year in enumerate(df4supplier['contract_year'].unique()):
        years_index = df4supplier[df4supplier['contract_year'] == year].index
        years_idx_dict[year] = years_index
    # Get pairwise combinations of years a supplier has contracts in // This is used to iterate through combinations of years. We have to use the indexes
    year_combinations = list(combinations(df4supplier['contract_year'].unique(), 2))
    # Calculate the simmilarity matrices
    df4supplier.drop(columns=identifiers, inplace=True)
    df4supplier = df4supplier.to_numpy()
    #cosine distance
    cosine_s = pdist(df4supplier, 'cosine')
    cosine_s_matrix = squareform(cosine_s)
    #correlation distance
    correlation_s = pdist(df4supplier, 'correlation')
    correlation_s_matrix = squareform(correlation_s)
    #hamming distance
    hamming_s = pdist(df4supplier, 'hamming')
    hamming_s_matrix = squareform(hamming_s)

    results_dict = {}
    print('Year combinations for supplier', supplier, ':', len(year_combinations))

    for year_pairs in year_combinations:
          
        # Get the indices for the two years
        year1_idx = years_idx_dict[year_pairs[0]]
        year2_idx = years_idx_dict[year_pairs[1]]
        # Extract the submatrix for the two years
        cosine_submatrix = cosine_s_matrix[np.ix_(year1_idx, year2_idx)]
        correlation_submatrix = correlation_s_matrix[np.ix_(year1_idx, year2_idx)]
        hamming_submatrix = hamming_s_matrix[np.ix_(year1_idx, year2_idx)]

        # Cosine distance
        cosine_mean = np.mean(cosine_submatrix) # Mean of mean distance of one contract to all other contracts = mean distance of all contracts in year1 to all other contracts in year2
        cosine_std = np.std(cosine_submatrix) # Standard deviation of the distances
        cosine_min = np.min(cosine_submatrix) # Minimum distance
        cosine_q25 = np.quantile(cosine_submatrix, 0.25) # 25th percentile
        cosine_q50 = np.median(cosine_submatrix)
        cosine_q75 = np.quantile(cosine_submatrix, 0.75)
        cosine_max = np.max(cosine_submatrix)

        results_dict = {
            #'year_pairs': year_pairs,
            'year1': int(year_pairs[0]),
            'year2': int(year_pairs[1]),
            'supplier_name_clean': supplier,
            'cosine_mean': cosine_mean,
            'cosine_std': cosine_std,
            'cosine_min': cosine_min,
            'cosine_q25': cosine_q25,
            'cosine_q50': cosine_q50,
            'cosine_q75': cosine_q75,
            'cosine_max': cosine_max,
     
        }
        
        results.append(results_dict)

    if idxs in [0, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, len(multiyear_suppliers)-1]:
        print(' Saving intermediate results in iteration', idxs)
        file_path = supplementary_data / 'similarity4yearlabels.json'
        with open(file_path, 'w') as f:
            json.dump(results, f)

