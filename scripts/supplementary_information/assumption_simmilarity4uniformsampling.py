import sys
from pyprojroot import here
sys.path.append(str(here() / 'methods')) 
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist


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

# Top 5% suppliers
ncontracts = df.groupby('supplier_name_clean').size().reset_index(name='ncontracts').sort_values('ncontracts', ascending=False)
th = ncontracts['ncontracts'].quantile(0.95)
top5suppliers = ncontracts[ncontracts['ncontracts'] > ncontracts['ncontracts'].quantile(0.95)]['supplier_name_clean'].unique()

df = df[df['supplier_name_clean'].isin(top5suppliers)].reset_index(drop=True)

#empty lists to save the results
supplier_list = []
ncontracts_list = []

cosine_mean_list = []
cosine_stdev_list = []
cosine_min_list = []
cosine_q25_list = []
cosine_q50_list = []
cosine_q75_list = []
cosine_max_list = []


for idx, supplier in tqdm(enumerate(top5suppliers), desc= 'Suppliers'):
    df4supplier = df.copy()
    df4supplier = df4supplier[df4supplier['supplier_name_clean'] == supplier].reset_index(drop=True)
    df4supplier.drop(columns=identifiers + ['sanctionedB_I_all'], inplace=True)

    assert len(df4supplier) > th , f'Supplier {supplier} has less than {th} contracts: {len(df4supplier)}'

    df4supplier = df4supplier.to_numpy()
    cosine_s = pdist(df4supplier, 'cosine')
    corrdistance_s = pdist(df4supplier, 'correlation')
    hamming_s = pdist(df4supplier, 'hamming')
    #save
    supplier_list.append(supplier)
    ncontracts_list.append(len(df4supplier))
    #cosine
    cosine_mean_list.append(cosine_s.mean())
    cosine_stdev_list.append(cosine_s.std())
    cosine_min_list.append(cosine_s.min())
    cosine_q25_list.append(np.quantile(cosine_s, 0.25))
    cosine_q50_list.append(np.quantile(cosine_s, 0.5))
    cosine_q75_list.append(np.quantile(cosine_s, 0.75))
    cosine_max_list.append(cosine_s.max())


    if idx in [len(top5suppliers)-1]:
        print(' Saving intermediate results in iteration', idx)
        # Save the results
        results = pd.DataFrame({
            'supplier_name_clean': supplier_list,
            'ncontracts': ncontracts_list,
            #cosine
            'cosine_mean': cosine_mean_list,
            'cosine_stdev': cosine_stdev_list,
            'cosine_min': cosine_min_list,
            'cosine_q25': cosine_q25_list,
            'cosine_q50': cosine_q50_list,
            'cosine_q75': cosine_q75_list,
            'cosine_max': cosine_max_list,
 
            })

        results.to_feather(supplementary_data / 'similarity4uniformsampling.feather')


    
