# This script calculates the conformity of the data to Benford's Law
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from pyprojroot import here
import benford as bf

processed_data = here('data/processed_data')
var2keep = ['purchasing_unit_id', 'contract_price_mx', 'contract_year']

# Read data as feather
contracts = pd.read_feather(processed_data / 'mxc11to22_base.feather', columns=var2keep)
print('contracts shape:', contracts.shape)  
contracts['contract_year'] = contracts['contract_year'].astype(int)
print('contracts shape:', contracts.shape)

mad_l = []

# Calculate the MAD for each contract_year
for cyear in tqdm(contracts['contract_year'].unique(), desc='contract_year'):
    df2work = contracts[contracts['contract_year'] == cyear].copy().reset_index(drop=True)
    buyers_q = df2work.groupby('purchasing_unit_id').size().reset_index(name='counts')

    qcutoff = int(buyers_q['counts'].quantile(0.75))
    print('qcutoff:', qcutoff)

    # Filter buyers
    buyers_q = buyers_q[buyers_q['counts'] >= qcutoff].copy().reset_index(drop=True)    
    buyers_q = buyers_q['purchasing_unit_id'].unique()
    print(f'buyers with at least {qcutoff} contracts: ', len(buyers_q) / df2work['purchasing_unit_id'].nunique())

    df_size_year = df2work.shape[0]
    df2work = df2work[df2work['purchasing_unit_id'].isin(buyers_q)].copy().reset_index(drop=True)
    print('proportion of df: ', df2work.shape[0] / df_size_year)

    # Calculate the MAD for each buyer
    for buyer in buyers_q:
        prov_df = df2work[df2work['purchasing_unit_id'] == buyer].copy().reset_index(drop=True)
        mad = bf.mad(prov_df['contract_price_mx'].copy(), test=1, decimals=2)

        mad_prov = {
            'contract_year': str(cyear),
            'purchasing_unit_id': str(buyer),
            'mad': str(mad)
        }
        mad_l.append(mad_prov)

# Save results as JSON
with open(processed_data / 'mad_peryear.json', 'w') as f:
    json.dump(mad_l, f)

