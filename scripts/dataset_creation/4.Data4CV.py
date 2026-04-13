from pyprojroot.here import here
import sys
sys.path.append(str(here() / 'methods')) 
import pandas as pd
import numpy as np

#my functions
from additional_utils.functions import qtop_bottom_split, uniform_sampling, balanced_split

processed_data = here('data/processed_data')
cs_uniform = here('data/processed_data/transductive_data/CS_Uniform')
cs_fullcontracts = here('data/processed_data/transductive_data/CS_FullContracts')

# Load the contracts data
contracts = pd.read_feather(processed_data / 'contracts2ml.feather')
contracts['data_id'] = contracts.index

# For D_I labels
# Remove the sanctioned hypothesis columns
keywords = ['sanctioned']
sanctioned_cols = [col for col in contracts.columns if any(kw in col for kw in keywords) and 'B_I_all' not in col]
contracts = contracts.drop(columns=sanctioned_cols)


# Rank companies by the number of contracts
ncontracts_ranking = contracts.groupby(['supplier_name_clean']).size().reset_index(name = 'ncontracts').sort_values('ncontracts', ascending= False).reset_index(drop = True)
# Get subsets of companies balancing the the number of contracts
subsets = balanced_split(ncontracts_ranking, n_subsets=5, random_state=42)

#get the suppliers of each subset
suppliers = []
for i in subsets:
    i['supplier_name_clean'].unique()
    suppliers.append(i['supplier_name_clean'].unique())

#get the contracts for each subset
contracts_subsets = []
for i in range(5):
    contracts_subset = contracts[contracts['supplier_name_clean'].isin(suppliers[i])].copy()
    contracts_subset['subset'] = i
    contracts_subsets.append(contracts_subset)

#check the shape of each subset
for i in contracts_subsets:
    print(i.shape)

target_column = 'sanctionedB_I_all'

# Get the uniform sampled version of the subsets
qcutoff = 0.95
random_state = 42
group_col = 'supplier_name_clean'

UniformSubsets = []

for i in contracts_subsets:
    print(i.shape)
    i[target_column] = i[target_column].fillna(0)
    #Get the bottom and top contracts
    qnum, topq_maxcontracts, topq_contracts, bottomq_contracts = qtop_bottom_split(df=i, qcutoff=qcutoff, target_column=target_column)
    topq_UniformContracts = uniform_sampling(df = topq_contracts, k = qnum, random_state = random_state, label_col = target_column, group_col = group_col)
    #make the final train and test
    train_test = pd.concat([topq_UniformContracts, bottomq_contracts]).reset_index(drop = True)

    UniformSubsets.append(train_test)


#see the prevalence of the target variable in each non-uniform subset
for i in contracts_subsets:
    print(i.shape)
    print(i[target_column].mean())
    print(i[target_column].sum())
    if len(i['subset'].unique()) == 1:
        subset_number = str(i['subset'].unique()[0])
        i.reset_index(drop = True, inplace = True)
        i.to_feather(cs_fullcontracts / f'CS_FullContracts_{subset_number}.feather')
    else:
        break
    print('########################')

#see the prevalence of the target variable in each uniform subset
for i in UniformSubsets:
    print(i.shape)
    print(i[target_column].mean())
    print(i[target_column].sum())
    if len(i['subset'].unique()) == 1:
       subset_number = str(i['subset'].unique()[0])
       i.to_feather(cs_uniform / f'CS_Uniform_{subset_number}.feather')

    else:
       break
    print('########################')
    