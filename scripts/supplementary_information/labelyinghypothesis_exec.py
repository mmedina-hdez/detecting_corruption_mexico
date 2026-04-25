##############################################
# Preliminars
##############################################
import sys
import os
from pyprojroot.here import here
sys.path.append(str(here() / 'methods')) 
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import expit
import datetime as dt
import json
import ast

from additional_utils.functions import generate_param_combinations
from additional_utils.functions import qtop_bottom_split, uniform_sampling, balanced_split

#sklearn related
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

#HDSRF
from pu_tree_simplified_linux._pu_randomforest import PURandomForestClassifier as PURF_SIMP
from pu_tree_simplified_linux._pu_classes import DecisionTreeClassifier

#PUBAGGING
from pos_noisyneg.PU_bagging import BaggingPuClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
import re


processed_data = here('data/processed_data')
results_folder = here('data/processed_data/supplementary_data')


target_v = str(sys.argv[1]) #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ CHANGE
#target_v = 'sanctionedA_C_all'
estimators = 1000 #@@@@@@@@@@@@@@@@@@@@@@@@@@@@ CHANGE

#############################################################################################
################## Load the datasets ##########################################################

# Load the contracts data
contracts = pd.read_feather(processed_data / 'contracts2ml.feather')
contracts['data_id'] = contracts.index

##########################
# Labels
##########################

#Fix labels
cols_A = [
    'sanctionedA_C_all',
    'sanctionedA_C_max1',
    'sanctionedA_C_max2',
    'sanctionedA_C_max3',
    'sanctionedA_I_all',
    'sanctionedA_I_max1',
    'sanctionedA_I_max2',
    'sanctionedA_I_max3',
]
contracts[cols_A] = contracts[cols_A].fillna(0)
#hypothesis B
cols_B = [
    'sanctionedB_C_all',
    'sanctionedB_C_max1',
    'sanctionedB_C_max2',
    'sanctionedB_C_max3',
    'sanctionedB_I_all',
    'sanctionedB_I_max1',
    'sanctionedB_I_max2',
    'sanctionedB_I_max3'
]
contracts[cols_B] = contracts[cols_B].fillna(0)



labels2test = cols_A + cols_B
# For D_I labels
# Remove the sanctioned hypothesis columns
keywords = ['sanctioned']
labels2remove = [col for col in contracts.columns if any(kw in col for kw in keywords)]
labels2remove = list(set(labels2remove).difference(set({target_v})))

#remove all other labels
contracts = contracts.drop(columns=labels2remove)


##########################
# Make subsets
##########################

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
CSFull_dfs = []
for i in range(5):
    contracts_subset = contracts[contracts['supplier_name_clean'].isin(suppliers[i])].copy()
    contracts_subset['subset'] = i
    CSFull_dfs.append(contracts_subset)

# Get the uniform sampled version of the subsets
qcutoff = 0.95
random_state = 42
group_col = 'supplier_name_clean'

CSU_dfs = []

for i in CSFull_dfs:
    print(i.shape)
    #i[target_column] = i[target_column].fillna(0)
    #Get the bottom and top contracts
    qnum, topq_maxcontracts, topq_contracts, bottomq_contracts = qtop_bottom_split(df=i, qcutoff=qcutoff, target_column=target_v)
    topq_UniformContracts = uniform_sampling(
        df = topq_contracts, 
        k = qnum, 
        random_state = random_state, 
        label_col = target_v, 
        group_col = group_col)
    #make the final train and test
    train_test = pd.concat([topq_UniformContracts, bottomq_contracts]).reset_index(drop = True)

    CSU_dfs.append(train_test)


#see the prevalence of the target variable in each non-uniform subset
for i in CSFull_dfs:
    print('#########################')
    print('Prevalence in target variable in NON uniform')
    print(i.shape)
    print(i[target_v].mean())
    print(i[target_v].sum())


#see the prevalence of the target variable in each uniform subset
for i in CSU_dfs:
    print('#########################')
    print('Prevalence in target variable in Uniform')
    print(i.shape)
    print(i[target_v].mean())
    print(i[target_v].sum())
    
#############################################################################################
################## Load the datasets ##########################################################

assert len(CSU_dfs) == 5

##################################################################################################
######################## Specifications ##########################################################
##################################################################################################

trial_test = False
algorithm = 'hdsrf'
param_num = '9996'
r = 42


identifiers = [
    'file_code',
    'contract_code',
    'purchasing_unit_id',
    'supplier_name_clean',
    'data_id',
    'subset',
    ]


identifiers = identifiers

###################################################################################################
################################# Training and test sets ##########################################
####################################################################################################

##########################################List to store datsets
#subset id
subset_list = [0, 1, 2, 3] #this are for training and testing, 4 is for calibration
#CS Uniform - TEST
y_test_CSU_list = []
dataid_test_CSU_list = []
supplier_name_test_CSU_list = []
X_test_CSU_list = []
#CS FullContracts - TEST
y_test_CSFull_list = []
dataid_test_CSFull_list = []
supplier_name_test_CSFull_list = []
X_test_CSFull_list = []
#Training set
y_train_list = []
X_train_list = []
#Calibration sets
y_calibration_CSU_list = []
y_calibration_CSFull_list = []
X_calibration_CSU_list = []
X_calibration_CSFull_list = []

for subset in tqdm(range(4), desc='Subset'):

    ########################################## Create the train and tests sets
    #CS Uniform - TEST
    y_test_CSU_list.append(CSU_dfs[subset][target_v].values)
    dataid_test_CSU_list.append(CSU_dfs[subset]['data_id'].values)
    supplier_name_test_CSU = CSU_dfs[subset]['supplier_name_clean'].values
    supplier_name_test_CSU_list.append(supplier_name_test_CSU)
    X_test_CSU = CSU_dfs[subset].drop(columns=[target_v] + identifiers).values
    
    #CS FullContracts - TEST
    y_test_CSFull_list.append(CSFull_dfs[subset][target_v].values)
    dataid_test_CSFull_list.append(CSFull_dfs[subset]['data_id'].values)
    supplier_name_test_csFull = CSFull_dfs[subset]['supplier_name_clean'].values
    supplier_name_test_CSFull_list.append(supplier_name_test_csFull)
    X_test_csFull = CSFull_dfs[subset].drop(columns=[target_v] + identifiers).values

    #Calibration sets
    calibration_dataset = 4
    #CS Uniform - Calibration
    y_calibration_CSU_list.append(CSU_dfs[calibration_dataset][target_v].values)
    X_calibration_CSU = CSU_dfs[calibration_dataset].drop(columns=[target_v] + identifiers).values
    #CS FullContracts - Calibration
    y_calibration_CSFull_list.append(CSFull_dfs[calibration_dataset][target_v].values)
    X_calibration_CSFull = CSFull_dfs[calibration_dataset].drop(columns=[target_v] + identifiers).values
    
    
    #Training set
    train_set_indexes = [subset_t for subset_t in [0, 1, 2, 3] if subset_t != subset]
    train_set = pd.DataFrame()
    
    for train_set_index in train_set_indexes:
        train_set = pd.concat([train_set, CSU_dfs[train_set_index]], ignore_index=True)
    y_train_list.append(train_set[target_v].values)
    supplier_name_train = train_set['supplier_name_clean'].values
    X_train = train_set.drop(columns=[target_v] + identifiers).values
    
    scaler = StandardScaler() 
    scaler.fit(X_train)


    X_train = scaler.transform(X_train)
    X_train_list.append(X_train)

    X_test_CSU = scaler.transform(X_test_CSU)
    X_test_CSU_list.append(X_test_CSU)

    X_test_csFull = scaler.transform(X_test_csFull)
    X_test_CSFull_list.append(X_test_csFull)

    X_calibration_CSU = scaler.transform(X_calibration_CSU)
    X_calibration_CSU_list.append(X_calibration_CSU)

    X_calibration_CSFull = scaler.transform(X_calibration_CSFull)
    X_calibration_CSFull_list.append(X_calibration_CSFull)


    #ASSERT THAT
    assert set(supplier_name_train).intersection(set(supplier_name_test_CSU)) == set()
    assert set(supplier_name_train).intersection(set(supplier_name_test_csFull)) == set()

######################## parameter combinations ########################

if algorithm == 'hdsrf':

    if trial_test == True:

        param_grid = { #FOR TEST PURPOSES
            'max_samples':[1],
            'n_estimators':[10,15, 25],
            'max_depth':[8],
            'min_samples_split':[100, 150],
            'max_features':['sqrt'],
            'random_state':[r], #for all
            'alpha':[0.1],
            }
    else:    
        param_grid = { 
            #'max_samples':[1],
            'n_estimators':[estimators],
            'max_depth':[8],
            'min_samples_split':[2],
            'max_features':['sqrt'],
            'random_state':[r], #for all
            'alpha':[0.05],
            }

   
param_combinations_all = generate_param_combinations(param_grid)  # Generate param combinations
params = param_combinations_all[0]  # Select the specific parameter combination based on the input argument

###################################################################################################
################################# Model execution ##########################################
####################################################################################################

results = []

print('Running parameters: ', params)


for idx_subset, \
    X_train, y_train, \
    X_test_CSFull, y_test_CSFull, dataid_test_CSFull, supplier_name_test_CSFull, \
    X_calibration_CSFull, y_calibration_CSFull \
    in tqdm(zip(subset_list,\
                X_train_list, y_train_list, \
                X_test_CSFull_list, y_test_CSFull_list, dataid_test_CSFull_list, supplier_name_test_CSFull_list, \
                X_calibration_CSFull_list, y_calibration_CSFull_list), \
                desc='Train subsets', total=len(X_train_list)):


    ######################### Model Execution ##########################
    
 
    if algorithm == 'hdsrf':
                            
        params_m = {k: v for k, v in params.items() if k != 'alpha'} #this is only for HDSRF
        
        
        model = PURF_SIMP(
                pu_biased_bootstrap=True,
                **params_m,
                max_samples= sum(y_train == 0), # Adjust max_samples based on the training set size
                )
        
        model.fit(
                X_train, 
                y_train, #it should be trained with the permuted labels
                p_y = params['alpha'])
        
        
        ######################## CSFull
        probs2calibrate_CSFull = model.predict_proba(X_calibration_CSFull)[:,1]
        calibrator_CSFull = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        calibrator_CSFull.fit(probs2calibrate_CSFull, y_calibration_CSFull)

        # Get uncalibrated and calibrated probabilities for CSFull
        uncalibrated_probabilities_CSFull = model.predict_proba(X_test_CSFull)[:,1]
        calibrated_probabilities_CSFull = calibrator_CSFull.transform(uncalibrated_probabilities_CSFull)

    
    dict_model = {
        #identifiers
        'param_id': str(param_num),
        'random_state': int(r),
        'subset': idx_subset,
        'model' : algorithm,
        'label2test': str(target_v),
        #CSFull
        'y_test_CSFull': y_test_CSFull.tolist(),
        'uncalibrated_probabilities_CSFull': uncalibrated_probabilities_CSFull.tolist(),
        'calibrated_probabilities_CSFull': calibrated_probabilities_CSFull.tolist(),
    }

    results.append(dict_model)



print('--------> Saving RESULTS in parameter id: ', param_num)
if trial_test == True:
    file_name = f'TESTTRIAL{algorithm}_param{param_num}.json'
else:
    file_name = f'LABELINGTEST_{algorithm}_param{param_num}_{str(target_v)}.json'

file_path = results_folder / file_name
with open(file_path, 'w') as f:
    json.dump(results, f)

print(f'#################################### I finished the label {str(target_v)}')
                








