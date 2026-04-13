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
import re

from additional_utils.functions import generate_param_combinations

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


CSFull_folder = here('data/processed_data/transductive_data/CS_FullContracts')
CSU_folder = here('data/processed_data/transductive_data/CS_Uniform')
models_folder = here('data/processed_data/transductive_data/models')
results_folder = here('data/processed_data/transductive_data/results')


#############################################################################################
################## Load the datasets ##########################################################

# fullcontracts files and dfs
CSFull_files = os.listdir(CSFull_folder)
CSFull_dfs = [pd.read_feather(os.path.join(CSFull_folder, f)) for f in CSFull_files]

# uniform files and dfs
CSU_files = os.listdir(CSU_folder)
CSU_dfs = [pd.read_feather(os.path.join(CSU_folder, f)) for f in CSU_files]

assert len(CSFull_files) == 5, 'more than expected'
##################################################################################################
######################## Specifications ##########################################################
##################################################################################################

algorithm = 'hdsrf'
param_num = 9993 #param num to save
param_num_act = 0
saving_interval = 1
r = 42


target_v = 'sanctionedB_I_all'
identifiers = [
    'file_code',
    'contract_code',
    'purchasing_unit_id',
    'supplier_name_clean',
    'data_id',
    'subset',
    'contract_year',
    ]


###################################################################################################
################################# Training and test sets ##########################################
####################################################################################################

##########################################List to store datsets
#subset id
subset_list = [0, 1, 2, 3]
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

param_grid = { # REAL GRID
            'n_estimators':[1000],
            'max_depth':[8],
            'min_samples_split':[2],
            'max_features':['sqrt'],
            'random_state':[r], #for all
            'alpha':[0.05],
        }

  
param_combinations_all = generate_param_combinations(param_grid)  # Generate param combinations
params = param_combinations_all[param_num_act]  # Select the specific parameter combination based on the input argument

###################################################################################################
################################# Model execution ##########################################
####################################################################################################

results = []

print('Running parameters: ', params)

for idx_subset, \
    X_train, y_train, \
    X_test_CSU, y_test_CSU, dataid_test_CSU, supplier_name_test_CSU, \
    X_test_CSFull, y_test_CSFull, dataid_test_CSFull, supplier_name_test_CSFull, \
    X_calibration_CSU, y_calibration_CSU, \
    X_calibration_CSFull, y_calibration_CSFull \
    in tqdm(zip(subset_list,\
                X_train_list, y_train_list, \
                X_test_CSU_list, y_test_CSU_list, dataid_test_CSU_list, supplier_name_test_CSU_list, \
                X_test_CSFull_list, y_test_CSFull_list, dataid_test_CSFull_list, supplier_name_test_CSFull_list, \
                X_calibration_CSU_list, y_calibration_CSU_list, \
                X_calibration_CSFull_list, y_calibration_CSFull_list), \
                desc='Train subsets', total=len(X_train_list)):


    ######################### Model Execution ##########################
    
    if algorithm == 'hdsrf':
                            
        params_m = {k: v for k, v in params.items() if k != 'alpha'} #this is only for HDSRF
        #params_m['max_samples'] = params_m['max_samples'] * y_train.sum()  # Adjust max_samples based on the training set size
        
        model = PURF_SIMP(
                pu_biased_bootstrap=True,
                **params_m,
                max_samples= sum(y_train == 0), # Adjust max_samples based on the training set size
                )
        
        model.fit(
                X_train, 
                y_train, 
                p_y = params['alpha'])
        
        ######################## CSU
        #Calibrate a model using isotonic regression CSU
        probs2calibrate_CSU = model.predict_proba(X_calibration_CSU)[:,1]
        calibrator_CSU = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        calibrator_CSU.fit(probs2calibrate_CSU, y_calibration_CSU)

        #Get uncalibrated and calibrated probabilities for CSU
        uncalibrated_probabilities_CSU = model.predict_proba(X_test_CSU)[:,1]
        calibrated_probabilities_CSU = calibrator_CSU.transform(uncalibrated_probabilities_CSU)
        
        ######################## CSFull
        probs2calibrate_CSFull = model.predict_proba(X_calibration_CSFull)[:,1]
        calibrator_CSFull = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        calibrator_CSFull.fit(probs2calibrate_CSFull, y_calibration_CSFull)

        # Get uncalibrated and calibrated probabilities for CSFull
        uncalibrated_probabilities_CSFull = model.predict_proba(X_test_CSFull)[:,1]
        calibrated_probabilities_CSFull = calibrator_CSFull.transform(uncalibrated_probabilities_CSFull)


    if idx_subset == 0:
        model_name = f'MANUAL{algorithm}_param{param_num}_subset{idx_subset}.pickle'
        model_path = models_folder / model_name
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    dict_model = {
        #identifiers
        'param_id': param_num,
        'subset': idx_subset,
        'model' : algorithm,
        'parameters': params,
        'class_prior' : params['alpha'],
        #CSU
        'y_test_CSU': y_test_CSU.tolist(),
        'dataid_test_CSU': dataid_test_CSU.tolist(),
        'supplier_name_test_CSU': supplier_name_test_CSU.tolist(),
        'uncalibrated_probabilities_CSU': uncalibrated_probabilities_CSU.tolist(),
        'calibrated_probabilities_CSU': calibrated_probabilities_CSU.tolist(),
        #CSFull
        'y_test_CSFull': y_test_CSFull.tolist(),
        'dataid_test_CSFull': dataid_test_CSFull.tolist(),
        'supplier_name_test_CSFull': supplier_name_test_CSFull.tolist(),
        'uncalibrated_probabilities_CSFull': uncalibrated_probabilities_CSFull.tolist(),
        'calibrated_probabilities_CSFull': calibrated_probabilities_CSFull.tolist(),
    }

    results.append(dict_model)


print('--------> Saving RESULTS in parameter id: ', param_num)
file_name = f'MANUAL{algorithm}_param{param_num}_{dt.datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")}.json'

file_path = results_folder / file_name
with open(file_path, 'w') as f:
    json.dump(results, f)

print('Finished!')
                








