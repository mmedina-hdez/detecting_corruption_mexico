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
from sklearn.utils import shuffle

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
results_folder = here('data/processed_data/supplementary_data/permutation_test')


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

trial_test = sys.argv[1] == 'True'
r = int(sys.argv[2])
algorithm = 'hdsrf' #str(sys.argv[2])
param_num = 9993 #int(sys.argv[3])
param_num_act = 0
saving_interval = 1

if trial_test == True:
    #DELETE AFTER TEST
    #take only a sample from the dfs

    ns = 20000
    new_CSU_dfs = []
    new_CSFull_dfs = []

    for i in range(5):
        kCSU = CSU_dfs[i].copy().sample(n=ns, random_state=i).reset_index(drop=True)
        new_CSU_dfs.append(kCSU)
        kCSFull = CSFull_dfs[i].copy().sample(n=ns, random_state=i).reset_index(drop=True)
        new_CSFull_dfs.append(kCSFull)

    CSU_dfs = new_CSU_dfs
    CSFull_dfs = new_CSFull_dfs

target_v = 'sanctionedB_I_all'
identifiers = [
    'file_code',
    'contract_code',
    'purchasing_unit_id',
    'supplier_name_clean',
    'data_id',
    'subset',
    ]


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
            #'max_samples':[1],
            'n_estimators':[10,15, 25],
            'max_depth':[8],
            'min_samples_split':[100, 150],
            'max_features':['sqrt'],
            'random_state':[r], #for all
            'alpha':[0.1],
            }
    else:    
        param_grid = { # REAL GRID
            #'max_samples':[1, 10, 100],
            'n_estimators':[1000],
            'max_depth':[8],
            'min_samples_split':[2],
            'max_features':['sqrt'],
            'random_state':[r], #for all
            'alpha':[0.05],
        }



if algorithm == 'pubagging':

    if trial_test == True:
        param_grid = { #FOR TEST PURPOSES
            'gamma': [1], #kernel approximation
            'ncomponents': [200], #kernel approximation
            'max_iter': [50, 100], #SGDClassifier
            'loss': ['hinge'], #SGDClassifier
            'shuffle': [True], #SGDClassifier
            'n_estimators': [50, 100], #BaggingPuClassifier
            'random_state': [r], #for all
            'alpha': [None]
        }

    else:
        param_grid = {
            'gamma': [0.01, 0.1, 'scale', 1, 5], #kernel approximation
            'ncomponents': [200, 500, 1000], #kernel approximation
            'max_iter': [1000, 2000], #SGDClassifier
            'loss': ['hinge'], #SGDClassifier
            'shuffle': [True], #SGDClassifier
            'n_estimators': [500, 1000], #BaggingPuClassifier
            'random_state': [r], #for all
            'alpha': [None]
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
    X_test_CSFull, y_test_CSFull, dataid_test_CSFull, supplier_name_test_CSFull, \
    X_calibration_CSFull, y_calibration_CSFull \
    in tqdm(zip(subset_list,\
                X_train_list, y_train_list, \
                X_test_CSFull_list, y_test_CSFull_list, dataid_test_CSFull_list, supplier_name_test_CSFull_list, \
                X_calibration_CSFull_list, y_calibration_CSFull_list), \
                desc='Train subsets', total=len(X_train_list)):


    ######################### Model Execution ##########################
    
    #permutation of the labels
    y_train = y_train.astype(int)
    y_train_perm = shuffle(y_train.copy(), random_state=r)

    assert len(y_train) == len(y_train_perm)
    assert np.array_equal(y_train,y_train_perm) == False, ' The two arrays are the same'
    assert sum(y_train == 0) == sum(y_train_perm == 0)
    
  
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
                y_train_perm, #it should be trained with the permuted labels
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
        'param_id': int(param_num),
        'random_state': int(r),
        'subset': idx_subset,
        'model' : algorithm,
        'y_test_CSFull': y_test_CSFull.tolist(),
        'uncalibrated_probabilities_CSFull': uncalibrated_probabilities_CSFull.tolist(),
        'calibrated_probabilities_CSFull': calibrated_probabilities_CSFull.tolist(),
    }

    results.append(dict_model)


print('--------> Saving RESULTS in parameter id: ', param_num)
if trial_test == True:
    file_name = f'perm_{algorithm}_param{param_num}_{int(r)}.json'
else:
    file_name = f'perm_{algorithm}_param{param_num}_{int(r)}.json'

file_path = results_folder / file_name
with open(file_path, 'w') as f:
    json.dump(results, f)

print('Finished!')
                
                








