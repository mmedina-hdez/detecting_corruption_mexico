#shap_calculation

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
import shap
#sklearn related
from sklearn.preprocessing import StandardScaler
import re

print('version without CRI')

#HDSRF
from pu_tree_simplified_linux._pu_randomforest import PURandomForestClassifier as PURF_SIMP
from pu_tree_simplified_linux._pu_classes import DecisionTreeClassifier


CSFull_folder = here('data/processed_data/transductive_data/CS_FullContracts')
CSU_folder = here('data/processed_data/transductive_data/CS_Uniform')
models_folder = here('data/processed_data/transductive_data/models')
shap_folder = here() / 'data' / 'processed_data' / 'transductive_data' / 'shap_values'


#############################################################################################
################## Load the datasets ##########################################################

# fullcontracts files and dfs
CSFull_files = os.listdir(CSFull_folder)
CSFull_dfs = [pd.read_feather(os.path.join(CSFull_folder, f)) for f in CSFull_files]

# uniform files and dfs
CSU_files = os.listdir(CSU_folder)
CSU_dfs = [pd.read_feather(os.path.join(CSU_folder, f)) for f in CSU_files]

assert len(CSFull_files) == 5

print(CSFull_files)

##################################################################################################
######################## Specifications ##########################################################
##################################################################################################
test_type = 1 #0 is CSU, 1 is CSFull
trial_test = False
algorithm = 'hdsrf'
param_num = 9993
saving_interval = 1
r = 42
subset_in_file = 0


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
    'contract_year'
    ]

embedding_cols = [col for col in CSU_dfs[0].columns if re.fullmatch(r'd\d+', col)]
identifiers = identifiers + embedding_cols

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

for subset in tqdm(range(1), desc='Subset'):

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
    X_test_CSU = scaler.transform(X_test_CSU)
    X_test_csFull = scaler.transform(X_test_csFull)

np.random.seed(r)
sampled_indices = np.random.choice(X_train.shape[0], size= int(X_train.shape[0]/4), replace=False)
background_data = X_train[sampled_indices, :]

model_file = f'MANUAL{algorithm}_param{param_num}_subset{subset_in_file}.pickle'
with open(models_folder / model_file, 'rb') as f:
    mytreemodel = pickle.load(f)

tree_dicts = []
for tree in mytreemodel.estimators_:
    tree_tmp = tree.tree_

    values = (tree_tmp.value[:, 0, :] / tree_tmp.value[:, 0, :].sum(axis=1, keepdims=True)) / 1000
    pos_values = values[:, -1].reshape(-1, 1)
    
    tree_dict = {
        "children_left": tree_tmp.children_left.copy(),
        "children_right": tree_tmp.children_right.copy(),
        "children_default": tree_tmp.children_right.copy(),
        "features": tree_tmp.feature.copy(),
        "thresholds": tree_tmp.threshold.copy(),
        "values": pos_values.copy(),
        #"values": tree_tmp.value[:, 0, :],  # This is now a 2D array
        "node_sample_weight": tree_tmp.weighted_n_node_samples.copy(),
    }
    tree_dicts.append(tree_dict)

model = {
    "trees": tree_dicts,
    "base_offset": 0,  # Random Forest doesn't have a base_offset
    "tree_output": "probability",  # Random Forest outputs probabilities directly
    "objective": 'randomobjective',#"binary_crossentropy",
    "input_dtype": np.float32,
    "internal_dtype": np.float64,
}



print('Test type 1: CS FullContracts')
explainer = shap.TreeExplainer(model, background_data, feature_perturbation="interventional", model_output="probability")
pred_explainer = explainer.model.predict(background_data, output="probability")
print(np.mean(pred_explainer))
pred_mytreemodel = mytreemodel.predict_proba(background_data)[:, 1]
print(np.mean(pred_mytreemodel))
assert np.abs(pred_explainer - pred_mytreemodel).max() < 1e-4, 'probabilities are not equal'

#calculate shap values
shap_values = explainer(X_test_csFull)
# # Save the shap_values object to a file            
filename = f'SHAP_{algorithm}_param{param_num}_testtype_CSFull_rs{r}.pickle'
with open(shap_folder / filename, 'wb') as file:
    pickle.dump(shap_values, file)

print("shap_values object saved successfully.")