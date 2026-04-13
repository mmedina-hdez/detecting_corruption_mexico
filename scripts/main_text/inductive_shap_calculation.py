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
from sklearn.model_selection import train_test_split
import re


#HDSRF
from pu_tree_simplified_linux._pu_randomforest import PURandomForestClassifier as PURF_SIMP
from pu_tree_simplified_linux._pu_classes import DecisionTreeClassifier

from additional_utils.functions import generate_param_combinations, annual_train_test_split

processed_data = here('data/processed_data')
annual_sampling_results = here('data/processed_data/inductive_data/results')
annual_sampling_models = here('data/processed_data/inductive_data/models')
annual_sampling_shap_values = here('data/processed_data/inductive_data/shap_values')

##################################################################################################
######################## Specifications ##########################################################
##################################################################################################

admin = int(sys.argv[1]) #0: EPN, 1: AMLO
test_type = int(sys.argv[2]) #0: YSUniform, 1: YSFull
algorithm = 'hdsrf'
param_num = 1113 #YearSampling
#param_num_act = 0 #Only if you are not doing hyperparameter tuning, otherwise it is the index of the parameter combination you want to run
#saving_interval = 1
#trial_test = True #CHANGE!!!!!!!!!!!!!!!!!!!
r = 42

if admin == 0:
    administration = 'EPN'
    target_v = 'sanctionedE_' + administration
    training_years = [2011, 2012, 2013, 2014, 2015, 2016]
    test_years = [2017]

elif admin == 1:
    administration = 'AMLO'
    target_v = 'sanctionedE_' + administration
    training_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    test_years = [2021]


identifiers = [
    'file_code',
    'contract_code',
    'purchasing_unit_id',
    'supplier_name_clean',
    'data_id',
    #'subset',
    'contract_year'
    ]

##################################################################################################
######################## Load datasets ###########################################################
##################################################################################################

print('Loading dataset...')
df = pd.read_feather(processed_data / 'contracts2ml.feather')
df['data_id'] = df.index

keywords = ['sanctioned']
sanctioned_cols = [col for col in df.columns if any(kw in col for kw in keywords) and target_v not in col]


df = df.drop(columns=sanctioned_cols )


###################################################################################################
################################# Training, test, and calibration sets ##########################################
####################################################################################################
print('Creating training, test, and calibration sets...')

#One training, two test, two calibration sets
tr_YSUniform, cal_YSUniform, cal_YSFull, ts_YSUniform, ts_YSFull = annual_train_test_split(
    df = df,
    training_years = training_years,
    test_years = test_years,
    target_v = target_v,
    r = r
)
#im not using calibration sets, just training and test

#y tests and dataids YSUniform
y_test_YSUniform = ts_YSUniform[target_v].values
dataid_test_YSUniform = ts_YSUniform['data_id'].values
X_test_YSUniform = ts_YSUniform.drop(columns=[target_v] + identifiers).values

#y tests and dataids YSFull
y_test_YSFull = ts_YSFull[target_v].values
dataid_test_YSFull = ts_YSFull['data_id'].values
X_test_YSFull = ts_YSFull.drop(columns=[target_v] + identifiers).values


training_set = tr_YSUniform.copy()
########################

# training set
y_train = training_set[target_v].values
training_set = training_set.reset_index(drop=True)
training_set = training_set.drop(columns=[target_v] + identifiers).values


########################
# standardize
########################

scaler = StandardScaler() 
scaler.fit(training_set)

X_train = scaler.transform(training_set)
#X_calibration = scaler.transform(calibration_set)
X_test_YSUniform = scaler.transform(X_test_YSUniform)
X_test_YSFull = scaler.transform(X_test_YSFull)

##################################################################################################
######################## Load model ###############################################################

np.random.seed(r)
sampled_indices = np.random.choice(X_train.shape[0], size= 150000, replace=False)
background_data = X_train[sampled_indices, :]

name = f'YS_{administration}_{algorithm}_param{param_num}.pickle'
with open(annual_sampling_models / name, 'rb') as f:
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




if test_type == 0:
    print('Test type 0: YS Uniform')
    explainer = shap.TreeExplainer(model, background_data, feature_perturbation="interventional", model_output="probability")
    pred_explainer = explainer.model.predict(background_data, output="probability")
    pred_mytreemodel = mytreemodel.predict_proba(background_data)[:, 1]
    assert np.abs(pred_explainer - pred_mytreemodel).max() < 1e-4, f'probabilities are not equal{pred_explainer[0:5]} and {pred_mytreemodel[0:5]}'

    #calculate shap values
    shap_values = explainer(X_test_YSUniform)
    # # Save the shap_values object to a file            
    filename = f'SHAP_{administration}_{algorithm}_param{param_num}_testtype_YSUniform_rs{r}.pickle'
    with open(annual_sampling_shap_values / filename, 'wb') as file:
        pickle.dump(shap_values, file)

    print("shap_values object saved successfully.")



if test_type == 1:
    print('Test type 1: CS FullContracts')
    explainer = shap.TreeExplainer(model, background_data, feature_perturbation="interventional", model_output="probability")
    pred_explainer = explainer.model.predict(background_data, output="probability")
    pred_mytreemodel = mytreemodel.predict_proba(background_data)[:, 1]
    assert np.abs(pred_explainer - pred_mytreemodel).max() < 1e-4, f'probabilities are not equal{pred_explainer[0:5]} and {pred_mytreemodel[0:5]}'

    #calculate shap values
    shap_values = explainer(X_test_YSFull)
    # # Save the shap_values object to a file            
    filename = f'SHAP_{administration}_{algorithm}_param{param_num}_testtype_YSFull_rs{r}.pickle'
    with open(annual_sampling_shap_values / filename, 'wb') as file:
        pickle.dump(shap_values, file)

    print("shap_values object saved successfully.")



