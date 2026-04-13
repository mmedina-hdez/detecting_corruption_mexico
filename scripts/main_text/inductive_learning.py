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

from additional_utils.functions import generate_param_combinations, annual_train_test_split

#sklearn related
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

#HDSRF
from pu_tree_simplified_linux._pu_randomforest import PURandomForestClassifier as PURF_SIMP
from pu_tree_simplified_linux._pu_classes import DecisionTreeClassifier

#PUBAGGING
from pos_noisyneg.PU_bagging import BaggingPuClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler


processed_data = here('data/processed_data')
inductive_results = here('data/processed_data/inductive_data/results')
inductive_models = here('data/processed_data/inductive_data/models')

##################################################################################################
######################## Specifications ##########################################################
##################################################################################################

#command: 
# python YS_exec.py 0 hdsrf #admin 0 for EPN, 1 for AMLO
# python YS_exec.py 1 pubagging

admin = int(sys.argv[1])
algorithm = str(sys.argv[2])
param_num = 1113 
param_num_act = 0 #Only if you are not doing hyperparameter tuning, otherwise it is the index of the parameter combination you want to run
saving_interval = 1
trial_test = False #CHANGE!!!!!!!!!!!!!!!!!!!
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

df = df.drop(columns=sanctioned_cols)

#DETELE AFTER TEST

if trial_test == True:
    df_ones = df[df[target_v] == 1].reset_index(drop=True)
    df_ones = df_ones.sample(n=500, random_state=42).reset_index(drop=True)
    df_zeros = df[df[target_v].isnull()].reset_index(drop=True)
    df_zeros = df_zeros.sample(n=500, random_state=42).reset_index(drop=True)
    df = pd.concat([df_ones, df_zeros], axis=0).reset_index(drop=True)
    print(df.shape)


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

########################
#Test sets
########################

#y tests and dataids YSUniform
y_test_YSUniform = ts_YSUniform[target_v].values
dataid_test_YSUniform = ts_YSUniform['data_id'].values
X_test_YSUniform = ts_YSUniform.drop(columns=[target_v] + identifiers).values

#y tests and dataids YSFull
y_test_YSFull = ts_YSFull[target_v].values
dataid_test_YSFull = ts_YSFull['data_id'].values
X_test_YSFull = ts_YSFull.drop(columns=[target_v] + identifiers).values

########################
#Calibration set
########################

#y calibration and dataids YSUniform
y_calibration_YSUniform = cal_YSUniform[target_v].values
dataid_calibration_YSUniform = cal_YSUniform['data_id'].values
X_calibration_YSUniform = cal_YSUniform.drop(columns=[target_v] + identifiers).values

y_calibration_YSFull = cal_YSFull[target_v].values
dataid_calibration_YSFull = cal_YSFull['data_id'].values
X_calibration_YSFull = cal_YSFull.drop(columns=[target_v] + identifiers).values

########################
#Training set
########################
y_train = tr_YSUniform[target_v].values
X_train = tr_YSUniform.reset_index(drop=True)
X_train = X_train.drop(columns=[target_v] + identifiers).values

########################
# standardize
########################

scaler = StandardScaler() 
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_calibration_YSUniform = scaler.transform(X_calibration_YSUniform)
X_calibration_YSFull = scaler.transform(X_calibration_YSFull)
X_test_YSUniform = scaler.transform(X_test_YSUniform)
X_test_YSFull = scaler.transform(X_test_YSFull)

######################## parameter combinations ########################

if algorithm == 'hdsrf':

    if trial_test == True:

        param_grid = { #FOR TEST PURPOSES
            'max_samples':[int(sum(y_train == 0))],
            'n_estimators':[10],
            'max_depth':[8],
            'min_samples_split':[100],
            'max_features':['sqrt'],
            'random_state':[r], #for all
            'alpha':[0.1],
            }
    else:    
        param_grid = { # REAL GRID
            'max_samples':[int(sum(y_train == 0))],
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
            'ncomponents': [50], #kernel approximation
            'max_iter': [50], #SGDClassifier
            'loss': ['hinge'], #SGDClassifier
            'shuffle': [True], #SGDClassifier
            'n_estimators': [10], #BaggingPuClassifier
            'random_state': [r], #for all
            'alpha': [None]
        }

    else:
        param_grid = {
            'gamma': [0.01], #kernel approximation
            'ncomponents': [200], #kernel approximation
            'max_iter': [1000], #SGDClassifier
            'loss': ['hinge'], #SGDClassifier
            'shuffle': [True], #SGDClassifier
            'n_estimators': [500], #BaggingPuClassifier
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

######################### Model Execution ##########################
    
if algorithm == 'hdsrf':
                                
        params_m = {k: v for k, v in params.items() if k != 'alpha'} #this is only for HDSRF
        
        model = PURF_SIMP(
                pu_biased_bootstrap=True,
                **params_m, #did you included max_samples in the params?
                )
        
        model.fit(
                X_train, 
                y_train, 
                p_y = params['alpha'])
   
if algorithm == 'pubagging':
                    
        kernel_approx_keys = ['gamma', 'n_components', 'random_state']
        params_kernel_approx = {k: params[k] for k in kernel_approx_keys if k in params}
        
        sgdc_keys = ['max_iter', 'loss', 'shuffle', 'random_state']
        params_sgdc = {k: params[k] for k in sgdc_keys if k in params}

        baggingpu_keys = ['n_estimators', 'random_state']
        params_baggingpu = {k: params[k] for k in baggingpu_keys if k in params}
        
        rbf_feature = RBFSampler(**params_kernel_approx)
        X_train = rbf_feature.fit_transform(X_train)
        X_test_YSUniform = rbf_feature.transform(X_test_YSUniform)
        X_test_YSFull = rbf_feature.transform(X_test_YSFull)
        X_calibration_YSUniform = rbf_feature.transform(X_calibration_YSUniform)
        X_calibration_YSFull = rbf_feature.transform(X_calibration_YSFull)
        

        model = BaggingPuClassifier(
                            SGDClassifier(**params_sgdc),
                            **params_baggingpu,
                            max_samples = y_train.sum() 
                            )
        
        model.fit(
                X_train, 
                y_train)
        
######################## Calibration YSUniform
#Calibrate a model using isotonic regression YSU
probs2calibrate_YSUniform = model.predict_proba(X_calibration_YSUniform)[:,1]
calibrator_YSUniform = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
calibrator_YSUniform.fit(probs2calibrate_YSUniform, y_calibration_YSUniform)

#Probabilities for YSUniform
uncalibrated_probabilities_YSUniform = model.predict_proba(X_test_YSUniform)[:,1]
calibrated_probabilities_YSUniform = calibrator_YSUniform.transform(uncalibrated_probabilities_YSUniform)

######################## Calibration YSFull
#Calibrate a model using isotonic regression YSFull
probs2calibrate_YSFull = model.predict_proba(X_calibration_YSFull)[:,1]
calibrator_YSFull = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
calibrator_YSFull.fit(probs2calibrate_YSFull, y_calibration_YSFull)

#Probabilities for YSFull
uncalibrated_probabilities_YSFull = model.predict_proba(X_test_YSFull)[:,1]
calibrated_probabilities_YSFull = calibrator_YSFull.transform(uncalibrated_probabilities_YSFull)         

# Save model

model_name = f'YS_{administration}_{algorithm}_param{param_num}.pickle'
model_path = inductive_models / model_name
with open(model_path, 'wb') as f:
        pickle.dump(model, f)

dict_model = {
    #identifiers
    'param_id': param_num,
    'model' : algorithm,
    'parameters': params,
    'class_prior' : str(params['alpha']),
    'administration': administration,
    #YSUniform
    'y_test_YSUniform': y_test_YSUniform.tolist(),
    'dataid_test_YSUniform': dataid_test_YSUniform.tolist(),
    'uncalibrated_probabilities_YSUniform': uncalibrated_probabilities_YSUniform.tolist(),
    'calibrated_probabilities_YSUniform': calibrated_probabilities_YSUniform.tolist(),
    #YSFull
    'y_test_YSFull': y_test_YSFull.tolist(),
    'dataid_test_YSFull': dataid_test_YSFull.tolist(),
    'uncalibrated_probabilities_YSFull': uncalibrated_probabilities_YSFull.tolist(),
    'calibrated_probabilities_YSFull': calibrated_probabilities_YSFull.tolist(),                
}

results.append(dict_model)



print('--------> Saving RESULTS in parameter id: ', param_num)
file_name = f'YS_{administration}_{algorithm}_param{param_num}_{dt.datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")}.json'

file_path = inductive_results / file_name
with open(file_path, 'w') as f:
    json.dump(results, f)

print('Finished!')
                








