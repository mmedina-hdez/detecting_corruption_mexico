#This file is to create a dataset with the results of 'cri' model, which means, no machine learning model but only the CRI
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

#sklearn related
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

CSFull_folder = here('data/processed_data/transductive_data/CS_FullContracts')
CSU_folder = here('data/processed_data/transductive_data/CS_Uniform')
models_folder = here('data/processed_data/transductive_data/models')
processed_data = here('data/processed_data/')


#############################################################################################
################## Load the datasets ##########################################################

red_flags_columns = [
    'rf_submission_period',
    'rf_decision_period',
    'rf_procedure_type',
    'rf_bl_conformity',
    'rf_buyer_dependence',
    'rf_single_bidder']
y_test_columns = ['sanctionedB_I_all']
columns2keep = red_flags_columns + y_test_columns

# fullcontracts files and dfs
CSFull_files = os.listdir(CSFull_folder)
CSFull_dfs = [pd.read_feather(os.path.join(CSFull_folder, f), columns=columns2keep) for f in CSFull_files]

#remove codification of nulls so they dont mess with cri computation
for idx, subset in enumerate(CSFull_dfs):
    for col in subset.columns:
        subset[col] = np.where(subset[col] == -1, np.nan, subset[col])

#compute CRI

red_flags_columns = ['rf_submission_period', 'rf_decision_period', 'rf_procedure_type', 'rf_bl_conformity', 'rf_buyer_dependence', 'rf_single_bidder']
for idx, subset in enumerate(CSFull_dfs):
    subset['CRI'] = subset[red_flags_columns].mean(axis=1, skipna=True)

results = []
for idx, subset in enumerate(CSFull_dfs):
    y_test = subset['sanctionedB_I_all'].copy()
    y_pred_prob = subset['CRI'].copy()
   
    results_dict = {
        'model': 'cri',
        'subset' : int(idx),
        'y_test_CSFull': list(y_test),
        'calibrated_probabilities_CSFull': list(y_pred_prob),    #this is the CRI, not probabilities, it is just to keep coherence with other datasets    
    }

    results.append(results_dict)

df = pd.DataFrame(results)
df = df[df['subset'] != 4] #because this is the one used for calibration

#save feather

df.to_feather(processed_data / 'cv_CRI.feather')