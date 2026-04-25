#functions

# Importing libraries
import sys
#add path
sys.path.append(r'C:\Users\mmedi\Documents\marti_TP16\optunity')
import operator as op
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score
import pandas as pd
from itertools import product
from tqdm import tqdm

#function to make binary a probability
def binarize_prob(prob, threshold=0.50):
  labels = np.zeros(prob.shape, dtype=int)
  labels[prob>threshold] = 1
  return labels

# #############################################################

# #Functions to create a proper dataset for the model

def stratified_company_split(df, random_state, target_column, split_proportion=0.7):

    cdistribution = df.groupby('supplier_name_clean')[target_column].sum().reset_index()
    pos_companies = cdistribution[cdistribution[target_column] > 0]['supplier_name_clean'].unique()
    unlabeled_companies = cdistribution[cdistribution[target_column] == 0]['supplier_name_clean'].unique()
    #select random pos companies
    np.random.seed(random_state)
    pos_train = np.random.choice(pos_companies, int(len(pos_companies)*split_proportion), replace=False)
    pos_test = np.array(list(set(pos_companies).difference(set(pos_train))))
    assert set(pos_train).intersection(set(pos_test)) == set()
    #select random unlabeled companies
    unlabeled_train = np.random.choice(unlabeled_companies, int(len(unlabeled_companies)*split_proportion), replace=False)
    unlabeled_test = np.array(list(set(unlabeled_companies).difference(set(unlabeled_train))))
    assert len(unlabeled_companies) == len(unlabeled_train) + len(unlabeled_test)
    #create 
    Utrain = df[df['supplier_name_clean'].isin(np.concatenate([pos_train, unlabeled_train]))].reset_index(drop=True)
    Utest = df[df['supplier_name_clean'].isin(np.concatenate([pos_test, unlabeled_test]))].reset_index(drop=True)
    print('######################## stratified_company_split ########################')
    print('train size: ', len(Utrain))
    print('test size: ', len(Utest))
    print('p_labeled TR: ', Utrain[target_column].mean())
    print('p_labeled TS: ', Utest[target_column].mean())
    print('###########################################################################')
    
    return Utrain, Utest

def qtop_bottom_split(df, qcutoff, target_column):
    cdistribution = df.groupby('supplier_name_clean').agg(nsanctioned = (target_column, 'sum'),
                                                             ncontracts = (target_column, 'count')).reset_index().sort_values(by = 'ncontracts', ascending = False)
    qnum = int(cdistribution['ncontracts'].quantile(qcutoff))
    print(f'Quantile {qcutoff} is {qnum}')

    topq_maxncontracts = cdistribution[cdistribution['nsanctioned'] > 0].sort_values(by = 'ncontracts', ascending = False).head(1)['ncontracts'].values[0]
    print(f'Top supplier with sanctions has {topq_maxncontracts} contracts')
    topq_suppliers = cdistribution[cdistribution['ncontracts'] > qnum]['supplier_name_clean'].unique()
    print(f'There are {len(topq_suppliers)} suppliers with more than {qnum} contracts')
    topq_contracts = df[df['supplier_name_clean'].isin(topq_suppliers)].reset_index(drop = True)
    print(f'There are {len(topq_contracts)} contracts from the top suppliers')
    bottomq_contracts = df[~df['supplier_name_clean'].isin(topq_suppliers)].reset_index(drop = True)
    print(f'There are {len(bottomq_contracts)} contracts from the bottom suppliers')
    assert set(topq_suppliers).intersection(set(bottomq_contracts['supplier_name_clean'].unique())) == set()

    return qnum, topq_maxncontracts, topq_contracts, bottomq_contracts

def uniform_sampling(df, k=None, random_state=None, label_col = None, group_col = 'supplier_name_clean'):

    assert df[label_col].isnull().sum() == 0, "Target column contains null values. Please clean the target variable before proceeding."
    
    sampled_df = []
    
    # Group by supplier
    grouped = df.groupby(group_col)
    
    for supplier, group in grouped:
        # Count total and positive samples
        total_count = len(group)
        positive_count = group[label_col].sum()
        
        if positive_count == 0:  # If no positives, take random sample
            sample = group.sample(n=min(k, total_count), replace=False, random_state=random_state)
        else:
            # Compute proportion of positives
            p = positive_count / total_count
            num_positives = max(1, round(k * p))  # Ensure at least 1 positive if possible
            num_negatives = k - num_positives
            
            # Sample positives and negatives separately
            positives = group[group[label_col] == 1].sample(n=min(num_positives, positive_count), replace=False, random_state=random_state)
            negatives = group[group[label_col] == 0].sample(n=min(num_negatives, total_count - positive_count), replace=False, random_state=random_state)
            
            # Concatenate sampled data
            sample = pd.concat([positives, negatives])
        
        sampled_df.append(sample)
    
    # Combine results
    return pd.concat(sampled_df).reset_index(drop=True)



def uniform_train_only(
        df,
        random_state,
        target_column, 
        qcutoff=None, 
        group_col = None):
    
    assert df[target_column].isnull().sum() == 0, "Target column contains null values. Please clean the target variable before proceeding."
        
    #Get the bottom and top contracts
    qnum, topq_maxcontracts, topq_contracts, bottomq_contracts = qtop_bottom_split(
        df = df,
        qcutoff = qcutoff, 
        target_column = target_column)
    
    #get the top uniform
    topq_UniformContracts = uniform_sampling(
        df = topq_contracts, 
        k = qnum, 
        random_state = random_state, 
        label_col = target_column, 
        group_col = group_col)
    
    #make the final train and test
    train_test = pd.concat([topq_UniformContracts, bottomq_contracts]).reset_index(drop = True)

    return train_test




def generate_param_combinations(param_dict):
    """
    Generates a list of all possible parameter combinations for a Random Forest model.
    
    Parameters:
    param_dict (dict): A dictionary where keys are parameter names and values are lists of possible values.
    
    Returns:
    list: A list of dictionaries, each representing a unique combination of parameters.
    """
    keys, values = zip(*param_dict.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    return param_combinations



def balanced_split(df, n_subsets=5, random_state=None):
    """
    Splits a ranked DataFrame into balanced subsets, maintaining distribution of 'ncontracts'.

    Parameters:
    - df: pandas DataFrame with 'supplier_name_clean' and 'ncontracts'
    - n_subsets: int, number of subsets to split into
    - random_state: int or None, for reproducibility

    Returns:
    - List of DataFrames (each a subset)
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)

    # Sort by ncontracts descending
    df_sorted = df.sort_values('ncontracts', ascending=False).reset_index(drop=True)

    # Prepare assignment list
    assignments = np.empty(len(df_sorted), dtype=int)

    for start in range(0, len(df_sorted), n_subsets):
        end = min(start + n_subsets, len(df_sorted))
        group_size = end - start

        # Random permutation of available subset indices
        perm = rng.permutation(n_subsets)[:group_size]
        assignments[start:end] = perm

    # Add assignment to DataFrame
    df_sorted['subset'] = assignments

    # Create subsets
    subsets = [df_sorted[df_sorted['subset'] == i].drop(columns='subset').reset_index(drop=True) for i in range(n_subsets)]

    return subsets



def mini_ranking_evaluations(y_true, y_pred_prob, top_k, prevalence = None):
    #GENERAL
    # Ensure y_true and y_pred_prob are numpy arrays
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    assert len(y_true) == len(y_pred_prob), "y_true and y_pred_prob must have the same length"
    #combine the arrays
    a = np.array(list(zip(y_true, y_pred_prob)))
    #sort by y_pred_prob
    a = a[a[:,1].argsort()[::-1]]
    #take only the top-k predictions according to y_pred_prob
    a = a[:top_k]

    # Cumulative gain and lift curve
    #get the cummulative sum of y_true
    cumsum = np.cumsum(a[:,0])  # Normalization comes after

    #ranking
    x_charts = np.arange(1, top_k + 1, step = 1)
    #Get expected cumulative sum of null model
    if prevalence is None:
        prevalence = np.mean(y_true)  # If prevalence is not provided, calculate it from y_true
    null_cumsum = np.arange(prevalence, top_k + prevalence, step = prevalence)[:top_k]  # Expected cumulative sum for 

    # Create group limits
    y_pred_prob_sorted = a[:,1]  # The sorted probabilities
    unique_groups, group_limits = np.unique(y_pred_prob_sorted, return_index=True)
    group_limits = list(group_limits - 1)  # Adjust to get the last index of each group
    group_limits.remove(-1)
    group_limits = group_limits + [0] + [len(y_pred_prob_sorted) - 1]  # Add the last index of the array
    group_limits = list(set(group_limits))  # Remove duplicates
    group_limits = sorted(group_limits)

    cumsum_group_limits = cumsum[group_limits]
    # Compute lengths of each group (difference between consecutive limits)
    lengths = np.diff(group_limits)
    # Repeat each cumsum value according to its group length
    fixed_cumsum = np.repeat(cumsum_group_limits[:-1], lengths)
    fixed_cumsum = np.append(fixed_cumsum, cumsum_group_limits[-1])  # Append the last value to match the length of x_charts
        
    # Gain
    gain = fixed_cumsum - null_cumsum
    average_gain = np.mean(gain) / y_true.sum() #normalized average gain

    # Lift
    lift = (fixed_cumsum / x_charts) / prevalence
    average_lift = np.mean(lift)


    return average_gain, average_lift



def annual_train_test_split(df, training_years, test_years, target_v, r):
    """
    Splits the contracts data into training and testing sets based on the specified administrations.
    """

    #years training
    tr_YSFull = df[df['contract_year'].isin(training_years)].reset_index(drop=True)
    tr_YSFull[target_v] = tr_YSFull[target_v].fillna(0).astype(int)

    #create U train and U cal (calibration set)
    U_train, U_cal = stratified_company_split(
        df = tr_YSFull, 
        random_state = r, 
        target_column = target_v, 
        split_proportion= 0.7)

    #get uniform sampling for training 
    tr_YSUniform = uniform_train_only(
        df = U_train,
        random_state = r,
        target_column = target_v, 
        qcutoff = 0.95, 
        group_col = 'supplier_name_clean')
    
    #get uniform sampling for calibration    
    cal_YSUniform = uniform_train_only(
        df = U_cal,
        random_state = r,
        target_column = target_v, 
        qcutoff = 0.95, 
        group_col = 'supplier_name_clean')
    
    #get the test set
    ts_YSFull = df[df['contract_year'].isin(test_years)].reset_index(drop=True)
    ts_YSFull[target_v] = ts_YSFull[target_v].fillna(0).astype(int)
    ts_YSUniform = uniform_train_only(
        df = ts_YSFull,
        random_state = r,
        target_column = target_v,
        qcutoff = 0.95,
        group_col = 'supplier_name_clean')
    
    return tr_YSUniform, cal_YSUniform, U_cal, ts_YSUniform, ts_YSFull


# #funtion that locates features with variation == 0
# def find_features_with_no_variation(df):
#     features = []
#     for column in df.columns:
#         if len(df[column].unique()) == 1:
#             features.append(column)
#     return features


def ranking_evaluations(
        y_true, 
        y_pred_prob, 
        topk_thresholds,
        prevalence = None):
    #GENERAL
    # Ensure y_true and y_pred_prob are numpy arrays
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    len_pred = len(y_true)
    topk_th_int = np.round(np.array(topk_thresholds)*len_pred,0).astype(int) -1
    assert len(y_true) == len(y_pred_prob), "y_true and y_pred_prob must have the same length"
    #combine the arrays
    a = np.array(list(zip(y_true, y_pred_prob)))
    #sort by y_pred_prob
    a = a[a[:,1].argsort()[::-1]]

    # Cumulative gain and lift curve
    #get the cummulative sum of y_true
    cumsum = np.cumsum(a[:,0])  # Normalization comes after

    #ranking
    x_charts = np.arange(1, len_pred + 1, step = 1)
    #Get expected cumulative sum of null model
    if prevalence is None:
        prevalence = np.mean(y_true)  # If prevalence is not provided, calculate it from y_true
    null_cumsum = np.arange(prevalence, (len_pred + 1) + prevalence, step = prevalence)[:len_pred]  # Expected cumulative sum for 

    # Same probability intervals
    y_pred_prob_sorted = a[:,1]  # The sorted probabilities
    unique_groups, sameprob_limits = np.unique(y_pred_prob_sorted, return_index=True)
    sameprob_limits = list(sameprob_limits - 1)  # Adjust to get the last index of each group
    sameprob_limits.remove(-1)
    sameprob_limits = sameprob_limits + [0] + [len(y_pred_prob_sorted) - 1]  # Add the last index of the array
    sameprob_limits = list(set(sameprob_limits))  # Remove duplicates
    sameprob_limits = sorted(sameprob_limits)

    #cumsum@k
    #cumsum_atk = cumsum[topk_th_int]

    #cumsum fixed by same probability limits
    cumsum_sameprob_limits = cumsum[sameprob_limits]
    # Compute lengths of each group (difference between consecutive limits)
    lengths = np.diff(sameprob_limits)
    
    # Repeat each cumsum value according to its group length
    robust_cumsum = np.repeat(cumsum_sameprob_limits[:-1], lengths)
    robust_cumsum = np.append(robust_cumsum, cumsum_sameprob_limits[-1])  # Append the last value to match the length of x_charts

      
    
    #Recall
    robust_recall = robust_cumsum / y_true.sum() #robust
    biased_recall = cumsum / y_true.sum() # biased
    null_recall = null_cumsum / y_true.sum() #null of recall curve

    # Gain
    robust_gain = robust_cumsum - null_cumsum #robust
    biased_gain = cumsum - null_cumsum #biased
    robust_average_gain = np.mean(robust_gain) / y_true.sum() #robust
    biased_average_gain = np.mean(biased_gain) / y_true.sum() #robust

    # Precision
    robust_precision = robust_cumsum / x_charts #robust
    biased_precision = cumsum / x_charts #biased
    robust_average_precision = np.mean(robust_precision) #robust
    biased_average_precision = np.mean(biased_precision)

    #Lift
    robust_lift = (robust_cumsum / x_charts) / prevalence #robust
    biased_lift = (cumsum / x_charts) / prevalence #robust
    robust_average_lift = np.mean(robust_lift) #robust
    biased_average_lift = np.mean(biased_lift)



    #normalize x coordinates
    x_charts = x_charts / len_pred

    all_coordinates = np.column_stack((
        x_charts, 
        #null model
        null_recall,
        #all robust curves
        robust_recall, 
        robust_precision,
        robust_lift,
        #all biased curves
        biased_recall,
        biased_precision,
        biased_lift,
        ))

    global_metrics = {
        'robust': {
            'average_gain': robust_average_gain,
            'average_precision': robust_average_precision,
            'average_lift': robust_average_lift,
        },
        'biased': {
            'average_gain': biased_average_gain,
            'average_precision': biased_average_precision,
            'average_lift':biased_average_lift,
        },
        'prevalence': prevalence
    }

    all_coordinates = all_coordinates[topk_th_int]
    all_coordinates = np.column_stack((topk_thresholds, all_coordinates))

    return all_coordinates, global_metrics


def coordinates2plot(
    df2work, #df2work
    y_test_col,
    pred_prob_col,
    ks = list(np.round(np.arange(0.01,1.01, 0.01),2)), #thresholds to measure
    group_col = None,
        ):
    
    df4coords = df2work.copy()
    coord_df = pd.DataFrame()

    for idx in tqdm(range(len(df4coords)), desc = 'rows'):
    
        y_true = df4coords.iloc[idx][y_test_col]
        y_pred_prob = df4coords.iloc[idx][pred_prob_col]
        group_ = df4coords.iloc[idx][group_col]
        #label_indicator = df4coords.iloc[idx]['label2check']

        all_coordinates, global_metrics = ranking_evaluations(
            y_true, 
            y_pred_prob, 
            ks,
            prevalence = None)
        
        current_coord = pd.DataFrame(
            all_coordinates, 
            columns = [
                    'topk_instances', 
                    'original_x', 
                    #null model
                    'null_recall',
                    #all robust curves
                    'robust_recall', 
                    'robust_precision',
                    'robust_lift',
                    #all biased curves
                    'biased_recall',
                    'biased_precision',
                    'biased_lift',                       
                       ])
        current_coord[group_col] = group_
        current_coord['prevalence'] = global_metrics['prevalence']
        for mtype in ['robust', 'biased']:
            for metric in ['gain', 'precision', 'lift']:            
                current_coord[f'{mtype}_average_{metric}'] = global_metrics[mtype][f'average_{metric}']

        coord_df = pd.concat([coord_df, current_coord], ignore_index = True)

    return coord_df


def get_top_features(shap_values, k):
    # Get absolute mean shap values per feature
    shap_values_values = np.abs(shap_values.values).mean(axis=0)
    
    # Pair feature names with their mean SHAP values
    feature_importances = list(zip(shap_values.feature_names, shap_values_values))
    
    # Sort by importance (descending)
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    
    # Take top-k feature names
    top_features = [f for f, _ in feature_importances[:k]]
    
    return top_features