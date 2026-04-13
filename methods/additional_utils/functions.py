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


# ##############################################################
# # PR CURVE
# ##############################################################

# def pr_curve_th(y_test, y_pred_prob, len_thresholds = 100):

#     #set thresholds for positive class
#     thresholds = list(np.round(np.linspace(0, 0.99, num=100),3))
#     pr_recall = []
#     pr_precision = []

#     for i in thresholds:
#         y_pred = binarize_prob(y_pred_prob, threshold=i)
#         pr_recall.append(recall_score(y_test, y_pred))
#         pr_precision.append(precision_score(y_test, y_pred))

#     #prepare garantee that starts on 0,1
#     thresholds = thresholds + [1]
#     pr_recall = pr_recall + [0]
#     pr_precision = pr_precision + [1]

#     #order
#     sorted_thresholds, sorted_pr_recall, sorted_pr_precision = zip(*sorted(zip(thresholds, pr_recall, pr_precision), key=op.itemgetter(0), reverse=True))



#     return sorted_thresholds, sorted_pr_recall, sorted_pr_precision

# ##############################################################
# # Naive evaluation PU learning
# ##############################################################
# # These measures use the positive and unlabeled observations of the test set to evaluate
# # the performance of the PU learning method
# # Therefore, they answer the question: how well the PU learning method is able to identify the positive and unlabeled observations?
# # There are two functions: 
# # 1) to get the best threshold and 
# # 2) get the optimal threshold to binarize the probability

# # Naive evaluation of PU learning method
# # Input:
# # y_true: true positive-unlabeled observations of the test set
# # y_pred_prob: predicted positive-unlabeled observations of the test set
# # threshold: threshold to binarize the probability
# # Output:
# # naive_recall: recall of the PU learning method
# # naive_precision: precision of the PU learning method
# # naive_f1: f1 score of the PU learning method
# # naive_ap: average precision score of the PU learning method


# def naive_evaluation(y_true, y_pred_prob, threshold):
#     #binarize the probability
#     y_pred = binarize_prob(prob = y_pred_prob, threshold = threshold)
#     #recall
#     naive_recall = recall_score(y_true, y_pred)
#     #precision
#     naive_precision = precision_score(y_true, y_pred)
#     #f1 score
#     naive_f1 = f1_score(y_true, y_pred)
#     #average precision score
#     naive_ap = average_precision_score(y_true, y_pred_prob)

#     return naive_recall, naive_precision, naive_f1, naive_ap



# # Threshold optimization
# # Input:
# # y_validation_set: true positive-unlabeled observations of the validation set (y[ix_tr_val] or noisy_y[ix_tr_val], for reference)
# # y_pred_prob: predicted positive-unlabeled observations of the validation set 
# # Output:
# # opt_threshold: optimal threshold to binarize the probability

# def naive_threshold_optimization(y_validation_set, y_pred_prob):

#     #set thresholds for positive class
#     set_thresholds = np.linspace(0.01, 0.99, num=100)
#     #binarize the probability
#     set_pred_labels = [binarize_prob(y_pred_prob, threshold=i) for i in set_thresholds]
#     #f1 score for each threshold
#     set_f1_scores = [f1_score(y_validation_set, i) for i in set_pred_labels]
#     #get the best threshold
#     max_f1score = np.max(set_f1_scores)
#     #index of the best threshold
#     ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
#     #optimal threshold
#     opt_threshold = set_thresholds[ix_fscore]

#     return opt_threshold



# ##############################################################
# # Indirect Correction of PU performance measures
# # According to Jain, White, Radivojac (2017). Recovering True Classifier Performance in Positive-Unlabeled Learning.
# ##############################################################

# # The authors make a correction of some evaluation measures assuming SCAR
# # Three functions: 
# # 1) one to get the best threshold 
# # 2) another to evaluate the performance of the PU learning method with the optimal threshold
# # 3) calculate average precision score

# # Indirect Correction of PU learning method measures
# # Input:
# # y_true: true positive-unlabeled observations of the test set
# # y_pred_prob: predicted positive-negative observations of the test set. It can be a probability or a binary label
# # threshold: threshold to binarize the probability, it can be a number between 0 and 1 or False, in which case, the ic evaluation will be done with the raw probability
# # p_y1_hat: Estimation of positive labels in the dataset / estimation of positive class prior / estimation of P(y=1) 


# def IC_evaluation(y_true, y_pred_prob, threshold, p_y1_hat):
    
#     if threshold == False:
#         y_pred = y_pred_prob
#     else:
#         #binarize the probability
#         y_pred = binarize_prob(prob = y_pred_prob, threshold = threshold)

#     # true positive rate given PU dataset, i.e., comparing positive and unlabeled observations. This is equivalent to the real TPR, given SCAR
#     tpr_pu = np.mean(y_pred[y_true == 1]) 

#     #false positive rate given PU dataset, i.e., comparing positive and unlabeled observations
#     fpr_pu = np.mean(y_pred[y_true == 0]) 

#     # proportion of labeled observations in the dataset (P(l=1))
#     p_l1 = np.mean(y_true)
    
#     # fraction of estimated positive observations AMONG unlabeled observations P(y=1|l=0)
#     alpha = p_y1_hat - p_l1

#     #if alpha is negative, stop the evaluation
#     if alpha < 0:
#         print("Error: alpha is negative. The proportion of labeled is greater than the estimated proportion of positive observations.")
#         return None

#     if alpha >= 0:
#         #indirect correction of true positive rate / recall
#         ic_recall = ((1 - alpha) * tpr_pu) / (1 - alpha) #true positive rate given SCAR

#         #indirect correction of false positive rate
#         ic_fpr = (fpr_pu - (alpha * tpr_pu)) / (1 - alpha) #false positive rate given SCAR

#         #indirect correction of precision
#         ic_precision = (alpha * ic_recall) / fpr_pu

#         #indirect correction of f1 score
#         ic_f1 = (2 * ic_precision * ic_recall) / (ic_precision + ic_recall)

#         #TODO: average precision score
    
#     return ic_recall, ic_precision, ic_f1 #, ic_ap




# # Threshold optimization
# # Input:
# # y_validation_set: true positive-unlabeled observations of the validation set (y[ix_tr_val] or noisy_y[ix_tr_val], for reference)
# # y_pred_prob: predicted positive-negative observations of the validation set
# # p_y1_hat: Estimation of positive labels in the dataset / estimation of positive class prior / estimation of P(y=1)
# # Output:
# # opt_threshold: optimal threshold to binarize the probability

# def IC_threshold_optimization(y_validation_set, y_pred_prob, p_y1_hat):

#     #set thresholds for positive class
#     set_thresholds = np.linspace(0.01, 0.99, num=100)
#     #binarize the probability
#     set_pred_labels = [binarize_prob(prob = y_pred_prob, threshold=i) for i in set_thresholds]
#     #f1 score for each threshold
#     set_f1_scores = []

#     try:
#         for i in set_pred_labels:
#             ic_recall, ic_precision, ic_f1 = IC_evaluation(y_true = y_validation_set, y_pred_prob =i, threshold=False, p_y1_hat = p_y1_hat)
#             set_f1_scores.append(ic_f1)
#     except:
#         print("Probably the error is because alpha is negative. The proportion of labeled is greater than the estimated proportion of positive observations.")
#         return None
    
#     #get the best threshold
#     max_f1score = np.nanmax(set_f1_scores)
#     #index of the best threshold
#     ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
#     #optimal threshold
#     opt_threshold = set_thresholds[ix_fscore]

#     return opt_threshold

# # Indirect Correction Average Precision Score
# # Input:
# # y_true: true positive-unlabeled observations of the validation set (y[ix_tr_val] or noisy_y[ix_tr_val], for reference)
# # y_pred_prob: predicted positive-negative observations of the validation set
# # p_y1_hat: Estimation of positive labels in the dataset / estimation of positive class prior / estimation of P(y=1)
# # Output:
# # ic_ap: indirect correction of average precision score
# # set_recall_precision: array with recall and precision for each thresholded prediction. The order is (i, recall, precision)

# def IC_average_precision_score(y_true, y_pred_prob, p_y1_hat):

#     #set thresholds for positive class
#     set_thresholds = np.linspace(0.01, 0.99, num=100)
#     #binarize the probability
#     set_pred_labels = [binarize_prob(prob = y_pred_prob, threshold=i) for i in set_thresholds]
    
#     #calculate recall and precision for each thresholded prediction
#     set_i = []
#     set_ic_recall_i = []
#     set_ic_precision_i = []

#     try:
#         for i in range(len(set_pred_labels)):
#             ic_recall_i, ic_precision_i, ic_f1_i = IC_evaluation(y_true = y_true, y_pred_prob =set_pred_labels[i], threshold=False, p_y1_hat = p_y1_hat)
#             set_i.append(i)
#             set_ic_recall_i.append(ic_recall_i)
#             set_ic_precision_i.append(ic_precision_i)
#     except:
#         print("Probably the error is because alpha is negative. The proportion of labeled is greater than the estimated proportion of positive observations.")
#         return None
    
#     # Create an array with recall and precision
#     set_recall_precision = np.array(list(zip(set_i, set_ic_recall_i, set_ic_precision_i)))
#     # Sort the array by recall
#     set_recall_precision_sorted = set_recall_precision[set_recall_precision[:,1].argsort()]

#     # Compute differences in recall and multiply by precision
#     increase_recall_times_precision = np.diff(set_recall_precision_sorted[:,1])* set_recall_precision_sorted[1:,2]
#     # Sum to get the average precision
#     ic_ap = np.nansum(increase_recall_times_precision)

#     return ic_ap, set_recall_precision_sorted


# #NOTE: for problems with definition when recall is zero , read https://yardstick.tidymodels.org/reference/average_precision.html


# ##############################################################
# # Lee-Liu-Performance Criteria (LL-PC)
# # According to Lee and Liu (2003). Learning with Positive and Unlabeled Examples Using Weighted Logistic Regression
# ##############################################################

# # Lee and Liu propose a performance criterion that tries to mimic the F1 score in order to evaluate PU learning methods
# # If the f1 score is defined as f1 = 2 * (precision * recall) / (precision + recall),
# # the LL-F1 is defined as LL-F1 = (precision * recall) / P(y = 1) and its equivalent to calculate
# # LL-F1 = recall ^ 2 / P(f(X) = 1)
# # With recall = P(f(X) = 1 | Y = 1) : "can be estimated from the performance of the hypothesis on the positive labeled examples of the validation set"
# # I understand that the calculation of recall is done with the positive labeled examples of the validation / test set, which then asumes SCAR
# # and P(f(X) = 1) "can be estimated from the validation set, giving us an estimate of the desired model selection criteria"
# # Since I'm assuming I know class prior, I can also calculate precision by solving the main equivalence equation

# # Three functions:
# # 1) one calculating the basic performance measures according to LL-PC
# # 2) another to get the best threshold using LL-PC
# # 3) calculate average precision score using LL-PC

# # Lee-Liu-Performance Criteria (LL-PC) evaluation
# # Input:
# # y_true: true positive-unlabeled observations of the test set
# # y_pred: predicted positive-negative observations of the test set. It can be a probability or a binary label
# # threshold: threshold to binarize the probability, it can be a number between 0 and 1 or False, in which case, the LL-PC evaluation will be done with the raw probability
# # p_y1_hat: Estimation of positive labels in the dataset / estimation of positive class prior / estimation of P(y=1)
# # Output:
# # LL_recall: recall of the PU learning method
# # LL_precision: precision of the PU learning method
# # LL_f1: f1 score of the PU learning method


# def LLPC_evaluation(y_true, y_pred_prob, threshold, p_y1_hat):

#     if threshold == False:
#         y_pred = y_pred_prob
#     else:
#         #binarize the probability
#         y_pred = binarize_prob(prob = y_pred_prob, threshold = threshold)
    
#     #recall
#     LL_recall = np.mean(y_pred[y_true == 1]) #P(f(X) = 1 | Y = 1)

#     #precision
#     LL_precision = (LL_recall * p_y1_hat) / np.mean(y_pred) 

#     #LL-F1
#     LL_f1 = LL_recall ** 2 / np.mean(y_pred) 

#     return LL_recall, LL_precision, LL_f1

# # Threshold optimization
# # Input:
# # y_validation_set: true positive-unlabeled observations of the validation set (y[ix_tr_val] or noisy_y[ix_tr_val], for reference)
# # y_pred_prob: predicted positive-negative observations of the validation set
# # p_y1_hat: Estimation of positive labels in the dataset / estimation of positive class prior / estimation of P(y=1)
# # Output:
# # opt_threshold: optimal threshold to binarize the probability

# def LLPC_threshold_optimization(y_validation_set, y_pred_prob, p_y1_hat):

#     #set thresholds for positive class
#     set_thresholds = np.linspace(0.01, 0.99, num=100)
#     #binarize the probability
#     set_pred_labels = [binarize_prob(prob = y_pred_prob, threshold=i) for i in set_thresholds]
#     #f1 score for each threshold
#     set_f1_scores = [LLPC_evaluation(y_true = y_validation_set, y_pred_prob =i, threshold=False ,p_y1_hat = p_y1_hat)[2] for i in set_pred_labels]
#     #get the best threshold
#     max_f1score = np.nanmax(set_f1_scores)
#     #index of the best threshold
#     ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
#     #optimal threshold
#     opt_threshold = set_thresholds[ix_fscore]

#     return opt_threshold

# # Lee-Liu-Performance Criteria (LL-PC) Average Precision Score
# # Input:
# # y_validation_set: true positive-unlabeled observations of the validation set (y[ix_tr_val] or noisy_y[ix_tr_val], for reference)
# # y_pred_prob: predicted positive-negative observations of the validation set
# # p_y1_hat: Estimation of positive labels in the dataset / estimation of positive class prior / estimation of P(y=1)
# # Output:
# # ll_ap: LL-PC average precision score
# # set_recall_precision_sorted: array with recall and precision for each thresholded prediction. The order is (i, recall, precision, increase_recall * precision)

# def LLPC_average_precision_score(y_true, y_pred_prob, p_y1_hat):

#     #set thresholds for positive class
#     set_thresholds = np.linspace(0.01, 0.99, num=100)
#     #binarize the probability
#     set_pred_labels = [binarize_prob(prob = y_pred_prob, threshold=i) for i in set_thresholds]
#     #calculate recall and precision for each thresholded prediction
#     set_i = []
#     set_ll_recall_i = []
#     set_ll_precision_i = []

#     for i in range(len(set_pred_labels)):
#         ll_recall_i, ll_precision_i, ll_f1_i = LLPC_evaluation(y_true = y_true, y_pred_prob =set_pred_labels[i], threshold = False , p_y1_hat = p_y1_hat)
#         set_i.append(i)
#         set_ll_recall_i.append(ll_recall_i)
#         set_ll_precision_i.append(ll_precision_i)
    
#     # Create an array with recall and precision
#     set_recall_precision = np.array(list(zip(set_i, set_ll_recall_i, set_ll_precision_i)))
#     # Sort the array by recall
#     set_recall_precision_sorted = set_recall_precision[set_recall_precision[:,1].argsort()]

#     # Compute differences in recall and multiply by precision
#     increase_recall_times_precision = np.diff(set_recall_precision_sorted[:,1])* set_recall_precision_sorted[1:,2]

#     # Add the increase_recall_times_precision to set_recall_precision_sorted
#     set_recall_precision_sorted = np.column_stack((set_recall_precision_sorted, np.insert(increase_recall_times_precision, 0, np.nan)))

#     # Sum to get the average precision
#     ll_ap = np.nansum(increase_recall_times_precision)

#     return ll_ap, set_recall_precision_sorted

# ##############################################################
# # Naive Top-K evaluation
# ##############################################################

# # These performance metrics are inspired in their use by recommenders systems, 
# # By focusing on the top recomendations of the PU learning method, the evaluation gives information of the ability of the model to retrieve relevant predictions
# # In J. L. Herlocker, et. al. “Evaluating collaborative filtering recommender systems,” 2004
# # the suggestion of the authors is to use only the subset of rated items in the evaluation,
# # in our case, these would correspond to the labeled-positive observations
# # However, using this approach would be able to calculate recall (under SCAR assumption), but not precision (precision would be always 1 because there are no false positives in the subset)
# # Therefore we will use the entire labeled-unlabeled dataset

# # Two functions:
# # 1) one calculating the basic performance measures according to Naive Top-K
# # 2) another to get the best threshold using Naive Top-K

# # Naive Top-K evaluation
# # Input:
# # y_true: true positive-unlabeled observations of the test set
# # y_pred_prob: predicted positive-unlabeled observations of the test set
# # k: number of top-k predictions
# # threshold: threshold to binarize the probability, it can be a number between 0 and 1 
# # Output:
# # naive_topk_recall: recall of the top-k predictions of the PU learning method
# # naive_topk_precision: precision of the top-k predictions of the PU learning method
# # naive_topk_f1: f1 score of the top-k predictions of the PU learning method
# # naive_topk_ap: average precision score of the top-k predictions of the PU learning method


# def naive_topk_evaluation(y_true, y_pred_prob, k, threshold):

#     #combine the arrays
#     a = np.array(list(zip(y_true, y_pred_prob)))

#     #sort by y_pred_prob
#     a = a[a[:,1].argsort()[::-1]]

#     #get the top-k predictions according to y_pred_prob
#     top_k = a[:k]

#     #prop positives in the top-k
#     topk_pl = np.mean(top_k[:,0])
    
#     #binarize the probability
#     top_k_pred = binarize_prob(prob = top_k[:,1], threshold = threshold)

#     #recall
#     naive_topk_recall = recall_score(top_k[:,0], top_k_pred)

#     #precision
#     naive_topk_precision = precision_score(top_k[:,0], top_k_pred)

#     #f1 score
#     naive_topk_f1 = f1_score(top_k[:,0], top_k_pred)

#     #average precision score
#     naive_topk_ap = average_precision_score(top_k[:,0], top_k[:,1])

#     return naive_topk_recall, naive_topk_precision, naive_topk_f1, naive_topk_ap, topk_pl

# # Threshold optimization
# # Input:
# # y_validation_set: true positive-unlabeled observations of the validation set (y[ix_tr_val] or noisy_y[ix_tr_val], for reference)
# # y_pred_prob: predicted positive-unlabeled observations of the validation set
# # k: number of top-k predictions
# # Output:
# # opt_threshold: optimal threshold to binarize the probability

# def naive_topk_threshold_optimization(y_validation_set, y_pred_prob, k):
    
#         #set thresholds for positive class
#         set_thresholds = np.linspace(0.01, 0.99, num=100)

#         #combine the arrays
#         a = np.array(list(zip(y_validation_set, y_pred_prob)))

#         #sort by y_pred_prob
#         a = a[a[:,1].argsort()[::-1]]

#         #get the top-k predictions according to y_pred_prob
#         top_k = a[:k]

#         #binarize the probability
#         set_pred_labels = [binarize_prob(prob = top_k[:,1], threshold=i) for i in set_thresholds]

#         #f1 score for each threshold
#         set_f1_scores = [f1_score(top_k[:,0], set_pred_labels[i]) for i in range(len(set_pred_labels))]

#         #get the best threshold
#         max_f1score = np.nanmax(set_f1_scores)
        
#         #index of the best threshold
#         ix_fscore = np.where(set_f1_scores == max_f1score )[0][0]
        
#         #optimal threshold
#         opt_threshold = set_thresholds[ix_fscore]
    
#         return opt_threshold


# ##############################################################
# # Lift index
# # According to Ling and Li, “Data Mining for Direct Marketing: Problems and Solutions”
# ##############################################################

# # The lift index is a measure of the effectiveness of a classification model 
# # at retrieving the positive class with the highest probability

# # One function:
# # 1) calculating the lift index and the lift table

# # Lift index evaluation
# # Input:
# # y_true: true positive-unlabeled observations of the test set
# # y_pred_prob: predicted positive-unlabeled observations of the test set
# # n_percentile: number of bins to divide the observations
# # Output:
# # lift_index: lift index of the PU learning method
# # lift_table: array ordered in the following columns: bin number, number of positive observations, proportion of positive observations in one bin among all bins, cummulative proportion of positive observations
# # perfect_classifier: array with the shape [percentage of positive observations, n_percentile in which the cumulative sum of positives should be 1]. 
# # A perfect classifier would have cover the 100% of positive observations in the n_percentile equivalent to the percentage of positive observations (e.g. if 30% are positive, in the 30th percentile all the positive observations should have been covered)

# def lift_index_evaluation(y_true, y_pred_prob, n_percentile):

#     #combine the arrays
#     a = np.array(list(zip(y_true, y_pred_prob)))

#     #sort by y_pred_prob
#     a = a[a[:,1].argsort()[::-1]]

#     #create n_percentile
#     a = np.array_split(a, n_percentile)

#     #get number of positive observations in each bin
#     n_pos = [np.sum(i[:,0]) for i in a]

#     #proportion of positive observations in each bin
#     prop_n_pos = [i / np.sum(n_pos) for i in n_pos]

#     #cummulative sum of prop_n_pos
#     cum_prop_n_pos = np.cumsum(prop_n_pos)
    
#     #get an array with the number of n_percentile and prop_pos
#     lift_table = np.array(list(zip(np.arange(1, n_percentile+1), n_pos, prop_n_pos, cum_prop_n_pos)))

#     #perfect classifier, the percentage of positives, and the bin in which the perfect classifier would be
#     perfect_classifier = [np.mean(y_true), np.mean(y_true)*n_percentile] 

#     #lift index: sum of the product the positive observations with the weighted value of the bin divided by the sum of the positive observations
#     lift_index = np.sum((np.arange(1, n_percentile+1)[::-1]*(1/n_percentile))*n_pos) / np.sum(n_pos)

#     return lift_index, lift_table, perfect_classifier




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


# def uniform_train_test(df, random_state, target_column, split_proportion=None, qcutoff=None, group_col = 'supplier_name_clean'):

#     assert df[target_column].isnull().sum() == 0, "Target column contains null values. Please clean the target variable before proceeding."
    
#     #Get the Utrain and Utest
#     Utrain, Utest = stratified_company_split(df = df, random_state = random_state, target_column = target_column, split_proportion = split_proportion)
#     #Get the bottom and top contracts
#     qnum, topq_maxcontracts, topq_contracts, bottomq_contracts = qtop_bottom_split(df=Utrain, qcutoff=qcutoff, target_column=target_column)
#     #get the top uniform
#     topq_UniformContracts = uniform_sampling(df = topq_contracts, k = qnum, random_state = random_state, label_col = target_column, group_col = group_col)
#     #make the final train and test
#     train_test = pd.concat([topq_UniformContracts, bottomq_contracts]).reset_index(drop = True)

#     return train_test, Utest

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


# ##############################################################
# # Naive Top-K evaluation ONLY AVERAGE PRECISION SCORE
# ##############################################################

# # These performance metrics are inspired in their use by recommenders systems, 
# # By focusing on the top recomendations of the PU learning method, the evaluation gives information of the ability of the model to retrieve relevant predictions
# # In J. L. Herlocker, et. al. “Evaluating collaborative filtering recommender systems,” 2004
# # the suggestion of the authors is to use only the subset of rated items in the evaluation,
# # in our case, these would correspond to the labeled-positive observations
# # However, using this approach would be able to calculate recall (under SCAR assumption), but not precision (precision would be always 1 because there are no false positives in the subset)
# # Therefore we will use the entire labeled-unlabeled dataset

# # Two functions:
# # 1) one calculating the basic performance measures according to Naive Top-K
# # 2) another to get the best threshold using Naive Top-K

# # Naive Top-K evaluation
# # Input:
# # y_true: true positive-unlabeled observations of the test set
# # y_pred_prob: predicted positive-unlabeled observations of the test set
# # k: number of top-k predictions
# # threshold: threshold to binarize the probability, it can be a number between 0 and 1 
# # Output:
# # naive_topk_recall: recall of the top-k predictions of the PU learning method
# # naive_topk_precision: precision of the top-k predictions of the PU learning method
# # naive_topk_f1: f1 score of the top-k predictions of the PU learning method
# # naive_topk_ap: average precision score of the top-k predictions of the PU learning method


# def naive_topk_evaluation_AP(y_true, y_pred_prob, k):

#     #combine the arrays
#     a = np.array(list(zip(y_true, y_pred_prob)))

#     #sort by y_pred_prob
#     a = a[a[:,1].argsort()[::-1]]

#     #get the top-k predictions according to y_pred_prob
#     top_k = a[:k]

#     #average precision score
#     naive_topk_ap = average_precision_score(top_k[:,0], top_k[:,1])

#     return naive_topk_ap



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

# ###############################################################
# # Relative Information Score
# ###############################################################
# #The relative information score is the ratio of the average information score over all n test cases x_i and the entropy of the prior class distribution

# def calculate_rel_info_score(y_true, y_pred, pos_class_prior=None):
#     if pos_class_prior is None:
#         pos_class_prior = np.mean(y_true)
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     class_priors = np.where(y_true == 1, pos_class_prior, 1 - pos_class_prior)
#     robust_y_pred = np.clip(y_pred, 10e-5, 1 - 10e-5)  # Avoid log(0)
#     #Calculate the terms regardless the condition
#     useful_score = -np.log2(class_priors) + np.log2(robust_y_pred)
#     misleading_score = np.log2(1 - class_priors) - np.log2(1 - robust_y_pred)
#     #create indicators
#     useful_indicator = np.where(y_true == 1, 1, 0)
#     misleading_indicator = np.where(y_true == 0, 1, 0)
#     #Calculate the information score
#     information_score = (useful_indicator*useful_score) + (misleading_indicator*misleading_score)
#     #Calculate the mean information score
#     mean_information_score = np.mean(information_score)
#     #Calculate the entropy of the class prior
#     entropy_class_prior = -(pos_class_prior * np.log2(pos_class_prior)) - ((1 - pos_class_prior) * np.log2(1 - pos_class_prior))
#     relative_information_score = mean_information_score / entropy_class_prior
#     #Return the relative information score and mean information score
#     return mean_information_score, relative_information_score


# ##############################################################
# # Gain Chart, Lift Chart, Average Gain and Average Lift
# # According to Berrar, Daniel (2018). Performance Measures for Binary Classification
# ##############################################################
# # The gain chart is a graphical representation of the cumulative gain of a classification model
# # It shows how many positive instances are captured by the model as a function of the number of instances considered
# # The average gain is the average of the difference between the cumulative gain of the model and the cumulative gain of the null model
# # The lift chart is a graphical representation of the lift of a classification model
# # It shows how much better the model is at capturing positive instances than random guessing
# # The average lift is the average of the lifts of the instances considered
# # This function outputs the gain chart and the lift chart, as well as the average gain and average lift
# # Input:
# # y_true: true positive-unlabeled observations of the test set
# # y_pred_prob: predicted positive-unlabeled observations of the test set (scores)
# # top_k: number of instances to be considered
# # prevalence: proportion of positive observations in the dataset (P(y=1))
# # Output:
# # gain_coordinates: array with the shape [top_k, 3], where the first column is an array of the considered instances, the second column is the cumulative gain of the model, and the third column is the cumulative gain of the null model
# # average_gain: average gain of the model
# # lift_coordinates: array with the shape [top_k, 2], where the first column is an array of the considered instances, and the second column is the lift of the model

# def ranking_evaluations(y_true, y_pred_prob, top_k, prevalence = None):
#     #GENERAL
#     # Ensure y_true and y_pred_prob are numpy arrays
#     y_true = np.array(y_true)
#     y_pred_prob = np.array(y_pred_prob)
#     assert len(y_true) == len(y_pred_prob), "y_true and y_pred_prob must have the same length"
#     #combine the arrays
#     a = np.array(list(zip(y_true, y_pred_prob)))
#     #sort by y_pred_prob
#     a = a[a[:,1].argsort()[::-1]]
#     #take only the top-k predictions according to y_pred_prob
#     a = a[:top_k]

#     # Cumulative gain and lift curve
#     #get the cummulative sum of y_true
#     cumsum = np.cumsum(a[:,0])  # Normalization comes after

#     #ranking
#     x_charts = np.arange(1, top_k + 1, step = 1)
#     #Get expected cumulative sum of null model
#     if prevalence is None:
#         prevalence = np.mean(y_true)  # If prevalence is not provided, calculate it from y_true
#     null_cumsum = np.arange(prevalence, top_k + prevalence, step = prevalence)[:top_k]  # Expected cumulative sum for 

#     # Create group limits
#     y_pred_prob_sorted = a[:,1]  # The sorted probabilities
#     unique_groups, group_limits = np.unique(y_pred_prob_sorted, return_index=True)
#     group_limits = list(group_limits - 1)  # Adjust to get the last index of each group
#     group_limits.remove(-1)
#     group_limits = group_limits + [0] + [len(y_pred_prob_sorted) - 1]  # Add the last index of the array
#     group_limits = list(set(group_limits))  # Remove duplicates
#     group_limits = sorted(group_limits)

#     cumsum_group_limits = cumsum[group_limits]
#     # Compute lengths of each group (difference between consecutive limits)
#     lengths = np.diff(group_limits)
#     # Repeat each cumsum value according to its group length
#     fixed_cumsum = np.repeat(cumsum_group_limits[:-1], lengths)
#     fixed_cumsum = np.append(fixed_cumsum, cumsum_group_limits[-1])  # Append the last value to match the length of x_charts
        
#     # Gain
#     gain = fixed_cumsum - null_cumsum
#     average_gain = np.mean(gain) / y_true.sum() #normalized average gain

#     # Lift
#     lift = (fixed_cumsum / x_charts) / prevalence
#     average_lift = np.mean(lift)

#     # # Gain chart coordinates
#     y_gainchart = fixed_cumsum / y_true.sum()
#     y_null_gainchart = null_cumsum / y_true.sum()

#     # #Lift chart coordinates
#     y_liftchart = lift #/ top_k

#     #normalize x coordinates
#     x_charts = x_charts / top_k

#     gain_coordinates = np.column_stack((x_charts, y_gainchart, y_null_gainchart))
#     lift_coordinates = np.column_stack((x_charts, y_liftchart))


#     return gain_coordinates, average_gain, lift_coordinates, average_lift


# def mini_ranking_evaluations(y_true, y_pred_prob, top_k, prevalence = None):
#     #GENERAL
#     # Ensure y_true and y_pred_prob are numpy arrays
#     y_true = np.array(y_true)
#     y_pred_prob = np.array(y_pred_prob)
#     assert len(y_true) == len(y_pred_prob), "y_true and y_pred_prob must have the same length"
#     #combine the arrays
#     a = np.array(list(zip(y_true, y_pred_prob)))
#     #sort by y_pred_prob
#     a = a[a[:,1].argsort()[::-1]]
#     #take only the top-k predictions according to y_pred_prob
#     a = a[:top_k]

#     # Cumulative gain and lift curve
#     #get the cummulative sum of y_true
#     cumsum = np.cumsum(a[:,0])  # Normalization comes after

#     #ranking
#     x_charts = np.arange(1, top_k + 1, step = 1)
#     #Get expected cumulative sum of null model
#     if prevalence is None:
#         prevalence = np.mean(y_true)  # If prevalence is not provided, calculate it from y_true
#     null_cumsum = np.arange(prevalence, top_k + prevalence, step = prevalence)[:top_k]  # Expected cumulative sum for 

#     # Create group limits
#     y_pred_prob_sorted = a[:,1]  # The sorted probabilities
#     unique_groups, group_limits = np.unique(y_pred_prob_sorted, return_index=True)
#     group_limits = list(group_limits - 1)  # Adjust to get the last index of each group
#     group_limits.remove(-1)
#     group_limits = group_limits + [0] + [len(y_pred_prob_sorted) - 1]  # Add the last index of the array
#     group_limits = list(set(group_limits))  # Remove duplicates
#     group_limits = sorted(group_limits)

#     cumsum_group_limits = cumsum[group_limits]
#     # Compute lengths of each group (difference between consecutive limits)
#     lengths = np.diff(group_limits)
#     # Repeat each cumsum value according to its group length
#     fixed_cumsum = np.repeat(cumsum_group_limits[:-1], lengths)
#     fixed_cumsum = np.append(fixed_cumsum, cumsum_group_limits[-1])  # Append the last value to match the length of x_charts
        
#     # Gain
#     gain = fixed_cumsum - null_cumsum
#     average_gain = np.mean(gain) / y_true.sum() #normalized average gain

#     # Lift
#     lift = (fixed_cumsum / x_charts) / prevalence
#     average_lift = np.mean(lift)

#     # Gain chart coordinates
#     #y_gainchart = cumsum / a[:,0].sum()
#     #y_null_gainchart = null_cumsum / a[:,0].sum()

#     #Lift chart coordinates
#     #y_liftchart = lift #/ top_k

#     #normalize x coordinates
#     #x_charts = x_charts / top_k

#     #gain_coordinates = np.column_stack((x_charts, y_gainchart, y_null_gainchart))
#     #lift_coordinates = np.column_stack((x_charts, y_liftchart))

#     return average_gain, average_lift



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