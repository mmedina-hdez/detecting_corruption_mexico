# Useful functions for the project

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# SWITCH MIXED COLUMNS
#This function is used specifically to clean the rfc and proveedor o contratista columns
#In some datasets rfc and proveedor o contratista are mixed, so we need to switch them in the appropiate cases
#Input:
#   df: dataframe
#   first_column: in this case, rfc
#   backup_first_column: in this case, rfc_original
#   second_column: in this case, proveedor o contratista
#Output:
#   df: dataframe with the columns switched

def switch_mixed_columns(df, first_column, backup_first_column, second_column ):
    #remove spaces in rfc
    df[first_column] = df[first_column].str.replace(' ', '')
    #keep only values that have 12 or 13 characters in rfc column
    df[first_column] = np.where(df[first_column].str.len().isin([12, 13]), df[first_column], np.nan)
    #if the value has values that doesnt contain at least one number, then write np.nan
    df[first_column] = np.where(df[first_column].str.contains(r'\d'), df[first_column], np.nan)
    
    #rfc_from_supplier_name
    temporal_column = first_column + '_from_' + second_column
    #remove spaces in supplier_name
    df[temporal_column] = df[second_column].str.replace(' ', '')
    #keep only values that have 12 or 13 characters in rfc column
    df[temporal_column] = np.where(df[temporal_column].str.len().isin([12, 13]), df[temporal_column], np.nan)
    #if the value has values that doesnt contain at least one number, then write np.nan
    df[temporal_column] = np.where(df[temporal_column].str.contains(r'\d'), df[temporal_column], np.nan)

    #fill missing values
    df[first_column] = np.where(df[first_column].isnull(), df[temporal_column], df[first_column])

    #CHANGE SECOND COLUMN
    #if RFC has the same value than provedor o contratista, then write np.nan in proveedor o contratista
    df[second_column] = np.where(df[first_column] == df[second_column], np.nan, df[second_column])

    #if proveedor o contratista is missing, then fill it with RFC original
    df[second_column] = np.where(df[second_column].isnull(), df[backup_first_column], df[second_column])


    #remove temporal column
    df = df.drop(columns=temporal_column)

    print(df[[first_column, second_column]])

    return df

# HOW MANY ERRORS IN pd.to_datetime
#This function is used to count how many errors are in a column if we use pd.to_datetime
#Input:
#   df: dataframe
#   column: column to check
#Output:
#   errors_of_date_input: number of errors in the column
def how_many_errors(df, column):
    errors_of_date_input = 0
    for i, date in enumerate(df[column]):
        try:
            pd.to_datetime(date)
        except Exception as e:
            print(f"Error at index {i}: {date}")
        #print(e)
            errors_of_date_input += 1
    return errors_of_date_input

# CLEAN NAMES
#This function is used to create a clean name for a column
#Input:
#   df: dataframe
#   column: column to clean
#Output:
#   new_column: column cleaned

# def clean_names(df, column):
#     replace_list = ['&','+','/','-','¿',"'",'"','.',',','(',')','@','|','•', '', '', '\t', '\x03']
#     new_column = df[column].str.lower().str.replace(' ', '').str.replace('ñ','n').str.replace('á','a').str.replace('é','e').str.replace('í','i').str.replace('ó','o').str.replace('ú','u').str.replace('ü','u')
#     for r in replace_list:
#         new_column = new_column.str.replace(r,'')
#     return new_column


def clean_names(df, column):
    replace_list = ['&','+','/','-','¿',"'",'"','.',',','(',')','@','|','•', '', '', '\t', '\x03']
    
    new_column = (
        df[column]
        .astype('string')
        .str.lower()
        .str.replace(' ', '', regex=False)
        .str.replace('ñ', 'n', regex=False)
        .str.replace('á', 'a', regex=False)
        .str.replace('é', 'e', regex=False)
        .str.replace('í', 'i', regex=False)
        .str.replace('ó', 'o', regex=False)
        .str.replace('ú', 'u', regex=False)
        .str.replace('ü', 'u', regex=False)
    )

    for r in replace_list:
        new_column = new_column.str.replace(r, '', regex=False)

    return new_column

# NON-DUPLICATE DATAFRAME
#This function is used to create a dataframe without duplicates
#Input:
#   source_df: dataframe
#   key_column: column used to merge the dataset
#   interest_column: column of interest
#   lower_to_higher: boolean to sort the dataset, if true the dataset will be sorted from lower to higher values of interest column
def non_duplicate_df(source_df = pd.DataFrame(), key_column = [] , interest_column = [], filter_duplicate_all = True, lower_to_higher=True):
    df = source_df[key_column + interest_column].dropna()
    df = df.drop_duplicates()
    complete_df = len(df)
    print('len df: ', complete_df)
    if filter_duplicate_all == False:
        df = df.sort_values(by=interest_column, ascending=lower_to_higher).reset_index(drop=True)
        df = df.drop_duplicates(subset=key_column, keep='first')
        len_non_duplicate_df = len(df)
        print('there are', complete_df, 'rows without nan and without duplicates.')
        print('We will delete ', complete_df - len_non_duplicate_df, 'rows that have different ', str(interest_column),  ' even with same ', str(key_column))
        print('We are keeping lower_to_higher = ', lower_to_higher)
    return df


######################################################
#This function is used to create a dataframe that allows to see the residuals of a model compared with a specific variable
#Input:
#   preprocessed_df: preprocessed dataframe, this means there's no misssing values and no variables without variance
#   variable2threshold: variable to compare with residuals
#   dependent_variable: dependent variable of the model
#   control_variables: control variables of the model
#   number_of_quantiles: number of quantiles to divide the variable2threshold
#   thresholds_m: list of thresholds to divide the variable2threshold
#   thresholds_labels: labels of the thresholds
#Output:
#   df2process: dataframe with the residuals of the model
def variable_vs_residuals_df(preprocessed_df,
                            variable2threshold,
                            dependent_variable = ['single_bidder'],
                            control_variables = None,
                            number_of_quantiles = 50,
                            thresholds_m = None,
                            thresholds_labels = None):
    
    formula = ' '.join(dependent_variable) + ' ~ ' + '+'.join(preprocessed_df[[variable2threshold] + control_variables].columns)
    print(formula)


    logit_model = smf.logit(formula, data= preprocessed_df).fit()

    if isinstance(thresholds_m, list) == True:
        preprocessed_df['thresholds_levels'] = pd.cut(
            preprocessed_df[variable2threshold],
            bins=thresholds_m,
            labels=thresholds_labels)

        df2logit = preprocessed_df.copy()
        formula2logit = formula.replace(variable2threshold, 'thresholds_levels')
        logit_model = smf.logit(formula2logit, data= df2logit).fit()

    
        #keep only the variable to plot
        df2process = preprocessed_df[[variable2threshold, 'thresholds_levels']]
        #add the residuals
        df2process['residuals'] = logit_model.resid_response
        #order by variable2threshold
        df2process = df2process.sort_values(by=variable2threshold).reset_index()
        #get mean of residuals by percetile and maximum and min value of the variable2threshold in that percentile
        df2process = df2process.groupby(pd.qcut(x = df2process.index, q = number_of_quantiles)).agg(mean_residuals=('residuals', 'mean'), min_limit=(variable2threshold, 'min'), max_limit=(variable2threshold, 'max'), t_labels = ('thresholds_levels', 'first' )).reset_index()
        #x labels are min_limit and max_limit
        df2process['x_labels'] = df2process['min_limit'].astype(str) + ' - ' + df2process['max_limit'].astype(str)
        #create column with positive or negative residuals
        df2process['sign'] = np.where(df2process['mean_residuals'] > 0, 1, 0)

    else:
        #keep only the variable to plot
        df2process = preprocessed_df[[variable2threshold]]
        #add the residuals
        df2process['residuals'] = logit_model.resid_response
        #order by variable2threshold
        df2process = df2process.sort_values(by=variable2threshold).reset_index()
        #get mean of residuals by percetile and maximum and min value of the variable2threshold in that percentile
        df2process = df2process.groupby(pd.qcut(x = df2process.index, q = number_of_quantiles)).agg(mean_residuals=('residuals', 'mean'), min_limit=(variable2threshold, 'min'), max_limit=(variable2threshold, 'max')).reset_index()
        #x labels are min_limit and max_limit
        df2process['x_labels'] = df2process['min_limit'].astype(str) + ' - ' + df2process['max_limit'].astype(str)
        #create column with positive or negative residuals
        df2process['sign'] = np.where(df2process['mean_residuals'] > 0, 1, 0)


    return df2process


#########################################################################
# This function is used to plot the residuals of a model compared with a specific variable
# Input:
#   df_varVSresiduals: dataframe with the residuals of the model
                        #x axis has to be called 'index'
                        #y axis has to be called 'mean_residuals'
                        #x axis labels has to be called 'x_labels'
#   variable2threshold: variable to compare with residuals
#   wthresholds_plot: boolean to plot with thresholds or not
# Output:
#   fig: figure with the plot

def plot_variableVSresiduals(df_varVSresiduals, 
                             variable2threshold,
                             wthresholds_plot):

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 10))
    #plot

    if wthresholds_plot == False:
        ax = sns.barplot(x="index", #these are the bins
                         y="mean_residuals",
                         data= df_varVSresiduals,
                         dodge=False,
                         color='#4c72b0')

    if wthresholds_plot == True:
        ax = sns.barplot(x="index", #these are the bins
                         y="mean_residuals",
                         data= df_varVSresiduals,
                         hue='t_labels',
                         dodge=False)
        ax.legend(title='Thresholds', title_fontsize='large', loc='upper right', fontsize='large')
        
    #label of axis
    x_label_title = variable2threshold + ' percentiles'
    ax.set(xlabel= x_label_title, ylabel='Residuals')
    #labels of ticks
    ax.set_xticklabels(df_varVSresiduals['x_labels'])
    #x axis labels rotation
    plt.xticks(rotation=60)
    #title
    if wthresholds_plot == False:
        ax.set_title('Residuals vs ' + variable2threshold)
    if wthresholds_plot == True:
        ax.set_title('Residuals vs ' + variable2threshold + ' with thresholds')


