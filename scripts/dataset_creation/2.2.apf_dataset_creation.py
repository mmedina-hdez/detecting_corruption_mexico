# APF - hacienda dataset 
#The purpose of this notebook is to combine all the json datasets of the procurement process and create a csv file

import pandas as pd
import json
import numpy as np
from pyprojroot import here
import utils
from tqdm import tqdm

apf_data = here('data/extended_procedures')
processed_data = here('data/processed_data')

#Function to create the dataset
#This function will create a dataset from the json files

def get_df(data, scope, package = 1):

    #general info
    package_id_list = []
    tender_dataset_id_list = []
    award_dataset_id_list = []
    awards_per_tender_list = []
    ocid_list = []
    publishedDate_list = []
    
    
    #tender
    tender_id_list = []
    tender_tenderPeriod_startDate_list = []
    tender_tenderPeriod_endDate_list = []
    tender_awardPeriod_endDate_list = []
    tender_numberOfTenderers_list = []
    tender_procurementMethod_list = []

    #awards
    awards_id_list = []
    awards_suppliers_id_list = []
    awards_suppliers_name_list = []
    awards_value_amount_list = []
    awards_contractPeriod_startDate_list = []
    awards_contractPeriod_endDate_list = []
    
 
    #buyer
    buyer_id_list = []
    buyer_parties_contactPoint_name_list = []
    buyer_parties_roles_list = []

    unsuccessful_tender = []
 

    for i in range(scope):
        #number of contracts
        number_of_cases = ''
        try:
            number_of_cases = len(data[i]['releases'][0]['awards'])
        except :
            number_of_cases = ''

        try:

            for j in range(number_of_cases):
                
                #general info
                package_id = package
                tender_dataset_id = i
                award_dataset_id = j
                awards_per_tender = number_of_cases

                try:
                    publishedDate = data[i]['releases'][0]['date']
                except :
                    publishedDate = np.nan
                
                try:
                    ocid = data[i]['releases'][0]['ocid']
                except :
                    ocid = np.nan

                #tender        
                try:
                    tender_id = data[i]['releases'][0]['tender']['id']
                except :
                    tender_id = np.nan
                try:
                    tender_tenderPeriod_startDate = data[i]['releases'][0]['tender']['tenderPeriod']['startDate']
                except :
                    tender_tenderPeriod_startDate = np.nan
                try:
                    tender_tenderPeriod_endDate = data[i]['releases'][0]['tender']['tenderPeriod']['endDate']
                except :
                    tender_tenderPeriod_endDate = np.nan
                try:
                    tender_awardPeriod_endDate = data[i]['releases'][0]['tender']['awardPeriod']['endDate']
                except :
                    tender_awardPeriod_endDate = np.nan

                try:
                    tender_numberOfTenderers = data[i]['releases'][0]['tender']['numberOfTenderers']
                except :
                    tender_numberOfTenderers = np.nan
                try:
                    tender_procurementMethod = data[i]['releases'][0]['tender']['procurementMethod']
                except :
                    tender_procurementMethod = np.nan
                
                #awards
                try:
                    awards_id = data[i]['releases'][0]['awards'][j]['id']
                except :
                    awards_id = np.nan

                try:
                    awards_suppliers_id = data[i]['releases'][0]['awards'][j]['suppliers'][0]['id']
                except :
                    awards_suppliers_id = np.nan
                try:
                    awards_suppliers_name = data[i]['releases'][0]['awards'][j]['suppliers'][0]['name']
                except :
                    awards_suppliers_name = np.nan
                try:
                    awards_value_amount = data[i]['releases'][0]['awards'][j]['value']['amount']
                except :
                    awards_value_amount = np.nan
                try:
                    awards_contractPeriod_startDate = data[i]['releases'][0]['awards'][j]['contractPeriod']['startDate']
                except :
                    awards_contractPeriod_startDate = np.nan
                try:
                    awards_contractPeriod_endDate = data[i]['releases'][0]['awards'][j]['contractPeriod']['endDate']
                except :
                    awards_contractPeriod_endDate = np.nan
                

                #buyer
                try:
                    buyer_id = data[i]['releases'][0]['buyer']['id']
                except :
                    buyer_id = np.nan

                try:
                    buyer_parties_contactPoint_name = data[i]['releases'][0]['parties'][0]['contactPoint']['name']
                except :
                    buyer_parties_contactPoint_name = np.nan

                try:
                    buyer_parties_roles = data[i]['releases'][0]['parties'][0]['roles']
                except :
                    buyer_parties_roles = np.nan


                #append
                package_id_list.append(package_id)
                tender_dataset_id_list.append(tender_dataset_id)
                award_dataset_id_list.append(award_dataset_id)
                awards_per_tender_list.append(awards_per_tender)
                ocid_list.append(ocid)
                publishedDate_list.append(publishedDate)
                tender_id_list.append(tender_id)
                tender_tenderPeriod_startDate_list.append(tender_tenderPeriod_startDate)
                tender_tenderPeriod_endDate_list.append(tender_tenderPeriod_endDate)
                tender_awardPeriod_endDate_list.append(tender_awardPeriod_endDate)
                tender_numberOfTenderers_list.append(tender_numberOfTenderers)
                tender_procurementMethod_list.append(tender_procurementMethod)
                awards_id_list.append(awards_id)
                awards_suppliers_id_list.append(awards_suppliers_id)
                awards_suppliers_name_list.append(awards_suppliers_name)
                awards_value_amount_list.append(awards_value_amount)
                awards_contractPeriod_startDate_list.append(awards_contractPeriod_startDate)
                awards_contractPeriod_endDate_list.append(awards_contractPeriod_endDate)
                buyer_id_list.append(buyer_id)
                buyer_parties_contactPoint_name_list.append(buyer_parties_contactPoint_name)
                buyer_parties_roles_list.append(buyer_parties_roles)

                lists = [tender_dataset_id_list, award_dataset_id_list, awards_per_tender_list, ocid_list, tender_id_list, tender_tenderPeriod_startDate_list, tender_tenderPeriod_endDate_list, tender_awardPeriod_endDate_list, tender_numberOfTenderers_list, tender_procurementMethod_list, awards_id_list, awards_suppliers_id_list, awards_suppliers_name_list, awards_value_amount_list, buyer_id_list, publishedDate_list, buyer_parties_contactPoint_name_list, buyer_parties_roles_list, awards_contractPeriod_startDate_list, awards_contractPeriod_endDate_list]

                if all(len(lst) != len(lists[0]) for lst in lists):
                    print('Not all lists have the same length in ' + str(i))

        except:
            unsuccessful_tender.append(i)
            
  
    df = pd.DataFrame({
        'package_id': package_id_list,
        'tender_dataset_id': tender_dataset_id_list,
        'award_dataset_id': award_dataset_id_list,
        'publishedDate': publishedDate_list,
        'awards_per_tender': awards_per_tender_list,
        'ocid': ocid_list,
        'tender_id': tender_id_list,
        'tender_tenderPeriod_startDate': tender_tenderPeriod_startDate_list,
        'tender_tenderPeriod_endDate': tender_tenderPeriod_endDate_list,
        'tender_awardPeriod_endDate': tender_awardPeriod_endDate_list,
        'tender_numberOfTenderers': tender_numberOfTenderers_list,
        'tender_procurementMethod': tender_procurementMethod_list,
        'awards_id': awards_id_list,
        'awards_suppliers_id': awards_suppliers_id_list,
        'awards_suppliers_name': awards_suppliers_name_list,
        'awards_value_amount_mx': awards_value_amount_list,
        'awards_contractPeriod_startDate': awards_contractPeriod_startDate_list,
        'awards_contractPeriod_endDate': awards_contractPeriod_endDate_list,
        'buyer_id': buyer_id_list,
        'buyer_parties_contactPoint_name': buyer_parties_contactPoint_name_list,
        'buyer_parties_roles': buyer_parties_roles_list
        }
        )
    return df

#Create the dataset
procedures_df = pd.DataFrame()
#open json source file
for i in tqdm(range(1,111), desc='files'):
    with open(apf_data / ('contratacionesabiertas_bulk_paquete' + str(i) + '.json')) as file:
        data = json.load(file)
    procedures_df = pd.concat([procedures_df, get_df(data, len(data), package=i)], axis=0)

print('size of df: ' + str(procedures_df.shape))

#clean names
procedures_df['buyer_parties_contactPoint_name_clean'] = utils.clean_names(procedures_df, 'buyer_parties_contactPoint_name')

#save the dataframe as csv with encoding latin-1
procedures_df.to_csv(processed_data / 'procedures_apf.csv', index=False, encoding = 'latin-1')

    