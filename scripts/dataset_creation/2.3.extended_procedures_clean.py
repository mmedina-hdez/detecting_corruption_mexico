import pandas as pd
import numpy as np
from pyprojroot import here
import utils

processed_data = here('data/processed_data')

#import procedures_apf
procedures_apf = pd.read_csv(processed_data / 'procedures_apf.csv', encoding='latin-1')

######################
#There are only some columns that interest me from both datasets
#tender_id
#tender_numberOfTenderers
#tender_procurementMethod
#tender_tenderPeriod_startDate
#tender_tenderPeriod_endDate
#tender_awardPeriod_endDate
#contracts_dateSigned (this is not present in apf dataset)
#awards_id
######################

procedures_apf = procedures_apf[[ 
    'tender_id',
    'tender_numberOfTenderers',
    'tender_procurementMethod',
    'tender_tenderPeriod_startDate', 
    'tender_tenderPeriod_endDate',
    'tender_awardPeriod_endDate',
    'awards_contractPeriod_startDate',        
    'awards_id',
    'awards_per_tender',
    'buyer_parties_contactPoint_name_clean',
    'buyer_parties_roles']] 

#tender_tenderPeriod_startDate
#keep the first 10 characters
procedures_apf['tender_tenderPeriod_startDate'] = procedures_apf['tender_tenderPeriod_startDate'].str[:10]
procedures_apf['tender_tenderPeriod_startDate'] = pd.to_datetime(procedures_apf['tender_tenderPeriod_startDate'], errors='raise')
#tender_tenderPeriod_endDate
#keep the first 10 characters
procedures_apf['tender_tenderPeriod_endDate'] = procedures_apf['tender_tenderPeriod_endDate'].str[:10]
procedures_apf['tender_tenderPeriod_endDate'] = pd.to_datetime(procedures_apf['tender_tenderPeriod_endDate'], errors='raise')
#tender_awardPeriod_endDate
#how many errors it has in the column
#utils.how_many_errors(procedures_merged, 'tender_awardPeriod_endDate')
#it has 74 errors, minimal, so I will coerce
procedures_apf['tender_awardPeriod_endDate'] = procedures_apf['tender_awardPeriod_endDate'].str[:10]
procedures_apf['tender_awardPeriod_endDate'] = pd.to_datetime(procedures_apf['tender_awardPeriod_endDate'], errors='coerce')
#contracts_contractPeriod_startDate
#keep the first 10 characters
procedures_apf['awards_contractPeriod_startDate'] = procedures_apf['awards_contractPeriod_startDate'].str[:10]
procedures_apf['awards_contractPeriod_startDate'] = pd.to_datetime(procedures_apf['awards_contractPeriod_startDate'], errors='raise')

# #drop duplicates
# print('size before dropping duplicates pdn:', procedures_pdn.shape)
print('size before dropping duplicates apf:', procedures_apf.shape)
# procedures_pdn = procedures_pdn.drop_duplicates()
procedures_apf = procedures_apf.drop_duplicates()
# print('size after dropping duplicates pdn:', procedures_pdn.shape)
print('size after dropping duplicates apf:', procedures_apf.shape)

#save
procedures_apf.to_csv(processed_data / 'procedures_apf_clean.csv', index=False)