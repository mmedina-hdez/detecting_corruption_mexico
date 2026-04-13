#This script has the objective of merging all the pcs files into one, and then clean it.
#year of the sanction should be from notification, since it is later than the file number

import pandas as pd
import numpy as np
from pyprojroot import here
import utils
sanctions_data = here('data/sanctions_data')
processed_data = here('data/processed_data')

#pcs DA April2023
pcs_da_april2023 = pd.read_csv(sanctions_data / 'proveedores_contratistas_sancionados_datosabiertos_06042023_manualcleaning.csv', encoding='latin1')
#pcs DA February2024
pcs_da_february2024 = pd.read_csv(sanctions_data / 'proveedores_contratistas_sancionados_datosabiertos_24022024_manualcleaning.csv', encoding='latin1')
#pcs Webpage February2024
pcs_w_february2024 = pd.read_csv(sanctions_data / 'Proveedores_Contratistas_Sancionados_webpage.csv', encoding='latin1')
#pcs falcon et al 2021
pcs_f_march2021 = pd.read_csv(sanctions_data / 'proveedores_contratistas_sancionados_falconetal_06032021.csv', encoding='latin1')


##############################
#sanctions DA April2023
#What interests me is:
# year of notification, 
# file number, 
# supplier name
# and supplier name clean
##############################

#remove from column names and column values '\t'
pcs_da_april2023.columns = pcs_da_april2023.columns.str.replace('\t', '')
pcs_da_april2023.replace('\t', '', regex=True, inplace=True)

#remove rows with SENTIDO DE RESOLUCION == 'ABSOLUTORIA'
pcs_da_april2023 = pcs_da_april2023[pcs_da_april2023[' SENTIDO DE RESOLUCION'] != 'ABSOLUTORIA']


#file number
pcs_da_april2023['file_number'] = pcs_da_april2023[' NUMERO DE EXPEDIENTE'].copy()

#supplier name
pcs_da_april2023['supplier_name'] = pcs_da_april2023['PROVEEDOR O CONTRATISTA'].copy()

#supplier name clean
pcs_da_april2023['supplier_name_clean'] = utils.clean_names(pcs_da_april2023, 'supplier_name')

#source
pcs_da_april2023['source'] = 'da_april2023'

#resolution
pcs_da_april2023['resolution'] = pcs_da_april2023[' SENTIDO DE RESOLUCION'].copy()
#time
pcs_da_april2023['sanction_time'] = pcs_da_april2023[' PLAZO'].copy()

pcs_da_april2023 = pcs_da_april2023[['file_number', 'supplier_name', 'supplier_name_clean', 'source', 'sanction_time', 'resolution']]

##############################
#sanctions DA February2024
#What interests me is:
# year of notification, 
# file number, 
# supplier name
# and supplier name clean
##############################

#remove from column names and column values '\t'
pcs_da_february2024.columns = pcs_da_february2024.columns.str.replace('\t', '')
pcs_da_february2024.replace('\t', '', regex=True, inplace=True)

#remove rows with SENTIDO DE RESOLUCION == 'ABSOLUTORIA'
pcs_da_february2024 = pcs_da_february2024[pcs_da_february2024[' SENTIDO DE RESOLUCION'] != 'ABSOLUTORIA']


#keep last 4 characters of file number
pcs_da_february2024['file_year'] = pcs_da_february2024[' NUMERO DE EXPEDIENTE'].str[-4:]

#file number
pcs_da_february2024['file_number'] = pcs_da_february2024[' NUMERO DE EXPEDIENTE'].copy()

#supplier name
pcs_da_february2024['supplier_name'] = pcs_da_february2024['PROVEEDOR O CONTRATISTA'].copy()

#supplier name clean
pcs_da_february2024['supplier_name_clean'] = utils.clean_names(pcs_da_february2024, 'supplier_name')

#source
pcs_da_february2024['source'] = 'da_february2024'

#resolution
pcs_da_february2024['resolution'] = pcs_da_february2024[' SENTIDO DE RESOLUCION'].copy()
# sanction time
pcs_da_february2024['sanction_time'] = pcs_da_february2024[' PLAZO'].copy()


pcs_da_february2024 = pcs_da_february2024[['file_number', 'supplier_name', 'supplier_name_clean', 'source', 'sanction_time', 'resolution']]


##############################
#sanctions W February2024
#What interests me is:
# file number, 
# supplier name
# and supplier name clean
##############################

# file number
pcs_w_february2024['file_number'] = pcs_w_february2024['Expediente'].copy()

#supplier name
pcs_w_february2024['supplier_name'] = pcs_w_february2024['Proveedor y Contratista'].copy()

#supplier name clean
pcs_w_february2024['supplier_name_clean'] = utils.clean_names(pcs_w_february2024, 'supplier_name')

#source
pcs_w_february2024['source'] = 'w_february2024'

#sanction tume
pcs_w_february2024['sanction_time'] = pcs_w_february2024['Periodo de Inhabilitación'].copy()

pcs_w_february2024 = pcs_w_february2024[['file_number', 'supplier_name', 'supplier_name_clean', 'source', 'sanction_time']]



##############################
#sanctions falcon et al March2021
#What interests me is:
# year of notification,
# file number, 
# supplier name
# and supplier name clean
##############################

#remove from column names and column values '\t'
pcs_f_march2021.columns = pcs_f_march2021.columns.str.replace('\t', '')
pcs_f_march2021.replace('\t', '', regex=True, inplace=True)

#remove rows with SENTIDO DE RESOLUCION == 'ABSOLUTORIA'
pcs_f_march2021 = pcs_f_march2021[pcs_f_march2021[' SENTIDO DE RESOLUCION'] != 'ABSOLUTORIA']

#file number
pcs_f_march2021['file_number'] = pcs_f_march2021[' NUMERO DE EXPEDIENTE'].copy()

#supplier name
pcs_f_march2021['supplier_name'] = pcs_f_march2021['PROVEEDOR O CONTRATISTA'].copy()

#supplier name clean
pcs_f_march2021['supplier_name_clean'] = utils.clean_names(pcs_f_march2021, 'supplier_name')

#source
pcs_f_march2021['source'] = 'f_march2021'

#resolution
pcs_f_march2021['resolution'] = pcs_f_march2021[' SENTIDO DE RESOLUCION'].copy()

#sanction time
pcs_f_march2021['sanction_time'] = pcs_f_march2021[' PLAZO'].copy()

pcs_f_march2021 = pcs_f_march2021[['file_number', 'supplier_name', 'supplier_name_clean', 'source', 'sanction_time', 'resolution']]


##############################
#Merge all the datasets
#f march should be the last one since it has missing values in year of notification
pcs_merged = pd.concat([pcs_da_april2023, pcs_da_february2024, pcs_w_february2024, pcs_f_march2021], axis=0)
pcs_merged = pcs_merged.drop_duplicates(subset=['file_number', 'supplier_name_clean'], keep = 'first')


#save
pcs_merged.to_csv(processed_data / 'pcs_merged.csv', index=False)