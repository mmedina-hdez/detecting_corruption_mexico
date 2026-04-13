#EFOS

#This script has the objective of merging all the efos files into one, and then clean it.
#year of the sanction should be from notification, since it is later than the file number

import pandas as pd
import numpy as np
from pyprojroot import here
import utils
sanctions_data = here('data/sanctions_data')
processed_data = here('data/processed_data')

#efos January2024
efos_jan24 = pd.read_csv(sanctions_data / 'Listado_Completo_69-B_31jan2024update.csv', encoding='latin1')

##############################
#efos January2024
#What interests me is:
# RFC
# supplier name (Nombre del Contribuyente)
# supplier name clean
# supplier situation (Situación del contribuyente)
# presumed publication date (Publicación página SAT presuntos)
# sanction_year
##############################

#drop rows with situacion de contribuyente equal to 'desvirtuados'
efos_jan24 = efos_jan24[efos_jan24['Situación del contribuyente'] != 'Desvirtuado']
#get supplier name
efos_jan24['supplier_name'] = efos_jan24['Nombre del Contribuyente'].copy()
#get supplier name clean
efos_jan24['supplier_name_clean'] = utils.clean_names(efos_jan24, 'supplier_name')
#get supplier situation
efos_jan24['supplier_situation'] = efos_jan24['Situación del contribuyente'].copy()
#get presumed publication date
efos_jan24['presumed_publication_date'] = efos_jan24['Publicación página SAT presuntos'].copy()
efos_jan24['presumed_publication_date'] = pd.to_datetime(efos_jan24['presumed_publication_date'], format='%d/%m/%Y')
#sanction year
efos_jan24['sanction_year'] = efos_jan24['presumed_publication_date'].dt.year
#keep columns
efos_jan24 = efos_jan24[['RFC', 'supplier_name', 'supplier_name_clean', 'supplier_situation', 'presumed_publication_date', 'sanction_year']]

#save
efos_jan24.to_csv(processed_data / 'efos_jan24.csv', index=False)



