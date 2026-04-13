#This script has the purpose to merge the published files datasets
#libraries
import pandas as pd
import numpy as np
from pyprojroot import here

minimal_procedures_data = here('data/minimal_procedures_data')
processed_data = here('data/processed_data')

#import all files
cfiles_2010 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2010.csv', encoding='latin-1')
cfiles_2011 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2011.csv', encoding='latin-1')
cfiles_2012 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2012.csv', encoding='latin-1')
cfiles_2013 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2013.csv', encoding='latin-1')
cfiles_2014 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2014.csv', encoding='latin-1')
cfiles_2015 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2015.csv', encoding='latin-1')
cfiles_2016 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2016.csv', encoding='latin-1')
cfiles_2017 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2017.csv', encoding='latin-1')
cfiles_2018 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2018.csv', encoding='latin-1')
cfiles_2019 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2019.csv', encoding='latin-1')
cfiles_2020 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2020.csv', encoding='latin-1')
cfiles_2021 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2021.csv', encoding='latin-1')
cfiles_2022 = pd.read_csv(minimal_procedures_data / 'ExpedientesPublicados2022.csv', encoding='latin-1')
cfiles_2023 = pd.read_csv(minimal_procedures_data / 'Expedientes_PICompraNet2023.csv', encoding='latin-1')

#import name equivalence file
name_equivalence = pd.read_excel(minimal_procedures_data / 'name_equivalence_expedientes_publicados.xlsx')
#delete spaces before and after every value in the dataset
name_equivalence = name_equivalence.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


dict_name_equivalence_2010_2017 = dict(zip(name_equivalence['ep2017'], name_equivalence['final_name']))
#rename columns of cfiles_2010 with dict_name_equivalence_2010_2017
cfiles_2010.rename(columns=dict_name_equivalence_2010_2017, inplace=True)
cfiles_2011.rename(columns=dict_name_equivalence_2010_2017, inplace=True)
cfiles_2012.rename(columns=dict_name_equivalence_2010_2017, inplace=True)
cfiles_2013.rename(columns=dict_name_equivalence_2010_2017, inplace=True)
cfiles_2014.rename(columns=dict_name_equivalence_2010_2017, inplace=True)
cfiles_2015.rename(columns=dict_name_equivalence_2010_2017, inplace=True)
cfiles_2016.rename(columns=dict_name_equivalence_2010_2017, inplace=True)
cfiles_2017.rename(columns=dict_name_equivalence_2010_2017, inplace=True)
#rename columns of cfiles with dict_name_equivalence for each year
dict_name_equivalence_2018 = dict(zip(name_equivalence['ep2018'], name_equivalence['final_name']))
cfiles_2018.rename(columns=dict_name_equivalence_2018, inplace=True)
dict_name_equivalence_2019 = dict(zip(name_equivalence['ep2019'], name_equivalence['final_name']))
cfiles_2019.rename(columns=dict_name_equivalence_2019, inplace=True)
dict_name_equivalence_2020 = dict(zip(name_equivalence['ep2020'], name_equivalence['final_name']))
cfiles_2020.rename(columns=dict_name_equivalence_2020, inplace=True)
dict_name_equivalence_2021 = dict(zip(name_equivalence['ep2021'], name_equivalence['final_name']))
cfiles_2021.rename(columns=dict_name_equivalence_2021, inplace=True)
dict_name_equivalence_2022 = dict(zip(name_equivalence['ep2022'], name_equivalence['final_name']))
cfiles_2022.rename(columns=dict_name_equivalence_2022, inplace=True)
dict_name_equivalence_2023 = dict(zip(name_equivalence['ep2023'], name_equivalence['final_name']))
cfiles_2023.rename(columns=dict_name_equivalence_2023, inplace=True)

#unify the datasets
published_files = pd.concat([cfiles_2010, cfiles_2011, cfiles_2012 , cfiles_2013, cfiles_2014 , cfiles_2015 , cfiles_2016, cfiles_2017, cfiles_2018,  cfiles_2019, cfiles_2020, cfiles_2021, cfiles_2022, cfiles_2023])

#####################
#Procedure type
#Most of the datasets have no procedure_type, therefore I will create a new column based on procedure_template
#Year 2023 is the only one that has procedure_type originally
#####################

#make lower case all values of procedure_type_2023
published_files['procedure_type_2023'] = published_files['procedure_type_2023'].str.lower()
#make lower case all values of procedure_template
published_files['procedure_template'] = published_files['procedure_template'].str.lower()

#Add the procedure type of 2023 to procedure_template
published_files['procedure_template'] = np.where(published_files['procedure_template'].isnull(), published_files['procedure_type_2023'], published_files['procedure_template'])

#Drop procedure_type_2023
published_files = published_files.drop(columns=['procedure_type_2023'])

#Create a dataframe that allows me to recodify the procedure template
new_codification_procedure_type = pd.DataFrame((published_files['procedure_template'].value_counts()).sort_index()).reset_index()
new_codification_procedure_type.columns = ['procedure_template', 'count']

#Recodify the procedure template according to the following
# if column procedure_template has the string 'Licitación', then recode it as 'open
new_codification_procedure_type['new_procedure_type'] = np.where(new_codification_procedure_type['procedure_template'].str.contains('licitación'), 'open', new_codification_procedure_type['procedure_template'])
# if column procedure_template  has the string 'Invitación', then recode it as 'at_least_three'
new_codification_procedure_type['new_procedure_type'] = np.where(new_codification_procedure_type['procedure_template'].str.contains('invitación'), 'at_least_three', new_codification_procedure_type['new_procedure_type'])
# if column procedure_template  has the string 'Directa', then recode it as 'direct'
new_codification_procedure_type['new_procedure_type'] = np.where(new_codification_procedure_type['procedure_template'].str.contains('directa'), 'direct', new_codification_procedure_type['new_procedure_type'])
#if column procedure_template  has the string 'Convocatoria', then recode it as 'project_call'
new_codification_procedure_type['new_procedure_type'] = np.where(new_codification_procedure_type['procedure_template'].str.contains('convocatoria'), 'other', new_codification_procedure_type['new_procedure_type'])
#if procedure_template  has a 'entes', then recode it as 'other'
new_codification_procedure_type['new_procedure_type'] = np.where(new_codification_procedure_type['procedure_template'].str.contains('entes'), 'other', new_codification_procedure_type['new_procedure_type'])
#if new_procedure_type has a number, then recode it as 'other'
new_codification_procedure_type['new_procedure_type'] = np.where(new_codification_procedure_type['new_procedure_type'].str.contains('[0-9]'), 'other', new_codification_procedure_type['new_procedure_type'])


#Create the dictionary
new_codification_procedure_type_dict = dict(zip(new_codification_procedure_type['procedure_template'], new_codification_procedure_type['new_procedure_type']))

#replace 'procedure_type' column in published_files with the new codification
published_files['procedure_type'] = published_files['procedure_template'].map(new_codification_procedure_type_dict)

print(published_files['procedure_type'].value_counts(dropna = False) / len(published_files))

#####################
# Submission deadline or award date
# 'Vigencia de anuncio' or 'submission_deadline_or_award_date' is a mixed date column that has the submission deadline or the award date
#According to the dictionary before 2024 the definition is:
# 'Plazo de participación o vigencia del anuncio 
# (Para proyecto de convocatoria FECHA LÍMITE PARA RECIBIR COMENTARIOS AL PROYECTO, 
# Para licitación pública e invitación a cuando menos tres personas: FECHA Y HORA DE APERTURA DE PROPOSICIONES 
# y para adjudicación directa FECHA DE LA NOTIFICACIÓN DE LA ADJUDICACIÓN)
# Therefore, I will create two columns:
# 1. submission_deadline_date, for observations that are open, at least three, and other
# 2. award_date, for observations that are direct
#####################

#Create a new column that has the submission deadline date
published_files['submission_deadline_date'] = np.where(published_files['procedure_type'].isin(['open', 'at_least_three', 'other']), published_files['submission_deadline_or_award_date'], np.nan)
#Create a new column that has the award date
published_files['award_date'] = np.where(published_files['procedure_type'].isin(['direct']), published_files['submission_deadline_or_award_date'], np.nan)

#####################
#Make the date columns into datetime
#####################
#advertisement_date
#keep first 10 characters of advertisement_date
published_files['advertisement_date'] = published_files['advertisement_date'].str[:10]
#replace the '/' with '-'
published_files['advertisement_date'] = published_files['advertisement_date'].str.replace('/', '-')
#some dates have Y-m-d and some have dmY, i will mask them
# mask is for the dates with the '%d-%m-%Y' format
mask = published_files['advertisement_date'].str.match(r'\d{4}-\d{2}-\d{2}')
mask = mask.fillna(False)
published_files.loc[mask, 'advertisement_date'] = pd.to_datetime(published_files.loc[mask, 'advertisement_date'], errors='raise', format='%Y-%m-%d')
# Then, convert the remaining dates with the '%Y/%m/%d' format
mask = ~mask
published_files.loc[mask, 'advertisement_date'] = pd.to_datetime(published_files.loc[mask, 'advertisement_date'], errors='raise', format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
#again to datetime
published_files['advertisement_date'] = pd.to_datetime(published_files['advertisement_date'], errors='raise')


#submission_deadline_date
#keep first 10 characters of submission_deadline_date
published_files['submission_deadline_date'] = published_files['submission_deadline_date'].str[:10]
#replace the '/' with '-'
published_files['submission_deadline_date'] = published_files['submission_deadline_date'].str.replace('/', '-')
#some dates have Y-m-d and some have dmY, i will mask them
# mask is for the dates with the '%d-%m-%Y' format
mask = published_files['submission_deadline_date'].str.match(r'\d{4}-\d{2}-\d{2}')
mask = mask.fillna(False)
published_files.loc[mask, 'submission_deadline_date'] = pd.to_datetime(published_files.loc[mask, 'submission_deadline_date'], errors='coerce', format='%Y-%m-%d') #it has 4 errors of formating, that's why im using coerce
# Then, convert the remaining dates with the '%Y/%m/%d' format
mask = ~mask
published_files.loc[mask, 'submission_deadline_date'] = pd.to_datetime(published_files.loc[mask, 'submission_deadline_date'], errors='coerce', format='%d-%m-%Y').dt.strftime('%Y-%m-%d') #it has 4 errors of formating, that's why im using coerce
#again to datetime
published_files['submission_deadline_date'] = pd.to_datetime(published_files['submission_deadline_date'], errors='raise')

#award_date
#keep first 10 characters of award_date
published_files['award_date'] = published_files['award_date'].str[:10]
#replace the '/' with '-'
published_files['award_date'] = published_files['award_date'].str.replace('/', '-')
#some dates have Y-m-d and some have dmY, i will mask them
# mask is for the dates with the '%d-%m-%Y' format
mask = published_files['award_date'].str.match(r'\d{4}-\d{2}-\d{2}')
mask = mask.fillna(False)
published_files.loc[mask, 'award_date'] = pd.to_datetime(published_files.loc[mask, 'award_date'], errors='coerce', format='%Y-%m-%d') #it has 1 errors of formating, that's why im using coerce
# Then, convert the remaining dates with the '%Y/%m/%d' format
mask = ~mask
published_files.loc[mask, 'award_date'] = pd.to_datetime(published_files.loc[mask, 'award_date'], errors='coerce', format='%d-%m-%Y').dt.strftime('%Y-%m-%d') #it has 1 errors of formating, that's why im using coerce
#again to datetime
published_files['award_date'] = pd.to_datetime(published_files['award_date'], errors='raise')

#####################
#save dataframe as csv
published_files.to_csv(processed_data / 'minimal_procedures_merged.csv', index=False)

