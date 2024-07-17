"""
This script for data preprocessing, in order to obtain the necessary prepared and normalized data 
for different models. Each model (section) generates a .csv file.

The base data file needed as input is "data.csv", where all the information is contained

Sections:
    1. MOST STABLE Eads/abs FOR 81 SURFACES
    2. LEAST STABLE Eads/abs FOR 81 SURFACES
    3. Eads/abs FOR ALL 390 SURFACES
    4. TOP SITES Eads/abs FOR 130 SURFACES
"""

# Import libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

n_rand = 95

# Import and prepare 'ElementData' database
from Elementdata import ElementData
elemdatabase = ElementData()

element_symbols = ["Zr","Zn","Y","Ti","Tc","Sc","Ru","Re","Os","Hf","Co","Cd","Rh",\
                  "Pt","Pd","Ni","Ir","Cu","Au","Ag","W","V","Ta","Nb","Mo","Fe","Cr"]

atomic_numbers = [elemdatabase.elementnr[symbol] for symbol in element_symbols]
valence_e = [elemdatabase.valenceelectrons[symbol] for symbol in element_symbols]
element_period = [elemdatabase.elementperiod[symbol] for symbol in element_symbols]
element_weight = [elemdatabase.elementweight[symbol] for symbol in element_symbols]
covalent_radius = [elemdatabase.CovalentRadius[symbol] for symbol in element_symbols]
ENPauling = [elemdatabase.ElectroNegativityPauling[symbol] for symbol in element_symbols]
ENAllred = [elemdatabase.ElectroNegativityAllredRochow[symbol] for symbol in element_symbols]

elementdata = pd.DataFrame({
    "AS": element_symbols,
    "Atomic Number": atomic_numbers,
    "Valence Electrons": valence_e,
    "Element Period": element_period,
    "Element Weight": element_weight,
    "Covalent Radius": covalent_radius,
    "EN Pauling": ENPauling,
    "EN Allred": ENAllred
})

# Import data
data_csv = pd.read_csv("sites.csv", sep=';', dtype={'MI': str})
data = pd.merge(data_csv, elementdata, on='AS', how='left')

# Replace labels in the 'MI' column
data['MI'] = data['MI'].replace({'0001': '001', '1010': '011', '1120': '111'})

# Choose model
print("Choose csv file to generate:")
print("  MS - Most stable Eads/abs for 81 surfaces")
print("  LS - Least stable Eads/abs for 81 surfaces")
print("  ALL - Eads/abs for all 390 surfaces")
print("  TOPS - only top sites' Eads/abs (130 surfaces)")
print("")
model_input = input("Enter: 'MS', 'LS', 'ALL', 'TOPS':")

################################################
# 1. MOST STABLE Eads/abs FOR 81 SURFACES
# - Generates: 'normalized_data_MS.csv'
################################################

if model_input == "MS":

    # Create a boolean mask to identify the rows to keep
    mask = data.groupby(['AS', 'MI'])['E'].transform(max) == data['E']
    filtered_data = data[mask]

    # Move the column "E" to the end
    columns = [col for col in filtered_data.columns if col != 'E'] + ['E']
    filtered_data = filtered_data[columns]
    data = filtered_data

    # Remove these columns
    columns = ["AS", "MI", "site"]
    data.drop(columns=columns, axis=1, inplace=True)

    # Energy values must be negative
    data['E'] = data['E'] * -1

    # Enconding categorical features as vectors
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    data = ct.fit_transform(data) # Enconding categorical features as vectors.
    data = DataFrame(data)
    data.columns = ["bcc", "fcc", "hcp", "Interatomic dist","nd", "CN sur", "Ed", "Edw", "Eu", "SE", "WF", "CN ads", \
                    "Atomic n.", "Valence el.", "Element period", "Element weight", "Covalent Radius", "EN Pauling", "EN Allred", "E"]
    pd.set_option("display.max_columns", None)

    # Normalize all parameters except E
    features_to_normalize = data.columns.difference(['E'])  # Exclude target variable column
    normalization_params = {}

    for feature in features_to_normalize:
        min_value = data[feature].min()
        max_value = data[feature].max()
        normalization_params[feature] = {'min': min_value, 'max': max_value}
        data[feature] = (data[feature] - min_value) / (max_value - min_value)
        
    # Save to csv file
    data.to_csv('./data_files/normalized_data_MS.csv', index=False)


################################################
# 2. LEAST STABLE Eads/abs FOR 81 SURFACES
# - Generates: 'normalized_data_LS.csv'
################################################

if model_input == "LS":

    # Create a boolean mask to identify the rows to keep
    mask = data.groupby(['AS', 'MI'])['E'].transform(min) == data['E']
    filtered_data = data[mask]

    # Move the column "E" to the end
    columns = [col for col in filtered_data.columns if col != 'E'] + ['E']
    filtered_data = filtered_data[columns]
    data = filtered_data

    # Remove these columns
    columns = ["AS", "MI", "site"]
    data.drop(columns=columns, axis=1, inplace=True)

    # Energy values must be negative
    data['E'] = data['E'] * -1

    # Enconding categorical features as vectors
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    data = ct.fit_transform(data) # Enconding categorical features as vectors.
    data = DataFrame(data)
    data.columns = ["bcc", "fcc", "hcp", "Interatomic dist","nd", "CN sur", "Ed", "Edw", "Eu", "SE", "WF", "CN ads", \
                    "Atomic n.", "Valence el.", "Element period", "Element weight", "Covalent Radius", "EN Pauling", "EN Allred", "E"]
    pd.set_option("display.max_columns", None)

    # Normalize all parameters except E
    features_to_normalize = data.columns.difference(['E'])  # Exclude target variable column
    normalization_params = {}

    for feature in features_to_normalize:
        min_value = data[feature].min()
        max_value = data[feature].max()
        normalization_params[feature] = {'min': min_value, 'max': max_value}
        data[feature] = (data[feature] - min_value) / (max_value - min_value)
        
    # Save to csv file
    data.to_csv('./data_files/normalized_data_LS.csv', index=False)

################################################
# 3. Eads/abs FOR ALL 390 SURFACES
# - Generates: 'normalized_data_sites.csv'
################################################

if model_input == "ALL":

    # Remove these columns
    columns = ["AS", "MI", "site"]
    data.drop(columns=columns, axis=1, inplace=True)

    # Energy values must be negative
    data['E'] = data['E'] * -1

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    data = ct.fit_transform(data)
    data = DataFrame(data)
    data.columns = ["bcc", "fcc", "hcp", "Interatomic dist", "nd", "cn sur", "Ed", "Edw", "Eu", "SE", "WF", "CN", \
                    "E", "atomic n", "valence e", "element period", "element weight", "covalent r", "EN Pauling", "EN Allred"]
    pd.set_option("display.max_columns", None)

    # Move the column "E" to the end
    columns = [col for col in data.columns if col != 'E'] + ['E']
    data = data[columns]

    # Normalization
    features_to_normalize = data.columns.difference(['E'])  # Exclude target variable column
    normalization_params = {}
    for feature in features_to_normalize:
        min_value = data[feature].min()
        max_value = data[feature].max()
        normalization_params[feature] = {'min': min_value, 'max': max_value}
        data[feature] = (data[feature] - min_value) / (max_value - min_value)

    # Save to csv file
    data.to_csv('./data_files/normalized_data_sites.csv', index=False)


################################################
# 4. TOP SITES Eads/abs FOR 130 SURFACES
# - Generates: 'normalized_data_tops.csv'
################################################

if model_input == "TOPS":

    data_tops = data[data['site'].isin(['ads t', 'ads top', 'abs t', 'abs top'])]
    columns = [0,3,13]
    data_tops.drop(data_tops.columns[columns],axis=1,inplace=True)
    data_tops['E'] = data_tops['E'] * -1

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    data = ct.fit_transform(data_tops) 
    data = DataFrame(data)
    data.columns = ["bcc", "fcc", "hcp", "Interatomic dist", "nd", "cn sur", "Ed", "Edw", "Eu", "SE", "WF", "CN", \
                    "E", "atomic n", "valence e", "element period", "element weight", "covalent r", "EN Pauling", "EN Allred"]
    pd.set_option("display.max_columns", None)

    # Move the column "E" to the end
    columns = [col for col in data.columns if col != 'E'] + ['E']
    data = data[columns]

    # Normalization
    features_to_normalize = data.columns.difference(['E'])  # Exclude target variable column
    normalization_params = {}

    for feature in features_to_normalize:
        min_value = data[feature].min()
        max_value = data[feature].max()
        normalization_params[feature] = {'min': min_value, 'max': max_value}
        data[feature] = (data[feature] - min_value) / (max_value - min_value)

    # Save to csv file
    data.to_csv('./data_files/normalized_data_tops.csv', index=False)
