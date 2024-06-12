"""
This script generates the necessary .csv files to proceed with the different ML models.
Comment or uncomment the needed sections to generate the files.

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
    "Element Symbol": element_symbols,
    "Atomic Number": atomic_numbers,
    "Valence Electrons": valence_e,
    "Element Period": element_period,
    "Element Weight": element_weight,
    "Covalent Radius": covalent_radius,
    "EN Pauling": ENPauling,
    "EN Allred": ENAllred
})

################################################
# 1. MOST STABLE Eads/abs FOR 81 SURFACES
# - Uses: 'data_proves.csv'
# - Generates: 'normalized_data_MS.csv'
################################################

#"""  -> Comment/remove this line to uncomment the section
# Triplicate each row to match the other dataframe
elementdata_x3 = pd.DataFrame(np.repeat(elementdata.values, 3, axis=0), columns=elementdata.columns)
elementdata_x3.reset_index(drop=True, inplace=True)

data_csv = pd.read_csv("data_proves.csv", sep=';', dtype={'MI': str})
data = pd.concat([elementdata_x3, data_csv], axis=1)       # Merge the two dataframes
columns = [0,8,11,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
data.drop(data.columns[columns],axis=1,inplace=True)       # remove these columns
data['MS Eads/abs'] = data['MS Eads/abs'] * -1             # Eads/abs values must be negative

# Encode the "Crystal structure" (fcc, bcc, hcp) descritpor
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
data = ct.fit_transform(data)
data = DataFrame(data)
data.columns = ["bcc", "fcc", "hcp", "Atomic number", "Valence electrons", "Element Period", "Element Weight", \
                "Covalent Radius", "EN Pauling", "EN Allred", "Interatomic dist", \
                "nd", "cn sur", "Ed", "Edw", "Eu", "SE", "WF", "cnADS", "Eads/abs"]
pd.set_option("display.max_columns", None)

# Normalize all parameters except Eads/abs
features_to_normalize = data.columns.difference(['Eads/abs'])  # Exclude target variable column
normalization_params = {}

for feature in features_to_normalize:
    min_value = data[feature].min()
    max_value = data[feature].max()
    normalization_params[feature] = {'min': min_value, 'max': max_value}
    data[feature] = (data[feature] - min_value) / (max_value - min_value)
    
# Save to csv file
data.to_csv('normalized_data_MS.csv', index=False)

#"""

################################################
# 2. LEAST STABLE Eads/abs FOR 81 SURFACES
# - Uses: 'data_less_stable.csv'
# - Generates: 'normalized_data_LS.csv'
################################################

"""  -> Comment/remove this line to uncomment the section
elementdata_x3 = pd.DataFrame(np.repeat(elementdata.values, 3, axis=0), columns=elementdata.columns)
elementdata_x3.reset_index(drop=True, inplace=True)

data_csv = pd.read_csv("data_less_stable.csv", sep=';', dtype={'MI': str})
data = pd.concat([elementdata_x3, data_csv], axis=1)
columns = [0,8,11,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]
data.drop(data.columns[columns],axis=1,inplace=True)       # remove these columns

# Encode the "Crystal structure" (fcc, bcc, hcp) descriptor
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
data = ct.fit_transform(data)
data = DataFrame(data)
data.columns = ["bcc", "fcc", "hcp", "Atomic number", "Valence electrons", "Element Period", "Element Weight", \
                "Covalent Radius", "EN Pauling", "EN Allred", "Interatomic dist", \
                "nd", "cn sur", "Ed", "Edw", "Eu", "SE", "WF", "cnADS", "Eads/abs"]
pd.set_option("display.max_columns", None)

# Normalization
features_to_normalize = data.columns.difference(['Eads/abs'])
normalization_params = {}

for feature in features_to_normalize:
    min_value = data[feature].min()
    max_value = data[feature].max()
    normalization_params[feature] = {'min': min_value, 'max': max_value}
    data[feature] = (data[feature] - min_value) / (max_value - min_value)

# Save to csv file
data.to_csv('normalized_data_LS.csv', index=False)

"""

################################################
# 3. Eads/abs FOR ALL 390 SURFACES
# - Uses: 'sites.csv'
# - Generates: 'normalized_data_sites.csv'
################################################

"""
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

data_csv = pd.read_csv("sites.csv", sep=';', dtype={'MI': str})
data = pd.merge(data_csv, elementdata, on='AS', how='left')
columns = [0,3,13] 
data.drop(data.columns[columns],axis=1,inplace=True)       # remove these columns
data['E'] = data['E'] * -1 

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
data = ct.fit_transform(data)
data = DataFrame(data)
data.columns = ["bcc", "fcc", "hcp", "Interatomic dist", "nd", "cn sur", "Ed", "Edw", "Eu", "SE", "WF", "CN", \
                "Eads/abs", "atomic n", "valence e", "element period", "element weight", "covalent r", "EN Pauling", "EN Allred"]
pd.set_option("display.max_columns", None)

# Normalization
features_to_normalize = data.columns.difference(['Eads/abs'])  # Exclude target variable column
normalization_params = {}
for feature in features_to_normalize:
    min_value = data[feature].min()
    max_value = data[feature].max()
    normalization_params[feature] = {'min': min_value, 'max': max_value}
    data[feature] = (data[feature] - min_value) / (max_value - min_value)

# Save to csv file
data.to_csv('normalized_data_sites.csv', index=False)

"""

################################################
# 4. TOP SITES Eads/abs FOR 130 SURFACES
# - Uses: 'sites.csv'
# - Generates: 'normalized_data_tops.csv'
################################################

"""

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

data_csv = pd.read_csv("sites.csv", sep=';', dtype={'MI': str})
data = pd.merge(data_csv, elementdata, on='AS', how='left')
data_tops = data[data['site'].isin(['ads t', 'ads top', 'abs t', 'abs top'])]
columns = [0,3,13]
data_tops.drop(data_tops.columns[columns],axis=1,inplace=True)
data_tops['E'] = data_tops['E'] * -1

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
data = ct.fit_transform(data_tops) 
data = DataFrame(data)
data.columns = ["bcc", "fcc", "hcp", "Interatomic dist", "nd", "cn sur", "Ed", "Edw", "Eu", "SE", "WF", "CN", \
                "Eads/abs", "atomic n", "valence e", "element period", "element weight", "covalent r", "EN Pauling", "EN Allred"]
pd.set_option("display.max_columns", None)

# Normalization
features_to_normalize = data.columns.difference(['Eads/abs'])  # Exclude target variable column
normalization_params = {}

for feature in features_to_normalize:
    min_value = data[feature].min()
    max_value = data[feature].max()
    normalization_params[feature] = {'min': min_value, 'max': max_value}
    data[feature] = (data[feature] - min_value) / (max_value - min_value)

# Save to csv file
data.to_csv('normalized_data_tops.csv', index=False)

"""

