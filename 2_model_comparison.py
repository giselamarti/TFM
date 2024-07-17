"""
Python script to perform cross validation across different ML models
and subsets of the dataset. 
- Uses: normalized .csv files
- Generates: Different plots in .pdf format
"""

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cross Validation tools
from sklearn.model_selection import ShuffleSplit, cross_val_score

# Models
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

n_rand = 95

# Load data
file_name = "normalized_data_tops.csv" # chose any of the normalized csv files
data = pd.read_csv(f"./data_files/{file_name}")

# Divide dataset into subsets
set0 = data.iloc[:, :-1]                                        # ALL DATA (1+2+3)
set1 = data.iloc[:,12:19]                                       # SET-1 ATOMISTIC FEATURES
set2 = pd.concat([data.iloc[:,0:4], data.iloc[:,5:6],
                  data.iloc[:,11:12]], axis=1)                  # SET-2 STRUCTURAL FEATURES
set3 = pd.concat([data.iloc[:,4:5], data.iloc[:,6:11]], axis=1) # SET-3 EELCTRONIC FEATURES     
set4 = pd.concat([set1, set2], axis=1)                          # SET-4 = 1+2
set5 = pd.concat([set1, set3], axis=1)                          # SET-5 = 1+3
set6 = pd.concat([set2, set3], axis=1)                          # SET-6 = 2+3

subsets = [set0, set1, set2, set3, set4, set5, set6]
subset_names = ['set0', 'set1', 'set2', 'set3', 'set4', 'set5', 'set6']

# Naming mmodels
LR = LinearRegression()
DTR = DecisionTreeRegressor(random_state=n_rand)
RFR = RandomForestRegressor(random_state=n_rand)
KR = KernelRidge()
SVR = SVR()
GBR = GradientBoostingRegressor(random_state=n_rand)
RID = Ridge()
KNN = KNeighborsRegressor()
BR = BayesianRidge()
ABR = AdaBoostRegressor(random_state=n_rand)

allmodels=[LR,DTR,RFR,KR,SVR,GBR,RID,KNN,BR,ABR]
model_names = ['LR', 'DTR', 'RFR', 'KR', 'SVR', 'GBR', 'RID', 'KNN', 'BR', 'ABR']

# Cross validation
subset_maes = {name: [] for name in subset_names}
mae_stds = {name: [] for name in subset_names}
predictions_dict = {name: [] for name in subset_names}
train_y = data['E']

for subset_name, subset in zip(subset_names, subsets):
    for model in allmodels:
        train_x = subset
        
        # Shuffle Split CV
        cv = ShuffleSplit(n_splits=30, train_size = 0.8, test_size=0.2, random_state=n_rand)        
        score = cross_val_score(model, train_x, train_y, scoring='neg_mean_absolute_error',cv=cv)

        subset_maes[subset_name].append(-score.mean())
        mae_stds[subset_name].append(score.std())
        
        model.fit(train_x, train_y)
        predictions = model.predict(train_x).flatten()        
        predictions_dict[subset_name].append(predictions)

######## PLOTS ########
# The plots are saved in pdf format. Rename as needed each of the plots.

### MAE values for each subset and model
colors = plt.cm.viridis(np.linspace(0, 1, len(subset_names)))        

fig, ax = plt.subplots(figsize=(7, 5))
n_models = len(allmodels)
bar_width = 0.1
ind = np.arange(n_models)  

for i, (subset_name, color) in enumerate(zip(subset_names, colors)):
    maes = [subset_maes[subset_name][j] for j in range(len(allmodels))]
    positions = ind + i * bar_width
    ax.bar(positions, maes, bar_width, label=subset_name, color=color)

ax.set_xlabel('Model')
ax.set_ylabel('MAE (KJ/mol)')
ax.set_title('MAE values for each subset and model')
ax.set_xticks(ind + bar_width * (len(subset_names) - 1) / 2)
ax.set_xticklabels(model_names, rotation=45)
ax.legend(title='Subset', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('./plots/subset_maes_tops.pdf', dpi=300, bbox_inches='tight') # Save plot

### MAE values for a specific subset
subset_name = subset_names[0]
x = np.arange(len(allmodels))
maes = subset_maes[subset_name]
stds = mae_stds[subset_name]

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x, maes, yerr=stds, capsize=4, alpha=0.5, color='blue', edgecolor='black')
ax.set_ylabel('MAE (kJ/mol)')
ax.set_ylim([0, 140])
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha="right")
ax.set_title("MAE values for the entire dataset")
plt.grid(linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('./plots/set0_maes_tops.pdf', dpi=300, bbox_inches='tight') # Save plot

for model, mae, stds in zip(allmodels, subset_maes['set0'], mae_stds['set0']):
    print(f"{model}: {mae:.2f} ± {stds:.2f}")
