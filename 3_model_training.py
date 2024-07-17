"""
Python script to perform different ML model. 
It prompts the user to choose a file to be used (MS, LS, tops or sites).
It also promts the used to choose a model among Random Forest Regressor (RFR),
Gradient Boosting Regressor (GBR), or Kernel Ridge Regressor (KRR).

Perfroms: a recursive feature elimination, hyperparameter optimization,
learning curve plotting and a regression plotting.

- Uses: normalized .csv files (normalized_data_MS.csv, ...)
- Generates: learning curve plots and prediction regression plots in .pdf format
"""

# Libraries
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.model_selection import ShuffleSplit, train_test_split, learning_curve
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor

n_rand = 95

# Read data file
print("Choose csv file to be used")
print("  MS = Most stable sites")
print("  LS = Least stable sites")
print("  tops = top sites")
print("  sites = All sites")
file_input = input("Enter 'MS', 'LS', 'tops', 'sites': ")
if file_input == "MS":
    data = pd.read_csv("./data_files/normalized_data_MS.csv")
elif file_input == "LS":
    data = pd.read_csv("./data_files/normalized_data_LS.csv")
elif file_input == "tops":
    data = pd.read_csv("./data_files/normalized_data_tops.csv")
elif file_input == "sites":
    data = pd.read_csv("./data_files/normalized_data_sites.csv")

# Choose model
print("")
print("Choose model to be used")
print("  RFR = Random Forest Regressor")
print("  GBR = Gradient Boosting Regressor")
print("  KRR = Kernel Ridge Regressor")
model_input = input("Enter 'RFR', 'GBR', or 'KRR': ")
print("")
if model_input == "RFR":
    print("Running code for RFR...")
elif model_input == "GBR":
    print("Running code for GBR...")
elif model_input == "KRR":
    print("Running code for KRR...")
print("")

# RECURSIVE FEATURE ELIMINATION
X = data.iloc[:, :-1]
y = data['E']
cv = ShuffleSplit(n_splits=30, train_size = 0.8, test_size=0.2, random_state=n_rand)

if model_input == "RFR":
    estimator = RandomForestRegressor(random_state=n_rand)
elif model_input == "GBR":
    estimator = GradientBoostingRegressor(random_state=n_rand)
elif model_input == "KRR":
    estimator = KernelRidge(alpha=0.1, kernel="laplacian")

if model_input == "RFR" or model_input == "GBR":
    # RFE is only performed for RFR and GBR, not for KRR
    print("Performing recursive feature elimination...")

    numf = [10,9,8,7,6,5,4,3,2,1]
    best_mae = float('inf')
    best_features = None

    for i in numf:
        selector = RFE(estimator, n_features_to_select=i, step=10, verbose=0, importance_getter='auto')
        selector.fit(X, y)
        selector_support = selector.get_support()
        selected_features = X.loc[:, selector_support].columns.tolist()
        print(selected_features)
        score = cross_val_score(estimator, X[selected_features], y, scoring='neg_mean_absolute_error',cv=cv, n_jobs=None)
        mae = -score.mean()
        
        if mae < best_mae:
            best_mae = mae
            best_features = selected_features
        
        print('num of features: ', i)
        print("MAE: {:.5f}".format(mae), u"\u00B1", "{:.5f}".format(score.std()))
        print("-------------------------------------------")

    #predictors = best_features
    predictors = ['Valence el.', 'Ed', 'CN ads']
    print("")
    print("Best predictors: ", predictors)
    X = DataFrame(data[predictors])

    # HYPERPARAMETER OPTIMIZATION
    print("")
    print("Performing hyperparameter optimization...")

    if model_input == "RFR":
        n_estimators = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        max_features = [None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9]
        min_samples_leaf = [1, 2 ,3, 4, 5, 6, 7, 8, 9]
        param_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}
        
    elif model_input == "GBR":

        n_estimators = [20, 30, 40, 50, 60, 70, 80, 90, 100 ]
        max_features = [None, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
        max_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        min_samples_split = [2, 4, 6, 8, 10, 12, 14, 16]
        min_samples_leaf = [1, 2, 3, 4, 5]

        param_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}

    grid_search = GridSearchCV(estimator, param_grid, cv = cv,
                            scoring = "neg_mean_absolute_error", return_train_score = True, n_jobs = -1)

    grid_search.fit(X, y)
    estimator = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    print("")
    print("Model to be used:", estimator)


# LEARNING CURVE
print("")
print("Plotting the learning curve...")

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=estimator,
    X=X, 
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 50),
    cv=cv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

train_mean = -train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
validation_mean = -validation_scores.mean(axis=1)
validation_std = validation_scores.std(axis=1)

plt.figure(figsize=(7, 4))
plt.plot(train_sizes, train_mean, color="b", label="Train MAE")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="b")
plt.plot(train_sizes, validation_mean, color="darkorange", label="Test MAE")
plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, alpha=0.2, color="darkorange")

plt.xlabel("Training Set Size")
plt.ylabel("MAE (kJ/mol)")
plt.ylim([0,120])
plt.legend(loc="best")
plt.grid(linestyle='--', linewidth=0.3)

valid_file_inputs = ["MS", "LS", "tops", "sites"]
valid_model_inputs = ["RFR", "GBR", "KRR"]

if file_input in valid_file_inputs and model_input in valid_model_inputs:
    plt.savefig(f'./plots/learning_curve_{model_input}_{file_input}.pdf', dpi=300, bbox_inches='tight')

print("Plot saved.")
print("")
print(f"Final test MAE: {validation_mean[-1]:.2f} ± {validation_std[-1]:.2f}")
print(f"Final train MAE: {train_mean[-1]:.2f} ± {train_std[-1]:.2f}")
print("")
print("---------------------------------------------")

if model_input == "RFR" or model_input == "GBR":
    # Feature importances are also printed in order to undertand which are the 
    # most relevant features for our model.
    results = cross_validate(estimator, X, y, scoring = "neg_mean_absolute_error", cv = cv, return_estimator = True)
    importances = []
    for model in results['estimator']:
        importances.append(model.feature_importances_)
    mean = sum(importances) / len(importances)
    print("")
    print("Mean Feature Importances:")
    print("")
    for i,v in enumerate(mean):
        print('%s: %.3f' % (predictors[i],v))


# REGRESSIONS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=n_rand)
estimator.fit(X_train, y_train)
train_predictions = estimator.predict(X_train).flatten()
test_predictions = estimator.predict(X_test).flatten()

# Correlation R
train_r = np.corrcoef(y_train, train_predictions)[0, 1]
test_r = np.corrcoef(y_test, test_predictions)[0, 1]

# Plot
fig, ax = plt.subplots(figsize=(3,3))
ax.plot(np.linspace(-1000, -100, 100), np.linspace(-1000, -100, 100), color="k", linestyle="--", alpha=0.7)
ax.scatter(y_train, train_predictions, alpha=0.8, c="blue", edgecolor="k", s=100, label="Training Set")
ax.scatter(y_test, test_predictions, alpha=0.8, c="darkorange", edgecolor="k", s=100, label="Test Set")
ax.set_xlabel('Calculated $E$ values')
ax.set_ylabel('Predicted $E$ values')
ax.set_xlim([-1000, -100])
ax.set_ylim([-1000, -100])
ax.set_xticks(np.arange(-1000, -100 + 1, 200))
ax.set_yticks(np.arange(-1000, -100 + 1, 200))
ax.grid(linestyle='--', linewidth=0.3)
ax.text(-950, -170, f'Train R = {train_r:.3f}', fontsize=12, color='blue')
ax.text(-950, -230, f'Test R = {test_r:.3f}', fontsize=12, color='darkorange')
plt.show()

if file_input in valid_file_inputs and model_input in valid_model_inputs:
    plt.savefig(f'./plots/regression_{model_input}_{file_input}.pdf', dpi=300, bbox_inches='tight')
