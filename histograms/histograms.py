"""
Python script to plot histograms
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

data = pd.read_csv("sites.csv", sep=';')
data['E'] = data['E'] * -1

####### SITE LABEL PLOTS #######

site_counts = data['site'].value_counts()
ads_data = data[data['site'].str.startswith('ads ')]
abs_data = data[data['site'].str.startswith('abs ')]
ads_counts = ads_data['site'].value_counts()
abs_counts = abs_data['site'].value_counts()

# Plot 1
fig = plt.figure(figsize=(12, 7))
gs = GridSpec(2, 2, width_ratios=[3, 1])

ax0 = fig.add_subplot(gs[:, 0])
ax0.bar(site_counts.index, site_counts.values, color='mediumorchid', edgecolor='black')
ax0.set_title('All sites counts')
ax0.set_xlabel('Site')
ax0.set_ylabel('Count')
ax0.tick_params(axis='x', rotation=45)
ax0.grid(linestyle='--', linewidth=0.5)

ax1 = fig.add_subplot(gs[0, 1])
ax1.bar(ads_counts.index, ads_counts.values, color='royalblue', edgecolor='black')
ax1.set_title('Adsorption site counts')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(linestyle='--', linewidth=0.5)

ax2 = fig.add_subplot(gs[1, 1])
ax2.bar(abs_counts.index, abs_counts.values, color='red', edgecolor='black', alpha=0.8)
ax2.set_title('Absorption site counts')
ax2.set_xlabel('Site')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('sites_counts.pdf', dpi=300, bbox_inches='tight')

####### ENERGY PLOTS #######

data_sorted = data.sort_values(by='E')
ads_data_sorted = data_sorted[data_sorted['site'].str.startswith('ads ')]
abs_data_sorted = data_sorted[data_sorted['site'].str.startswith('abs ')]

# Plot 2
fig = plt.figure(figsize=(12, 7))
gs = GridSpec(2, 2, width_ratios=[3, 1])

ax0 = fig.add_subplot(gs[:, 0])
ax0.hist(data_sorted['E'], bins=50, color='mediumorchid', edgecolor='black')
ax0.set_title('All energy values')
ax0.set_xlabel('Energy (kJ/mol)')
ax0.set_ylabel('Count')
ax0.grid(linestyle='--', linewidth=0.5)

ax1 = fig.add_subplot(gs[0, 1])
ax1.hist(ads_data_sorted['E'], bins=25, color='royalblue', edgecolor='black')
ax1.set_title('Adsorption energy values')
ax1.set_ylabel('Count')
ax1.grid(linestyle='--', linewidth=0.5)

ax2 = fig.add_subplot(gs[1, 1])
ax2.hist(abs_data_sorted['E'], bins=25, color='red', edgecolor='black', alpha=0.8)
ax2.set_title('Absorption energy values')
ax2.set_xlabel('Energy (kJ/mol)')
ax2.set_ylabel('Count')
ax2.grid(linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('energy_counts.pdf', dpi=300, bbox_inches='tight')

# ADS energies
ads_data = data[data['site'].str.startswith('ads ')]
ads_top_data = ads_data[ads_data['site'].str.contains('ads top')]
ads_b_data = ads_data[ads_data['site'].str.contains('ads b|ads bS|ads bL')]
ads_h_data = ads_data[ads_data['site'].str.contains('ads h|ads hE|ads h fcc|ads h hcp')]

# ABS energies
abs_data = data[data['site'].str.startswith('abs ')]
abs_top_data = abs_data[abs_data['site'].str.contains('abs top')]
abs_b_data = abs_data[abs_data['site'].str.contains('abs b|abs bS|abs bL')]
abs_h_data = abs_data[abs_data['site'].str.contains('abs h|abs hE|abs h fcc|abs h hcp')]

# Plot 3
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

ax1.hist(ads_top_data['E'], bins=20, color='red', edgecolor='black', alpha=0.7, label='top')
ax1.hist(ads_h_data['E'], bins=20, color='blue', edgecolor='black', alpha=0.7, label='hollow')
ax1.hist(ads_b_data['E'], bins=20, color='yellow', edgecolor='black', alpha=0.7, label='bridge')
ax1.set_title('Adsorption energies sorted by site type')
ax1.set_xlabel('Energy (kJ/mol)')
ax1.set_ylabel('Count')
ax1.legend()

ax2.hist(abs_top_data['E'], bins=20, color='red', edgecolor='black', alpha=0.7, label='top')
ax2.hist(abs_h_data['E'], bins=20, color='blue', edgecolor='black', alpha=0.7, label='hollow')
ax2.hist(abs_b_data['E'], bins=20, color='yellow', edgecolor='black', alpha=0.7, label='bridge')
ax2.set_title('Absorption energies sorted by site type')
ax2.set_xlabel('Energy (kJ/mol)')
ax2.set_ylabel('Count')
ax2.legend()

plt.tight_layout()
plt.savefig('energy_counts_2.pdf', dpi=300, bbox_inches='tight')