# https://www.youtube.com/watch?v=YaKMeAlHgqQ

import pandas as pd
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as skl

# Turn off interactive plotting (uncomment this if you want to show the plots
# as they are created and replace plt.close() with plt.show())
plt.ioff()

# Set up data output
writer1 = pd.ExcelWriter('PCA_correlation_figures/correl_data.xlsx', engine = 'xlsxwriter')
writer2 = pd.ExcelWriter('PCA_correlation_figures/pca_data.xlsx', engine = 'xlsxwriter')

# Set up cmd arguments for the user
parser = argparse.ArgumentParser(description = 'Enter file names')
parser.add_argument('filename', type = str, help = 'Features extracted from algorithm (one sheet)')
parser.add_argument('gtruth', type = str, help = 'Manual ground truth data (one sheet)')
args = parser.parse_args()

# Extract data from csv files
df = pd.read_csv(args.filename, index_col = False)
gt = pd.read_excel(args.gtruth, index_col = False)

# Merge the two data tables based on the images that each file have in common
dt = pd.merge(df, gt, on='Name', how='inner')
dt.to_excel('merged_data_global_thresh_' + date.today().strftime('%m-%d-%Y') + '.xlsx')

# Set up a data frame with only numbers
nums_only_dt = pd.concat([dt.select_dtypes('int64'), dt.select_dtypes('float64')], axis = 1)

# Set the strain to be the index instead of the image name now that we have
# combined the files
strains = dt.Folder.unique()
features = nums_only_dt.columns

# Can get the averages by folder/strain
avgs = dt.groupby(['Folder']).mean()


#####################################################################################
# If a feature has very low variance (the values are not very
# different between images, then we should ignore it).
#####################################################################################
variance = nums_only_dt.var()
dt_high_var = dt.copy()
for my_num, feature in zip(variance, features):
    if my_num < 1:
        dt_high_var = dt_high_var.drop(feature, axis = 1)

#####################################################################################
# Can drop one feature if it is pairwise correlating with another
# feature (reduce redundancy)
#####################################################################################
# Correlation, regarless of strain
corr = dt.corr()
# Show figure
plt.matshow(corr, fignum = 10)
#
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.rcParams.update({'font.size':10})
plt.gcf().set_size_inches(14.5, 14.5)
plt.title('Correlation of All Strains', pad = 60)
plt.savefig('PCA_correlation_figures/corr.png')
plt.close()

sig = (abs(corr.select_dtypes('float64','int64')) > 0.95).astype(int)
corr.to_excel(writer1, sheet_name = 'all_strains')
sig.to_excel(writer1, sheet_name = 'all_strains_above_0.95')

# Look at correlation by strain
my_count = 0
for group in dt.groupby(['Folder']):
    corr = group[1].corr()
    plt.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gcf().set_size_inches(14.5, 14.5)
    plt.title('Correlation between features in strain ' + strains[my_count], pad = 60)
    plt.savefig('PCA_correlation_figures/' + strains[my_count] + '_corr.png')
    plt.close()

    corr.to_excel(writer1, sheet_name = strains[my_count])
    sig = (abs(corr.select_dtypes('float64','int64')) > 0.95).astype(int)
    sig.to_excel(writer1, sheet_name = strains[my_count] + '_above_0.95')

    my_count += 1

writer1.save()

#####################################################################################
# PCA: uses orthogonal transformation to reduce excessive multicollinearity,
# suitable for unsupervised learning when explanation of predictors
# is not important
#####################################################################################
# Standardize the data to optimize performance
x = nums_only_dt.loc[:,features].values
y = df.loc[:,['Folder']].values
x = StandardScaler().fit_transform(x)
# Set up 2D PCA
pca = skl.PCA(n_components = 2)
principalComponents = pca.fit_transform(nums_only_dt)
# Generate the percent variance explained by each component
evr2 = pca.explained_variance_ratio_
print(evr2)
# Create data from for PCA output
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1','pc2'])
principalDf.to_excel(writer2, sheet_name = 'PCA2_raw_df')
finalDf = pd.concat([principalDf, dt[['Folder']]], axis = 1)
finalDf.to_excel(writer2, sheet_name = 'PCA2_final_df')
# Graph PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
colors = ['r', 'g', 'b', '#eeefff', 'y', 'c', 'k', 'm']
for strain, color in zip(strains, colors):
    # For the selected color, you only want one strain to be plotted
    indicesToKeep = finalDf['Folder'] == strain
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep, 'pc2'],
               c = color, s = 50)
ax.legend(strains)
ax.grid()
fig.savefig('PCA_correlation_figures/PCA2.png')
fig.show()

# Try different numbers of components
pca = skl.PCA(n_components = 5)
pca.fit(nums_only_dt)
evr5 = pca.explained_variance_ratio_
print(evr5)

# Set up 2D PCA for high variance components only
pca = skl.PCA(n_components = 2)
principalComponents = pca.fit_transform(pd.concat([dt_high_var.select_dtypes('float64'),
    dt_high_var.select_dtypes('int64')], axis = 1))
# Generate the percent variance explained by each component
evr2_hv = pca.explained_variance_ratio_
print(evr2_hv)
# Create data from for PCA output
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1','pc2'])
principalDf.to_excel(writer2, sheet_name = 'PCA2_raw_df_high_var')
finalDf = pd.concat([principalDf, dt[['Folder']]], axis = 1)
finalDf.to_excel(writer2, sheet_name = 'PCA2_final_df_high_var')
# Graph PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
colors = ['r', 'g', 'b', '#eeefff', 'y', 'c', 'k', 'm']
for strain, color in zip(strains, colors):
    # For the selected color, you only want one strain to be plotted
    indicesToKeep = finalDf['Folder'] == strain
    ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep, 'pc2'],
               c = color, s = 50)
ax.legend(strains)
ax.grid()
fig.savefig('PCA_correlation_figures/PCA2_high_var.png')
fig.show()

writer2.save()


#####################################################################################
# Forward/backward/stepwise selection: only keep the best/most accurate
# variables (ML extend module)
#####################################################################################




#####################################################################################
# LASSO: Least Absolute Shrinkage and Selection Operator: does feature
# selection for you for linear model
#####################################################################################




#####################################################################################
# Tree-based: evaluate importance of features using trees
#####################################################################################
