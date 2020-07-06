# https://www.youtube.com/watch?v=YaKMeAlHgqQ

import pandas as pd
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import SegmentAnalysis as SA
import mlxtend.feature_selection as mlx

# Turn off interactive plotting (uncomment this if you want to show the plots
# as they are created and replace plt.close() with plt.show())
plt.ioff()

# Set up data output
writer1 = pd.ExcelWriter('feature_exploration/correl_data.xlsx', engine = 'xlsxwriter')

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
# Can drop one feature if it is pairwise correlating with another
# feature (reduce redundancy)
#####################################################################################
# Correlation, regarless of strain
corr = dt.corr()
# Show figure
plt.matshow(corr, fignum = 10)
# Plot
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.rcParams.update({'font.size':10})
plt.gcf().set_size_inches(14.5, 14.5)
plt.title('Correlation of All Strains', pad = 60)
plt.savefig('feature_exploration/corr.png')
plt.close()

# Identify all relationships that are more correlated than 0.95
sig = (abs(corr.select_dtypes('float64','int64')) > 0.95).astype(int)
corr.to_excel(writer1, sheet_name = 'all_strains')
sig.to_excel(writer1, sheet_name = 'all_strains_above_0.95')

# Look at correlation by strain
for group, strain in zip(dt.groupby(['Folder']), strains):
    corr = group[1].corr()
    plt.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gcf().set_size_inches(14.5, 14.5)
    plt.title('Correlation between features in strain ' + strain, pad = 60)
    plt.savefig('feature_exploration/' + strain + '_corr.png')
    plt.close()

    corr.to_excel(writer1, sheet_name = strain)
    sig = (abs(corr.select_dtypes('float64','int64')) > 0.95).astype(int)
    sig.to_excel(writer1, sheet_name = strain + '_above_0.95')

writer1.save()

#####################################################################################
# PCA: uses orthogonal transformation to reduce excessive multicollinearity,
# suitable for unsupervised learning when explanation of predictors
# is not important
#####################################################################################
# Multiple loads
out_multiple_loads1 = SA.pca_plot(nums_only_dt, ['BKMeanInt', 'BKStdInt',
    'BKSumInt', 'est_len', 'tipWidthRatio', 'BodMeanInt', 'BodStdInt', 'BodSumInt',
    'BodAvgWidth', 'border_sum'], ['Single_Loaded'] , '1')
out_multiple_loads2 = SA.pca_plot(nums_only_dt, ['BKMeanInt', 'BKStdInt'],
    ['Single_Loaded'], '2')

# Whole animal
out_whole_animal = SA.pca_plot(nums_only_dt, ['intRatio2-9', 'MeanInt-9',
    'stdInt-9', 'perct95-9', 'border_sum'], ['Whole_Animal'])

# Straight
out_straight = SA.pca_plot(nums_only_dt, ['est_len', 'intRatio2-9', 'BodMeanInt',
    'BodStdInt', 'BodSumInt', 'BodAvgWidth', 'MeanInt-10', 'avgWidth-10', 'stdInt-10', 'perct95-10',
    'medianInt-10', 'sumInt-10', 'est_len'], ['Straight'])

# Clear
out_clear = SA.pca_plot(nums_only_dt, ['BodMeanInt', 'stdInt-1', 'stdInt-2', 'stdInt-3',
    'stdInt-4', 'stdInt-5', 'stdInt-6', 'stdInt-7', 'stdInt-8', 'stdInt-9', 'BodStdInt', 'stdInt-10',
    'MeanInt-1', 'MeanInt-2', 'MeanInt-3', 'MeanInt-4', 'MeanInt-5', 'MeanInt-6', 'MeanInt-7',
    'MeanInt-8', 'MeanInt-9', 'MeanInt-10', 'BodMeanInt', 'perct95-1', 'perct95-2', 'perct95-3',
    'perct95-4', 'perct95-5', 'perct95-6', 'perct95-7', 'perct95-8', 'perct95-9' , 'perct95-10'], ['Clear'])

# Head first
out_head_first = SA.pca_plot(nums_only_dt, ['BodMeanInt', 'MeanInt-1', 'MeanInt-2',
    'MeanInt-3', 'MeanInt-4', 'MeanInt-5', 'MeanInt-6', 'MeanInt-7', 'MeanInt-8', 'MeanInt-9',
    'MeanInt-10', 'BodMeanInt', 'perct95-1', 'perct95-2', 'perct95-3', 'perct95-4', 'perct95-5',
    'perct95-6', 'perct95-7', 'perct95-8', 'perct95-9' , 'perct95-10', 'intRatio2-9', 'stdInt-9'],
    ['Head_First'])


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
