# https://www.youtube.com/watch?v=YaKMeAlHgqQ

import pandas as pd
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import SegmentAnalysis as SA

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
# Multiple loads
out_multiple_loads1 = SA.pca_plot(nums_only_dt, ['BKMeanInt', 'BKStdInt', 'BKPerct95', 'BKMedianInt',
    'BKSumInt', 'est_len', 'tipWidthRatio', 'BodMeanInt', 'BodStdInt', 'BodSumInt',
    'BodAvgWidth', 'Single_Loaded' ,'border_sum'], ['Single_Loaded'] , '1')
out_multiple_loads2 = SA.pca_plot(nums_only_dt, ['BKMeanInt', 'BKStdInt', 'Single_Loaded'], ['Single_Loaded'], '2')
out_multiple_loads3 = SA.pca_plot(nums_only_dt, nums_only_dt.columns, ['Single_Loaded'], '3')

# Whole animal
out_whole_animal = SA.pca_plot(nums_only_dt, ['Whole_Animal', 'border_sum', 'intRatio2-9', 'MeanInt-9',
    'stdInt-9', 'perct95-9', 'border_sum'], ['Whole_Animal'])

# Straight
out_straight = SA.pca_plot(nums_only_dt, ['Straight', 'est_len', 'intRatio2-9', 'BodMeanInt',
    'BodStdInt', 'BodSumInt', 'BodAvgWidth', 'MeanInt-10', 'avgWidth-10', 'stdInt-10', 'perct95-10',
    'medianInt-10', 'sumInt-10', 'est_len'], ['Straight'])

# Clear
out_clear = SA.pca_plot(nums_only_dt, ['Clear', 'BodMeanInt', 'stdInt-1', 'stdInt-2', 'stdInt-3',
    'stdInt-4', 'stdInt-5', 'stdInt-6', 'stdInt-7', 'stdInt-8', 'stdInt-9', 'BodStdInt', 'stdInt-10',
    'MeanInt-1', 'MeanInt-2', 'MeanInt-3', 'MeanInt-4', 'MeanInt-5', 'MeanInt-6', 'MeanInt-7',
    'MeanInt-8', 'MeanInt-9', 'MeanInt-10', 'BodMeanInt', 'perct95-1', 'perct95-2', 'perct95-3',
    'perct95-4', 'perct95-5', 'perct95-6', 'perct95-7', 'perct95-8', 'perct95-9' , 'perct95-10'], ['Clear'])

# Head first
out_head_first = SA.pca_plot(nums_only_dt, ['Head_First', 'BodMeanInt', 'MeanInt-1', 'MeanInt-2',
    'MeanInt-3', 'MeanInt-4', 'MeanInt-5', 'MeanInt-6', 'MeanInt-7', 'MeanInt-8', 'MeanInt-9',
    'MeanInt-10', 'BodMeanInt', 'perct95-1', 'perct95-2', 'perct95-3', 'perct95-4', 'perct95-5',
    'perct95-6', 'perct95-7', 'perct95-8', 'perct95-9' , 'perct95-10', 'intRatio2-9', 'stdInt-9'],
    ['Head_First'])

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
