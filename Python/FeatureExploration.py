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
# plt.ioff()

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
nums_only_dt = pd.concat([dt[['mean_fourier_freq_real', 'mean_fourier_freq_imag']],
    dt.select_dtypes('int64'), dt.select_dtypes('float64')], axis = 1)

# Set the strain to be the index instead of the image name now that we have
# combined the files
strains = dt.Folder.unique()
main_five = ['Single_Loaded', 'Clear', 'Straight', 'Head_First', 'Whole_Animal']
features = nums_only_dt.drop(main_five, axis = 1).columns


#####################################################################################
# Fourier Transformation analysis
#####################################################################################
bins = np.linspace(-10, 10, 100)
bins2 = np.linspace(-1e-16, 1e-16, 100)
# Clear
graphing = dt[['Clear', 'mean_fourier_freq_real']]
clear = graphing[graphing.Clear == 1]['mean_fourier_freq_real']
blurry = graphing[graphing.Clear == 0]['mean_fourier_freq_real']
plt.hist(clear.values, bins = bins, alpha = 0.7, rwidth=0.85, label = 'Clear')
plt.hist(blurry.values, bins = bins, alpha = 0.7, rwidth=0.85, label = 'Blurry')
plt.xlabel('Frequencies')
plt.xlim(-3.1,5.1)
plt.ylabel('Count')
plt.title('Fourier Transformation Frequencies (Real)')
plt.legend(loc = 'best')
plt.savefig('feature_exploration/fft_clear_real.png')
plt.close()
# print(SA.my_auc(np.array([graphing['Clear'].to_numpy()]), np.array([graphing['mean_fourier_freq_real'].to_numpy()]).T,
#     'FFT_Clear'))
graphing = dt[['Clear', 'mean_fourier_freq_imag']]
clear = graphing[graphing.Clear == 1]['mean_fourier_freq_imag']
blurry = graphing[graphing.Clear == 0]['mean_fourier_freq_imag']
plt.hist(clear.values, bins = bins2, alpha = 0.7, rwidth=0.85, label = 'Clear')
plt.hist(blurry.values, bins = bins2, alpha = 0.7, rwidth=0.85, label = 'Blurry')
plt.xlabel('Frequencies')
plt.ylabel('Count')
plt.title('Fourier Transformation Frequencies (Imag)')
plt.legend(loc = 'best')
plt.savefig('feature_exploration/fft_clear_imag.png')
plt.close()
# Head first
graphing = dt[['Head_First', 'mean_fourier_freq_real']]
hf = graphing[graphing.Head_First == 1]['mean_fourier_freq_real']
tf = graphing[graphing.Head_First == 0]['mean_fourier_freq_real']
plt.hist(hf.values, bins = bins, alpha = 0.7, rwidth=0.85, label = 'Head First')
plt.hist(tf.values, bins = bins, alpha = 0.7, rwidth=0.85, label = 'Tail First')
plt.xlabel('Frequencies')
plt.xlim(-3.1,5.1)
plt.ylabel('Count')
plt.title('Fourier Transformation Frequencies (Real)')
plt.legend(loc = 'best')
plt.savefig('feature_exploration/fft_head_first_real.png')
plt.close()
graphing = dt[['Head_First', 'mean_fourier_freq_imag']]
hf = graphing[graphing.Head_First == 1]['mean_fourier_freq_imag']
tf = graphing[graphing.Head_First == 0]['mean_fourier_freq_imag']
plt.hist(hf.values, bins = bins2, alpha = 0.7, rwidth=0.85, label = 'Head First')
plt.hist(tf.values, bins = bins2, alpha = 0.7, rwidth=0.85, label = 'Tail First')
plt.xlabel('Frequencies')
plt.ylabel('Count')
plt.title('Fourier Transformation Frequencies (Imag)')
plt.legend(loc = 'best')
plt.savefig('feature_exploration/fft_head_first_imag.png')
plt.close()
# SA.my_auc(np.array([graphing['Head_First'].to_numpy()]), np.array([graphing['mean_fourier_freq_real'].to_numpy()]).T,
#     'FFT_Head_First')
sys.exit()

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
pca, principalDf = SA.my_pca(nums_only_dt, features, 8)
pca_corr_dt = pd.DataFrame()
for feature in main_five:
    out = SA.pca_correlation(nums_only_dt[feature], principalDf, feature)
    pca_corr_dt[feature] = out
    SA.pca_plot(nums_only_dt, principalDf, feature)
    highest_corr = out.abs().nlargest(2).index
    # Plot the principal components which are most highly correlated with each main feature
    SA.pca_plot(nums_only_dt, principalDf, feature, highest_corr[0], highest_corr[1])


