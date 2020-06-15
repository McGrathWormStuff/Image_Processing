# https://www.youtube.com/watch?v=YaKMeAlHgqQ

import pandas as pd
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Set up cmd arguments for the user
parser = argparse.ArgumentParser(description = 'Enter file names')
parser.add_argument('filename', type = str)
parser.add_argument('gtruth', type = str)
args = parser.parse_args()

# Extract data from csv files
df = pd.read_csv(args.filename, index_col = 1)
gt = pd.read_excel(args.gtruth, index_col = 1)

# Merge the two data tables based on the images that each file have in common
dt = pd.merge(df,gt,on='Name',how='inner')

strains = dt.index.unique()
features = dt.columns()
for strain in strains:
    temp = dt.loc[strain]

# Can get the averages by folder/strain
# avgs = dt.groupby(['Folder']).mean()



#####################################################################################
# If there are a lot of missing values, you can skip that feature
#####################################################################################




#####################################################################################
# If a feature has very low variance (the values are not very
# different between images, then we should ignore it).
#####################################################################################




#####################################################################################
# Can drop one feature if it is pairwise correlating with another
# feature (reduce redundancy)
#####################################################################################
# with open('correlations.txt','w') as f:
#     f.write(dt.corr())
#     f.write(dt.groupby(['Folder']).corr())
corr = dt.corr()

plt.figure(figsize=(10,5))
plt.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Correlation between features')
plt.savefig('corr.png')
plt.show()

plt.figure(figsize=(10,5))
plt.matshow(dt.groupby(['Folder']).corr())
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title('Correlation between features by strain')
plt.savefig('corr_by_strain.png')
plt.show()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)

#####################################################################################
# PCA: uses orthogonal transformation to reduce excessive multicollinearity,
# suitable for unsupervised learning when explanation of predictors
# is not important
#####################################################################################




#####################################################################################
# If the correlation with the target is low, feature can be dropped
#####################################################################################




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
