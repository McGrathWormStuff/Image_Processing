# https://www.youtube.com/watch?v=YaKMeAlHgqQ

import pandas as pd
import sys
import argparse

# Set up cmd arguments for the user
parser = argparse.ArgumentParser(description = 'Enter file name')
parser.add_argument('filename', type = str)
args = parser.parse_args()

dt = pd.read_csv(args.filename, index_col = 1)
print(dt)

# If there are a lot of missing values, you can skip that feature

# If a feature has very low variance (the values are not very
# different between images, then we should ignore it).

# Can drop one feature if it is pairwise correlating with another
# feature (reduce redundancy)
dt.corr()

# PCA: uses orthogonal transformation to reduce excessive multicollinearity,
# suitable for unsupervised learning when explanation of predictors
# is not important

# If the correlation with the target is low, feature can be dropped

# Forward/backward/stepwise selection: only keep the best/most accurate
# variables (ML extend module)

# LASSO: Least Absolute Shrinkage and Selection Operator: does feature
# selection for you for linear model

# Tree-based: evaluate importance of features using trees