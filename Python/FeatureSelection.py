import sys
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import SegmentAnalysis as SA
# Sklearn does not automatically import submodules
import sklearn.ensemble as skle
import sklearn.model_selection as sklms
import sklearn.linear_model as skllm
import sklearn.discriminant_analysis as sklda

# Set up cmd arguments for the user
parser = argparse.ArgumentParser(description = 'Enter absolute path and background coordinates')
parser.add_argument('filename', type = str, help = 'Features extracted from algorithm (one sheet)')
args = parser.parse_args()

# Open excel file
dt = pd.read_excel(args.filename)
# Drop all attributes which are not integers or floats
dt = pd.concat([dt.select_dtypes('int64'), dt.select_dtypes('float64')], axis = 1)
dt = dt.drop(['Unnamed: 0', 'Num'], axis = 1)
# Set up variables to use for later
main_five = ['Single_Loaded', 'Clear', 'Straight', 'Head_First', 'Whole_Animal']
attributes = dt.drop(main_five, axis = 1)


#####################################################################################
# LASSO: Least Absolute Shrinkage and Selection Operator: does feature
# selection for you for linear model (L1 Regression)
#####################################################################################
# Lasso from SegmentAnalysis.py
print('\nLasso')
lass1, grid1 = SA.my_lasso(dt, 'Single_Loaded')
lass2, grid2 = SA.my_lasso(dt, 'Clear')
lass3, grid3 = SA.my_lasso(dt, 'Straight')
lass4, grid4 = SA.my_lasso(dt, 'Head_First')
lass5, grid5 = SA.my_lasso(dt, 'Whole_Animal')
lass_weights = pd.DataFrame([lass1.coef_, lass2.coef_, lass3.coef_, lass4.coef_, lass5.coef_])
lass_weights.columns = attributes.columns
lass_weights['Features'] = main_five
lass_weights.to_excel('feature_selection/lasso_weights.xlsx')
# plt.matshow(lass_weights.drop(['Features'], axis = 1))
# plt.xticks(range(len(attributes.columns)), attributes.columns, rotation=90)
# plt.yticks(range(len(main_five)), main_five)
# plt.title('Lasso Weights for Each Classification', pad = 120)
# plt.savefig('feature_selection/lasso_weights' + '.png')
# plt.show()

a = lass_weights.drop(['Features'], axis = 1).T
a.columns = main_five
for feature in main_five:
    print(feature)
    print(a.abs().nlargest(5, feature)[feature])


# L1 regression
print('\nL1')
# l1_1, coefs1 = SA.l1_reg(dt,'Single_Loaded')
# l1_2, coefs2 = SA.l1_reg(dt, 'Clear')
# l1_3, coefs3 = SA.l1_reg(dt, 'Straight')
# l1_4, coefs4 = SA.l1_reg(dt, 'Head_First')
# l1_5, coefs5 = SA.l1_reg(dt, 'Whole_Animal')

# for coef, feature in zip([coefs1.T, coefs2.T, coefs3.T, coefs4.T, coefs5.T], main_five):
#     print(feature)
#     for i in coef.columns:
#         print('C = ' + str(i))
#         print(attributes.columns[coef[i].abs().nlargest(5).index])
#     plt.matshow(coef)
#     plt.title('change in coeffs by c for ' + feature)
#     plt.savefig('feature_selection/change_in_coeffs_by_c_for_' + feature + '.png')
#     plt.show()
#     print('\n')


#####################################################################################
# LDA
#####################################################################################
# Use lda function from SegmentAnalysis.py
print('\nLDA')
# test_acc1, lda1 = SA.lda(attributes, dt['Single_Loaded'], 'Single_Loaded')
# test_acc2, lda2 = SA.lda(attributes, dt['Clear'], 'Clear')
# test_acc3, lda3 = SA.lda(attributes, dt['Straight'], 'Straight')
# test_acc4, lda4 = SA.lda(attributes, dt['Head_First'], 'Head_First')
# test_acc5, lda5 = SA.lda(attributes, dt['Whole_Animal'], 'Whole_Animal')
# # Put all of the weight coefficients from lda into a single pandas data frame
# lda_weights = pd.DataFrame([lda1.coef_[0], lda2.coef_[0], lda3.coef_[0], lda4.coef_[0], lda5.coef_[0]])
# lda_weights.columns = attributes.columns
# # Get the absolute values of the weights for later
# abs_lda_weights = lda_weights.abs()
# # Add a column designating which feature corresponds to which row of the data frame
# lda_weights.index = main_five
# # Plot weights
# plt.matshow(lda_weights)
# plt.yticks(range(len(main_five)), main_five)
# plt.title('Coefficient values for each feature from LDA', pad = 65)
# plt.xticks(range(len(attributes.columns)), attributes.columns, rotation=90)
# plt.savefig('feature_selection/lda_weights_heatmap.png')
# plt.show()
# # Output the weights to an excel file
# lda_weights.to_excel('feature_selection/lda_weights.xlsx')

# # Identify largest weights
# abs_lda_weights = abs_lda_weights.T
# abs_lda_weights.columns = main_five
# for column in main_five:
#     print(column + ': \n')
#     print(abs_lda_weights.nlargest(5, column)[column])


#####################################################################################
# Forward/backward/stepwise selection: only keep the best/most accurate
# variables (ML extend module)
#####################################################################################
# Step forward feature selection from SegmentAnalysis.py
print('\nSFS')
# sfs1, classifier1, data1 = SA.step_forward(attributes, dt['Single_Loaded'], 'Single_Loaded')
# sfs2, classifier2, data2 = SA.step_forward(attributes, dt['Clear'], 'Clear')
# sfs3, classifier3, data3 = SA.step_forward(attributes, dt['Straight'], 'Straight')
# sfs4, classifier4, data4 = SA.step_forward(attributes, dt['Head_First'], 'Head_First')
# sfs5, classifier5, data5 = SA.step_forward(attributes, dt['Whole_Animal'], 'Whole_Animal')
# sfs_data = pd.concat([data1, data2, data3, data4, data5])
# sfs_data.to_excel('feature_selection/sfs_data.xlsx')