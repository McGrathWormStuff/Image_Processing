import numpy as np
import sys
import sklearn.decomposition as skl
import sklearn.preprocessing as sklp
import matplotlib.pyplot as plt
import pandas as pd


def stats(img):
    MeanInt = np.mean(img[np.nonzero(img)])
    stdInt = np.std(img[np.nonzero(img)])
    perct95 = np.percentile(img[np.nonzero(img)],95)
    medianInt = np.percentile(img[np.nonzero(img)],50)
   
    [row, col] = np.nonzero(img)
    avgWidth = max(row) - min(row)

    SumInt = np.sum(img[np.nonzero(img)])

    return [MeanInt, avgWidth, stdInt, perct95, medianInt, SumInt]

def standardize(df, features):
    scaler = sklp.StandardScaler()
    scaled_dt = pd.DataFrame(scaler.fit_transform(df.values))
    scaled_dt.columns = features
    return scaled_dt

def pca_plot(dt, features, target, name_tag = ''):
    x = dt.loc[:, features]
    y = dt.loc[:, target]
    scaled_dt = standardize(x, features)
    pca = skl.PCA(n_components = 2)
    principalComponents = pca.fit_transform(scaled_dt)
    print(pca.explained_variance_ratio_)
    # Create data from PCA output
    principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1','pc2'])
    # principalDf.to_excel(writer2, sheet_name = 'PCA2_raw_df_' + target)
    finalDf = pd.concat([principalDf, dt[target]], axis = 1)
    # finalDf.to_excel(writer2, sheet_name = 'PCA2_final_df_' + target)
    # Graph PCA
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA: ' + target[0], fontsize = 20)
    colors = ['r', 'g']
    for i, color in zip((0,1), colors):
        # For the selected color, you only want one strain to be plotted
        indicesToKeep = finalDf[target[0]] == i
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1'], finalDf.loc[indicesToKeep, 'pc2'],
                   c = color, s = 50)
    ax.legend(['Not ' + target[0], target[0]])
    ax.grid()
    fig.savefig('PCA_correlation_figures/PCA2_' + target[0] + name_tag + '.png')
    fig.show()
    return fig, pca