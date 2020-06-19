import numpy as np

def stats(img):
    MeanInt = np.mean(img[np.nonzero(img)])
    stdInt = np.std(img[np.nonzero(img)])
    perct95 = np.percentile(img[np.nonzero(img)],95)
    medianInt = np.percentile(img[np.nonzero(img)],50)
   
    [row, col] = np.nonzero(img)
    avgWidth = max(row) - min(row)

    return([MeanInt, avgWidth, stdInt, perct95, medianInt])
