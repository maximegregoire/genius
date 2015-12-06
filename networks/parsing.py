import numpy as np

# Convert the first column of data to one hot
# Ex for values ranging from 0 to 2:
# 0 = [0, 0, 1]
# 1 = [0, 1, 0]
# 2 = [1, 0, 0]
# Returns : First column (as np.array)
# Returns : First column with one-hot encoding
def convertOneHot(data, column = 0):
    y=np.array([int(i[column]) for i in data])
    y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        y_onehot[i]=[0]*(y.max() + 1)
        y_onehot[i][j]=1
    return (y,y_onehot)
