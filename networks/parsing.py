import numpy as np
import pandas as pd

# Convert the first column of data to one hot
# Ex for values ranging from 0 to 2:
# 0 = [0, 0, 1]
# 1 = [0, 1, 0]
# 2 = [1, 0, 0]
# Returns : First column (as np.array)
# Returns : First column with one-hot encoding
def convertOneHot(data, column, max_value = None):
    y=np.array([int(i[column]) for i in data])
    y_onehot=[0]*len(y)
    if (max_value == None):
        max_value == y.max()
    for i,j in enumerate(y):
        y_onehot[i]=[0]*(max_value + 1)
        y_onehot[i][j]=1
    return (y,y_onehot)
    
def isCategory(col):
    for row in col:
        if isinstance(row, basestring):
            return True
    return False
    
def parse(file_path, start_column, end_column, qualitative = False, output_column = 0):
    df = pd.read_csv(file_path)
    parsed_df = pd.DataFrame()
    output = pd.DataFrame()
    for column in df:
        if output_column == df.columns.get_loc(column):
            if qualitative:
                output = pd.get_dummies(df[column])
            else:
                output = df[column]
        if df.columns.get_loc(column) > end_column or df.columns.get_loc(column) < start_column:
            continue
        elif isCategory(df[column]):
            parsed_df = pd.concat([parsed_df, pd.get_dummies(df[column])], axis=1)
        else:
            parsed_df[column] = df[column]
    return parsed_df.values, output.values

