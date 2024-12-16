import pandas as pd 
from model import NeuralNetwork
import numpy as np
import myModelParameters as mmp
import os 
import matplotlib.pyplot as plt



# Carica il file 
file_path = "./dataset/ML-CUP24-TR.csv"
data = pd.read_csv(file_path, comment="#", header=None)

# drop first column not useful 
data = data.drop(data.columns[0], axis=1)

#shuffle data-set
# random_state=1 meaning that for all run the result is the same
data = data.sample(frac=1, random_state=1)

# take row number 
numsample = data.shape[0]

#select 80 % of rows
trainingPercentage = numsample * 0.8


# take the rows for k-fold validation 
selectionSet = data.iloc[:int(trainingPercentage), :]

# take the row for final testing 
testSet =  data.iloc[int(trainingPercentage):, :]


# split target from input for k-fold validation
#targetForSelection = selectionSet.iloc[:, -3:] 
#trainingForSelection = selectionSet.iloc[:, :-3]


# split target from input for test
#targetTest = testSet.iloc[:, -3:] 
#inputTest = testSet.iloc[:, :-3]

# build numpy MATRIX for training and target for training
#targetForSelection = targetForSelection.to_numpy()
#trainingForSelection = trainingForSelection.to_numpy()

#like before
#targetTest = targetTest.to_numpy() 
#inputTest = inputTest.to_numpy()

selectionSet = selectionSet.to_numpy()
testSet = testSet.to_numpy()




# number of partitions
k = 4
testSplits = mmp.myModelParameters.kFoldPartition(selectionSet, k)

listOfDict = []

for numSplit, kfolSplit in enumerate(testSplits):

    print(f"k fold number: {numSplit}")
    trSetFold = kfolSplit[0]
    vlSetFold = kfolSplit[1]

    # split target from input for k-fold validation
    x_Training = trSetFold[:, :-3]
    y_Training = trSetFold[:, -3:]
        
    x_Validation = vlSetFold[:, :-3]
    y_Validation = vlSetFold[:, -3:]

    resultOptIperParam = mmp.myModelParameters.doGridSearch(x_Training, x_Validation, y_Training, y_Validation, [12,4,3], ['sigmoid','linear'], False)
    listOfDict.append(resultOptIperParam)


for dict in listOfDict:
    print(dict)



