import pandas as pd 
from model import NeuralNetwork
import numpy as np
import myModelParameters as mmp
import os 

import math 


modelParametersFile = "./finalModel.txt"




# Carica il file 
file_path = "./dataset/monk/monks-1.train"
data = pd.read_csv(file_path, sep=" ", header=None, skipinitialspace=True)

path_test = "./dataset/monk/monks-1.test"
test_set = pd.read_csv(path_test, sep=" ", header=None, skipinitialspace=True)

# drop last column not useful 
data = data.drop(data.columns[-1], axis=1)

data_test = test_set.drop(test_set.columns[-1], axis=1)

target = data.iloc[:, 0]  # Seleziona la prima colonna (indice 0)
target_test = data_test.iloc[:, 0]

training_set = data.iloc[:, 1:]  # Tutte le colonne tranne la prima
training_set = pd.get_dummies(training_set, columns=training_set.columns[0:], drop_first=False)
input_test = data_test.iloc[:, 1:]
input_test = pd.get_dummies(input_test, columns=input_test.columns[0:], drop_first=False)

# convert to numpy 
x = training_set.to_numpy()
x_test = input_test.to_numpy()

#print(x_test)
#print(x_test.shape[0])
#print(x_test.shape[1])
#print("\n")

y = target.to_numpy().reshape(-1, 1)
y_test = target_test.to_numpy().reshape(-1, 1)

# if you want to use tanh 
y[y == 0] = -1
y_test[y_test == 0] = -1

#print(y)
#print(y.shape[0])
#print(y.shape[1])

# added selection mode (standard mode, debug mode)
debugMode = False

while True:
    try :
        input = input("insert \"y\" for debug mode or others for standard mode\n")
        if input == "y":
            debugMode = True
        else: debugMode = False
        break
    except ValueError :
        print(f"errore: {ValueError}")

if debugMode == False and os.path.isfile(modelParametersFile) :

    objectReloaded = NeuralNetwork.realoadModel()
    model = NeuralNetwork(objectReloaded.units_for_levels, objectReloaded.activation, debugMode)
    
    result = model.predict_class(x_test, True, "tanh", objectReloaded.weights)
    print(f"classification error on TR set: {model.classification_error(result, y_test)}")

else :

    # instantiate the neural network
    model = NeuralNetwork([17,4,1], ['sigmoid','tanh'], debugMode)

    model.train(x, y, 500, 0.00001, "xavier", 1)

    # retreive all parameters from model
    mymodelparameters = mmp.myModelParameters(*model.getParameters())

    # save all parameters in a file
    model.saveModel(mymodelparameters)

    # predict after training
    result = model.predict_class(x_test, False)

    print(f"classification error on TR set: {model.classification_error(result, y_test)}")

    """
    result = model.predict(np.array([ [4],
                        [8],
                        [16],
                        [32],
                        [64],
                        ]) )
    """

    # print(f" predizione del log in base 2 di 1, 2 , 4 , 8 e 16  : {result}")