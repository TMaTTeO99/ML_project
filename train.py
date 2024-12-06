from model import NeuralNetwork
import numpy as np
import math 
from sklearn.preprocessing import StandardScaler
# instantiate the neural network
model = NeuralNetwork([1,4,2,1], ['sigmoid','sigmoid','linear'])

print("Matrice dei pesi iniziali: ")
model.print_matrices_fancy(model.get_list_weight_matrices())
"""
matrix = np.array([ [1],
                    [2],
                    [3],
                    [4],
                    [5],
                     ]) 
#output = model.predict(matrix)
# print("Stampa finale")
# print(output)
"""
x = np.random.uniform(1,50,100).reshape(-1,1)
y = np.log2(x) #math.log(x, 2)

# model.train(matrix, np.array([[2], [4], [6], [8], [10]]))
model.train(x, y)

result = model.predict(np.array([ [1],
                    [2],
                    [4],
                    [8],
                    [16],
                     ]) )


print(f" predizione del doppio di 50, 65 , 32 , 114 e 43  : {result}")
