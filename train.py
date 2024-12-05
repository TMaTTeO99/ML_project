from model import NeuralNetwork
import numpy as np

# instantiate the neural network
model = NeuralNetwork([1,4,1], ['relu','linear'])

print("Matrice dei pesi: ")
model.print_matrices_fancy(model.get_list_weight_matrices())

    
matrix = np.array([ [1],
                    [2],
                    [3],
                    [4],
                    [5],
                    [6],
                    [7] ]) 
#output = model.predict(matrix)
# print("Stampa finale")
# print(output)

model.train(matrix, np.array([[2], [4], [6], [8], [10], [12],[14]]))

result = model.predict(np.array(  [[50], [65], [32],[114]] ))
print(f" predizione del doppio di 50, 65 e 32 e 114 : {result}")
