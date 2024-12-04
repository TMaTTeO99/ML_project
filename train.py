from model import NeuralNetwork
import numpy as np

# instantiate the neural network
model = NeuralNetwork([3,2,1], ['sigmoid','linear'])

# print("Matrice dei pesi: ")
# model.print_matrices_fancy(model.get_list_weight_matrices())

    
matrix = np.array([[1, 2, 3]]) 
output = model.predict(matrix)
# print("Stampa finale")
# print(output)

model.train(matrix, [5])

