from model import NeuralNetwork
import numpy as np

# instantiate the neural network
model = NeuralNetwork([2,2,1], ['sigmoid','sigmoid'])

print("Matrice dei pesi: ")
model.print_matrices_fancy(model.get_list_weight_matrices())

    
matrix = np.array([ [1, 0],
                [1, 1],
                [0, 0],
                [0, 1] ]) 
#output = model.predict(matrix)
# print("Stampa finale")
# print(output)

model.train(matrix, np.array([[1], [0], [0], [1]]))

result = model.predict(np.array(  [[0, 0]] ))
print(f" predizione 0, 1 : {result}")
