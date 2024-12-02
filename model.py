import numpy as np 
import time 

class NeuralNetwork():
    # constructor  
    # units_for_levels is the list of units for each level, the length of the list is the number of levels
    # 
    

    def sigmoid(self, x, a=1):
        return 1/(1+np.exp(-a*x))
    
    def relu(self, x):
        return np.maximum(0,x)
    
    ## TO-DO add more activation functions like tanh 
    
    @staticmethod
    def random_matrix(rows, columns, min_val=-0.7, max_val=0.7):
        return np.random.uniform(low=min_val, high=max_val, size=(rows, columns))
         
    def initalizeWeightMatrix (self):
        weightMatricesList = []
        for i in range(len(self.units_for_levels)-1): 
            weightMatricesList.append(self.random_matrix(self.units_for_levels[i], self.units_for_levels[i+1]))
        return weightMatricesList

    def print_matrices_fancy(self):
        for idx, matrix in enumerate(self.listOfWeightMatrices):
            print(f"Matrix {idx + 1}:")
            for row in matrix:
                formatted_row = " | ".join(f"{elem:6.2f}" for elem in row)
                print(f"| {formatted_row} |")
            print("-" * (len(matrix[0]) * 10))  # Separator between matrices


    def __init__(self, units_for_levels, activation): 
        self.units_for_levels = units_for_levels
        
        if activation == 'sigmoid':
            self.activation = self.sigmoid
        elif activation == 'relu':
            self.activation = self.relu

       
        # initialization of the weight matrix to random small values 
        # we need a list of matrices, one for each layer, 
        # each matrix column represents the weight for a single unit of that level 
        # matrix for level l in position i,j  has the weight from unit j to unit i
        # the matix is m x n where m is the number of input unit and n is the number of the unit of that level

        self.listOfWeightMatrices = self.initalizeWeightMatrix()

