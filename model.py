import numpy as np 
import time 

class NeuralNetwork():
    
    def sigmoid(self, x, a=1, derivative =False):
        if(derivative):
            exp_term = np.exp(-a * x)
            return a * exp_term / (1 + exp_term)**2
        return 1/(1+np.exp(-a*x))
    
    def relu(self, x, derivative =False):
        if(derivative):
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0,x)
    
    def linear(self, x, derivative =False):
        if derivative:
            return np.ones_like(x)
        return x
    
    ## TO-DO add more activation functions like tanh 
    
    @staticmethod
    def random_matrix(rows, columns, min_val=-0.7, max_val=0.7):
        return np.random.uniform(low=min_val, high=max_val, size=(rows, columns))
         
    def initalizeWeightMatrix (self):
        weightMatricesList = []
        for i in range(len(self.units_for_levels)-1): 
            weightMatricesList.append(self.random_matrix(self.units_for_levels[i], self.units_for_levels[i+1]))
        return weightMatricesList

    @staticmethod
    def print_matrices_fancy(list_to_print):
        for idx, matrix in enumerate(list_to_print):
            print(f"Matrix {idx + 1}:")
            for row in matrix:
                formatted_row = " | ".join(f"{elem:6.2f}" for elem in row)
                print(f"| {formatted_row} |")
            print("-" * (len(matrix[0]) * 10))  # Separator between matrices

    
    def initializeActivationFunctions(self, activation):
        # declare an empty list of activation functions
        self.activation = []
        for levels in range(len(activation)):
            if activation[levels] == 'sigmoid':
                self.activation.append(self.sigmoid)
            elif  activation[levels] == 'relu':
                self.activation.append(self.relu)
            elif  activation[levels] == 'linear':
                self.activation.append(self.linear)

    # constructor  
    # units_for_levels is the list of units for each level, the length of the list is the number of levels
    # activation is a list of string with the name of the activation function for all the levels 
    def __init__(self, units_for_levels, activation): 
        self.units_for_levels = units_for_levels
        self.numberOfLevels = len(units_for_levels)-1
        # print(f"number of levels: {self.numberOfLevels}")
        # initial the activation function 
        self.initializeActivationFunctions(activation)
       
        # initialization of the weight matrix to random small values 
        # we need a list of matrices, one for each layer, 
        # each matrix column represents the weight for a single unit of that level 
        # matrix for level l in position i,j  has the weight from unit j to unit i
        # the matix is m x n where m is the number of input unit and n is the number of the unit of that level

        self.listOfWeightMatrices = self.initalizeWeightMatrix()
    
    # inputX is an input Matrix 
    def feedForeward (self, inputX):

        localInputX = inputX
        self.listOfHiddenRepr = []
        self.listOfHiddenRepr.append(inputX)
        self.listOfNet = []
        #level means W matrix for all levels
        for level in range(len(self.listOfWeightMatrices)):
            # compute the net 
            localInputX = np.matmul(localInputX, self.listOfWeightMatrices[level])
            # save the net 
            self.listOfNet.append(localInputX)
            # compute the hidden unit output 
            localInputX = self.activation[level](localInputX)
            self.listOfHiddenRepr.append(localInputX)

            # print for debug
            print("Stampa rappresentazione interna")
            self.print_matrices_fancy(self.listOfHiddenRepr)

        # localInputX == output of feedForeward of the NN  
        return localInputX; 

    # y is the target matrix
    # o is the output matrix 
    def backPropagate(self, y, o):
        error = y - o 
        # print(f"len of listOfNet: {len(self.listOfNet)}")
        net_k = self.listOfNet[self.numberOfLevels-1]
        delta_k = error * self.activation[self.numberOfLevels-1](net_k, True)
        listOfDelta = []
        delta_temp = delta_k
        for levels in range(self.numberOfLevels-1,1,-1):
            delta_temp = delta_temp * self.listOfWeightMatrices[levels]
            net_j = self.listOfNet[levels-1]
            delta_temp = delta_temp * self.activation[levels-1](net_j, derivative = True)
            listOfDelta.append(delta_temp)
        
        # compute the gradients for each level 
        grad_output = delta_k * self.listOfHiddenRepr[self.numberOfLevels-1]
        grad_hidden = []
        for levels in range(self.numberOfLevels-1,1,-1):
            grad_hidden.append(listOfDelta[levels]*self.listOfHiddenRepr[levels-1])
        return grad_output, grad_hidden


    def train (self, X, Y ):
        # compute model output 
        o = self.feedForeward(X)
        self.backPropagate(Y, o)
        print("Non sono esploso")


            

    def predict(self, inputX):
        return self.feedForeward(inputX)



    def get_list_weight_matrices(self):
        return self.listOfWeightMatrices

