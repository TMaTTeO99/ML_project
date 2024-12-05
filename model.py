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
        for i in range(self.numberOfLevels): 
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
    # at the end of feedForeward I should have as many elements as number of levels in both listOfHiddenRepr and listOfNet
    # 0,.., level-1 
    def feedForeward (self, inputX):

        localInputX = inputX
        self.listOfHiddenRepr = []
        #self.listOfHiddenRepr.append(inputX)
        self.listOfNet = []
        #level means W matrix for all levels
        for level in range(self.numberOfLevels):
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
    def backPropagate(self, x, y, o):
        
        error = y - o 
        # print(f"len of listOfNet: {len(self.listOfNet)}")
        net_k = self.listOfNet[self.numberOfLevels-1]
        
        # Previus, here there is an error: 
        # self.activation[self.numberOfLevels-1](net_k, True) need to be  transpose
        # and we need to change the multiplication 
        # delta_k = np.matmul(error,self.activation[self.numberOfLevels-1](net_k, True))

        #delta_k = np.matmul(error, self.activation[self.numberOfLevels-1](net_k, True).T)
        delta_k = np.zeros((error.shape[0], error.shape[1]))

        for pattern in range(x.shape[0]):
            delta_k[pattern] = error[pattern] * self.activation[self.numberOfLevels-1](net_k, True)[pattern]
            
        #print(f" delta_k: \n {delta_k.shape[0], delta_k.shape[1]}")


        listOfDelta = []
        delta_temp = delta_k

        for levels in range(self.numberOfLevels-1,0,-1):
            delta_temp = np.matmul(delta_temp,self.listOfWeightMatrices[levels].T)
            net_j = self.listOfNet[levels-1]

            for pattern in range(x.shape[0]):
                delta_temp[pattern] = delta_temp[pattern] * self.activation[levels-1](net_j, derivative = True)[pattern]
            
            
            # like before here there are few errors (look notes) 
            #delta_temp = np.matmul(delta_temp,self.activation[levels-1](net_j, derivative = True).T)
            listOfDelta.append(delta_temp)
        
        
        # compute the gradients for each level 
        grad_output = np.matmul(delta_k.T, self.listOfHiddenRepr[self.numberOfLevels-1])
        grad_hidden = []

        """ 
        for levels in range(self.numberOfLevels-1,0,-1):
            if levels == 1:
                grad_hidden.append(np.matmul(listOfDelta[levels-1].T, x).T)
            else:
                grad_hidden.append(np.matmul(listOfDelta[levels-1].T, self.listOfHiddenRepr[levels-2]).T)
        """
        listOfDelta.reverse()
        for levels in range(0,self.numberOfLevels-1,1):
            if levels == 0:
                grad_hidden.append(np.matmul(listOfDelta[levels].T, x).T)
            else:
                grad_hidden.append(np.matmul(listOfDelta[levels].T, self.listOfHiddenRepr[levels-1]).T)


        return grad_output.T, grad_hidden


    def train (self, X, Y ):
        

        i = 0
        while i < 100 :

            # compute model output 
            o = self.feedForeward(X)
            grad_output, grad_hidden = self.backPropagate(X, Y, o)
            # grad_hidden.reverse()

            for j in range(0, len(grad_hidden) - 1, 1):
                # print(f"Shape of weight matrix {j}: {self.listOfWeightMatrices[j].shape}")
                # print(f"Shape of gradient {j}: {grad_hidden[j].shape}")
                self.listOfWeightMatrices[j] = self.listOfWeightMatrices[j] + (0.01 * grad_hidden[j]) 
            
            self.listOfWeightMatrices[len(self.listOfWeightMatrices)-1] = self.listOfWeightMatrices[len(self.listOfWeightMatrices)-1] + (0.01 * grad_output) 
            
            i += 1

        # print(f"grad_output : {grad_output}")
        # print(f"grad_hidden : {grad_hidden}")
        


        #print("grad_hidden")
        #print(f"{grad_hidden}")
        #print("grad_output")
        #print(f"{grad_output}")

            

    def predict(self, inputX):
        return self.feedForeward(inputX)



    def get_list_weight_matrices(self):
        return self.listOfWeightMatrices

