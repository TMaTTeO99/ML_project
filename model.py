import numpy as np 
import time 
import pickle
import copy


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
    
    def tanh(self, x, a=1, derivative=False):
        if derivative:
            tanh_x = np.tanh(a * x)
            return a * (1 - tanh_x**2)
        return np.tanh(a * x)
    
    ## TO-DO add more activation functions like tanh 

    #w_{i,j}: arrow from unit i to unit j 
    @staticmethod
    def random_matrix(rows, columns, min_val, max_val):
        return np.random.uniform(low=min_val, high=max_val, size=(rows, columns))
         
    def getRandomWeights(self) :

        weightMatricesList = []
        for i in range(self.numberOfLevels): 
            weightMatricesList.append(self.random_matrix(self.units_for_levels[i], self.units_for_levels[i+1], -0.7, 0.7))
        return weightMatricesList

    def getXavierWeights(self):

        weightMatricesList = []
        for i in range(self.numberOfLevels): 
            weightMatricesList.append(self.random_matrix(self.units_for_levels[i], self.units_for_levels[i+1], -1/np.sqrt(self.units_for_levels[i]), 1/np.sqrt(self.units_for_levels[i])))
        return weightMatricesList



    def initalizeWeightMatrix(self, mode):

        match mode:
            case "random" :
                return self.getRandomWeights()
            case "xavier" :
                return self.getXavierWeights()
        
    
    """
    def mean_squared_error_loss (self, Y, O):
        num_pattern = Y.shape[0]
        diff = Y - O 
        squared_error = np.square(diff)
        squared_error = np.sum(squared_error, axis = 1)
        squared_error = np.sum(squared_error)
        return squared_error/num_pattern
    """
    # Correct Averaging: MSE is the mean of squared errors over all data points and all features.
    def mean_squared_error_loss(self, Y, O):
        diff = Y - O
        squared_error = np.square(diff)
        mse = np.mean(squared_error)  
        return mse

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
            elif  activation[levels] == 'tanh':
                self.activation.append(self.linear)

    # constructor  self, mdlParams, units_for_levels, activation, dbgMode, VariableLROption, eta0=0.8, eta_tau=0.5, tau=100, lambda_reg=0.01, alpha = 0.9
    # units_for_levels is the list of units for each level, the length of the list is the number of levels
    # activation is a list of string with the name of the activation function for all the levels 
    def __init__(self, mdlParams, dbgMode):

        
        self.VariableLROption = mdlParams.VariableLROption
        self.activationListName = mdlParams.activation
        # for variable learning rate
        self.eta0 = mdlParams.eta0 
        self.eta_tau = mdlParams.eta_tau 
        self.tau = mdlParams.tau 

        # Thikonov Regularization 
        self.lambda_reg = mdlParams.lambda_reg

        # momentum param
        self.alpha = mdlParams.alpha

        self.debugMode = dbgMode
        self.units_for_levels = mdlParams.units_for_levels
        self.numberOfLevels = len(mdlParams.units_for_levels)-1

        self.initializeActivationFunctions(mdlParams.activation)

        if mdlParams.validationErrorCheck == True : 
            # TODO: FIX HERE (i need a copy not the pointer)
            self.listOfWeightMatrices = copy.deepcopy(mdlParams.weights)


       
    
    # inputX is an input Matrix 
    # at the end of feedForeward I should have as many elements as number of levels in both listOfHiddenRepr and listOfNet
    # 0,.., level-1 
    def feedForeward (self, inputX, listOfWeights):

        localInputX = inputX
        self.listOfHiddenRepr = []
        self.listOfNet = []

        #level means W matrix for all levels
        for level in range(self.numberOfLevels):
            # compute the net 
            localInputX = np.matmul(localInputX, listOfWeights[level])
            # save the net 
            self.listOfNet.append(localInputX)
            # compute the hidden unit output 
            localInputX = self.activation[level](localInputX)
            self.listOfHiddenRepr.append(localInputX)
 
        # print for debug
        
        if self.debugMode :
            print("Stampa net ")
            self.print_matrices_fancy(self.listOfNet)
            print("Stampa rappresentazione interna")
            self.print_matrices_fancy(self.listOfHiddenRepr)
            # localInputX == output of feedForeward of the NN
         
        return localInputX; 

    # y is the target matrix
    # o is the output matrix 
    def backPropagate(self, x, y, o):
        
        error = y - o 
        if self.debugMode :
            print(f"error: {error}\n")

        net_k = self.listOfNet[self.numberOfLevels-1]
        delta_k = np.zeros((error.shape[0], error.shape[1]))

        for pattern in range(x.shape[0]):
            delta_k[pattern] = error[pattern] * self.activation[self.numberOfLevels-1](net_k, True)[pattern]
            
        if self.debugMode :
            print(f"delta_k: {delta_k}\n")

        listOfDelta = []
        delta_temp = delta_k

        for levels in range(self.numberOfLevels-1,0,-1):
            delta_temp = np.matmul(delta_temp,self.listOfWeightMatrices[levels].T)
            
            if self.debugMode :
                print(f" delta_j temp: {delta_temp}\n")
            
            net_j = self.listOfNet[levels-1]
            
            if self.debugMode :
                print(f"derivata di net_J {self.activation[levels-1](net_j, derivative = True)}\n")
            
            for pattern in range(x.shape[0]):
                delta_temp[pattern] = delta_temp[pattern] * self.activation[levels-1](net_j, derivative = True)[pattern]
            
            listOfDelta.append(delta_temp)
        
        # compute the gradients for each level 
        grad_output = np.matmul(delta_k.T, self.listOfHiddenRepr[self.numberOfLevels-2])

        # normalization widh batch dim
        grad_output = grad_output / x.shape[0]
        grad_hidden = []

        
        listOfDelta.reverse()
        if self.debugMode :
            for delta in listOfDelta:
                print(f"delta{delta}")

        for levels in range(0,self.numberOfLevels-1,1):
            if levels == 0:
                normGradHidden = np.matmul(listOfDelta[levels].T, x).T / x.shape[0] 
                grad_hidden.append(normGradHidden)
            else:
                # forgot / x.shape[0] 
                normGradHidden = np.matmul(listOfDelta[levels].T, self.listOfHiddenRepr[levels-1]).T / x.shape[0] 
                grad_hidden.append(normGradHidden)


        return grad_output.T, grad_hidden


    def train (self, X, Y, epochs=100, treshold=0.1, initMode="random", numberOfRestart=5, validationErrorCheck = False, xValid = None, yValid = None):
        
        if validationErrorCheck :
            return self.selectOptimalStartingWeights(X, Y, epochs, treshold, initMode, numberOfRestart, validationErrorCheck, xValid, yValid)
        else : 
            return self.selectOptimalStartingWeights(X, Y, epochs, treshold, initMode, numberOfRestart)
        
                
    def selectOptimalStartingWeights(self, X, Y, epochs, treshold, initMode, numberOfRestart, validationErrorCheck = False, xValid = None, yValid = None):

        
        self.optimalListOfWeightMatrices = []
        e = float("inf")
        i = 0
        listLog = []
        listLogMatrices = [] 
        optLogVL = []
        listOfLogsTR = []
        while i < numberOfRestart:

            
            if validationErrorCheck :
                etmp, logVL, logTR = self.realTraining(X, Y, epochs, treshold, initMode, validationErrorCheck, xValid, yValid)
            else : 
                etmp, logTR = self.realTraining(X, Y, epochs, treshold, initMode, validationErrorCheck, xValid, yValid)

            # list of lists that contains TRlog for all training
            

            if self.debugMode :
                listLog.append(f" etmp for all iteration::::: {etmp} \n")
                listLogMatrices.append(self.listOfWeightMatrices)

            if etmp < e:
                
                if validationErrorCheck :
                    optLogVL = copy.deepcopy(logVL)
                
                listOfLogsTR = copy.deepcopy(logTR)
                # TODO: FIX HERE (i need a copy not the pointer)
                self.optimalListOfWeightMatrices = copy.deepcopy(self.listOfWeightMatrices)
                # TODO: FIX HERE (i need a copy not the pointer)
                self.initWeights = copy.deepcopy(self.tmpStartWeights)
                e = etmp         
            i += 1    
        
        if self.debugMode :
            for stringLog in listLog:
                print(stringLog)

            print(f" optimal e: {e}\n")
            print(f"liste delle matrici per ogni iterazione\n")
            k = 0
            for matricesLog in listLogMatrices:
                print(f"lista {k}\n")
                self.print_matrices_fancy(matricesLog)
                k += 1

            print(f"lista migliore\n")
            self.print_matrices_fancy(self.optimalListOfWeightMatrices)
        

        #TODO: write on file optLogVL for plot

        if validationErrorCheck : 
            return e, optLogVL, listOfLogsTR
        else :
            return e, listOfLogsTR

    #befoure def realTraining(self, X, Y, epochs, treshold, initMode, file, xValid = None, yValid = None):
    def realTraining(self, X, Y, epochs, treshold, initMode, validationErrorCheck = False, xValid = None, yValid = None):
        
        # initialization of the weight matrix to random small values 
        # we need a list of matrices, one for each layer, 
        # each matrix column represents the weight for a single unit of that level 
        # matrix for level l in position i,j  has the weight from unit j to unit i
        # the matix is m x n where m is the number of input unit and n is the number of the unit of that level
        
        if validationErrorCheck == False : 
            self.listOfWeightMatrices = self.initalizeWeightMatrix(initMode)
        
        self.tmpStartWeights = copy.deepcopy(self.listOfWeightMatrices)

        
 
        # log for training 
        logTR = []  

        # log for validation
        if validationErrorCheck : logVL = []
        
        i = 1
        e = float("inf")

        # init old gradient for momentum 
        oldGrad_hidden = [np.zeros_like(w) for w in self.listOfWeightMatrices[:-1]]
        oldGrad_output = np.zeros_like(self.listOfWeightMatrices[-1])


        while i <= epochs and e > treshold:
            
            # print weight matrices
            if self.debugMode :
                print(f"matrice dei pesi iterazione {i}")
                self.print_matrices_fancy(self.listOfWeightMatrices)
                
            # compute model output 
            o = self.feedForeward(X, self.listOfWeightMatrices)

            # print the error 
            grad_output, grad_hidden = self.backPropagate(X, Y, o)

            # compute VL error for 10 % of iterations
            # and i % (epochs * 0.1) == 0
            if validationErrorCheck   :
                outVal = self.feedForeward(xValid, self.listOfWeightMatrices) 
                
                #for classification
                #eVL = self.classification_error(yValid, outVal, activation="tanh")
                
                #for regretion
                eVL = self.mean_squared_error_loss(yValid, outVal, activation="tanh")
                
                logVL.append(f"Epoch : {i}, MSE : {eVL}\n")
                



            e = self.mean_squared_error_loss(Y, o)

            # TODO : salavre l'indice dell iterazione ed errore e plottare
            
            logTR.append(f"Epoch : {i}, MSE : {e}\n")


            # change learning rate
            # if variable learning rate is enable
            if self.VariableLROption :
                if i <= self.tau :
                    etas = self.learning_rate_schedule(self.eta0, self.eta_tau, self.tau, i)
                else : 
                    etas = self.eta_tau
            else:
                etas = self.eta0
            
            for j in range(0, len(grad_hidden), 1):
                
                # compute the momentum contribution for the hidden gradient update rule 
                velocityHidden = self.alpha * oldGrad_hidden[j] + ((etas) * grad_hidden[j]) 
                # compute penalty term for regularization
                penalty_term = self.lambda_reg * self.listOfWeightMatrices[j]
                self.listOfWeightMatrices[j] = self.listOfWeightMatrices[j] + velocityHidden  - penalty_term
                # save old momentum contribution for next iteration
                oldGrad_hidden[j] = velocityHidden
            

            # compute the momentum contribution for the output gradient update rule
            velocityOutput = self.alpha * oldGrad_output + ((etas) * grad_output)
            # compute penalty term for regularization 
            penalty_term = self.lambda_reg * self.listOfWeightMatrices[-1]
            # list[-1] = last elem of the list = weights between hidden and output
            self.listOfWeightMatrices[-1] = self.listOfWeightMatrices[-1] + velocityOutput   - penalty_term 
            
            # save old momentum contribution for next iteration
            oldGrad_output = velocityOutput 

            i += 1
    
        """
        for string in logTR:
            file.write(string)

        """
        
        #scrivere il class err in un file per poi plottare     
        if validationErrorCheck :
            return e, logVL, logTR
        else : 
            return e, logTR

    @staticmethod
    def classification_error(Y, o, activation="tanh"):

        num_pattern = Y.shape[0]
        if activation == "sigmoid":     
            o = (o >= 0.5).astype(int)  # Converte in 0 o 1
        elif activation == "tanh": 
            o[o >= 0] = 1
            o[o < 0] = -1
        
        err = (Y != o).all(axis=1).astype(int)
        return np.sum(err)/num_pattern
    

    # added by Matteo Torchia to change learning rate
    def learning_rate_schedule(self, eta0, eta_tau, tau, step): 
        gamma = step / tau 
        eta_s = (1 - gamma) * eta0 + gamma * eta_tau
        return eta_s
    # end

    
    def predict(self, inputX, reloaded, optimalListOfWeightMatrices):
        
        #if reloaded == true then use optimalListOfWeightMatrices retreived from disk
        if reloaded :
            return self.feedForeward(inputX, optimalListOfWeightMatrices)
        
        #else use optimalListOfWeightMatrices in model just trained
        else :
            return self.feedForeward(inputX, self.optimalListOfWeightMatrices)
    

    def predict_class(self, inputX, reloaded, activation="tanh", optimalListOfWeightMatrices=None):

        o = self.predict(inputX, reloaded, optimalListOfWeightMatrices)

        if activation == "tanh":
            o[o >= 0] = 1
            o[o < 0] = -1    
        elif activation == "sigmoid":
            o = (o >= 0.5).astype(int)  # Converte in 0 o 1
        return o

    def get_list_weight_matrices(self):
        return self.listOfWeightMatrices
    
    def get_list_init_weight_matrices(self):
        return self.initWeights

    # function to retreive all parameters from model
    def getParameters(self):
        return (self.optimalListOfWeightMatrices, self.units_for_levels, self.activationListName, self.eta0, self.eta_tau, self.tau, self.lambda_reg, self.alpha)

    def getOptimalWeights(self):
        return self.optimalListOfWeightMatrices

    # function to store all parameters on disk
    @staticmethod
    def saveModel(mymodelparameters, pathFile="./finalModel.txt"):
        

        with open(pathFile,  mode = 'wb') as fileModel: 
            
            modelAsString = pickle.dumps(mymodelparameters)
            fileModel.write(modelAsString)


    # function to read parameteres from disk
    @staticmethod
    def realoadModel(pathFile="./finalModel.txt"):

        with open(pathFile,  mode = 'rb') as fileModel:

            modelAsString = fileModel.read()
            mymodelparameters = pickle.loads(modelAsString)

        return mymodelparameters
   





