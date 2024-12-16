from model import NeuralNetwork
from sklearn.model_selection import KFold

class myModelParameters:




    def __init__(self, weights, units_for_levels, activation, VariableLROption = False, eta0=0.8, eta_tau=0.5, tau=100, lambda_reg=0.01, alpha = 0.9, validationErrorCheck = False):
        
        self.validationErrorCheck = validationErrorCheck
        self.VariableLROption = VariableLROption
        self.weights = weights
        self.units_for_levels = units_for_levels
        self.activation = activation
        self.eta0 = eta0
        self.eta_tau = eta_tau
        self.tau = tau
        self.lambda_reg = lambda_reg
        self.alpha = alpha
    



    # function to split data set in k-fold
    @staticmethod
    def kFoldPartition(trainingForSelection, k):


        kf = KFold(n_splits=k, shuffle=False)
        splits = []
        for trainIDX, validIDX in kf.split(trainingForSelection):
            splits.append((trainingForSelection[trainIDX] , trainingForSelection[validIDX]))    
        
        return splits 


    @staticmethod
    def doGridSearch(xTrain, xValid, yTrain, yValid, units_for_levels, activation, debugMode):

        resultOptIperParam = {}

        optimalKeys = (None, None, None, None)
        optimalValue = (None, float("inf"), None)

        """
        rangeEta = [0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rangeLambda = [0.0001, 0.001, 0.01, 0.1, 0,2]
        rangeAlpha = [0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rangeEpochs = [500, 600, 700, 800, 900, 1000]
        """ 
        """
        rangeEta = [0.8, 0.9]
        rangeLambda = [0.0001, 0.001]
        rangeAlpha = [0.8, 0.9]
        rangeEpochs = [10, 20]
        """
    
        rangeEta0 = [0.1]
        rangeLambda = [0.1, 0.01, 0.001]
        rangeAlpha = [0.1, 0.5, 0.9]
        rangeEpochs = [100, 200, 300, 500]
        rangeEtaFinal = [0.01, 0.1, 0.001]

        optLogsTR = []
        startWeightsForOptimalTraining = []

        optModel = None
        # start with learning rate
        for idxEta0, eta0 in enumerate(rangeEta0):
            for idxetaF, etaFinal in enumerate(rangeEtaFinal):
                for idxLambda , Lambda in enumerate(rangeLambda):
                    for idxAlpha, Alpha in enumerate(rangeAlpha):
                        for idxepochs, epochs in enumerate(rangeEpochs):

                            print(f"idxs combination: idxEta0 : {idxEta0} , idxetaF : {idxetaF} , idxLambda : {idxLambda} , idxAlpha : {idxAlpha} , idxepochs : {idxepochs}")
                            
                            prm = myModelParameters(None, units_for_levels, activation, True, eta0, etaFinal, epochs / 3 , Lambda, Alpha)
                            model = NeuralNetwork(prm, debugMode)
                            

                            trainError, LogsTR = model.train(xTrain, yTrain, epochs, 0.0001, "random", 5, False, xValid, yValid)

                            #for classification 
                            #result = model.predict_class(xValid, False)
                            
                            #for regretion
                            try :
                                result = model.predict(xValid, False, None)

                                #for classification 
                                #valError = model.classification_error(yValid, result)
                                
                                #for regretion
                                valError = model.mean_squared_error_loss(yValid, result)
                                

                                if debugMode :
                                    #for classification
                                    #print(f"classification error on VL set: {valError}")
                                    
                                    #for regression
                                    print(f"MS error on VL set: {valError}")
                                
                                optWeights = model.getOptimalWeights()
                                resultOptIperParam[(eta0, etaFinal, Lambda, Alpha, epochs)] = (trainError, valError, optWeights, model.get_list_init_weight_matrices())

                                if valError < optimalValue[1] :
                                    optimalValue = (trainError, valError, optWeights)
                                    optimalKeys = (eta0, etaFinal, Lambda, Alpha, epochs)
                                    optModel = model
                                    optLogsTR = LogsTR
                                    startWeightsForOptimalTraining = optModel.get_list_init_weight_matrices()
                            except ValueError :
                                print(ValueError)
                                print(f" bad combination: {eta0}, {etaFinal}, {Lambda}, {Alpha}, {epochs}\n")


                            




        # retraining model with best hiperparameters
        """
        
        modelToBuildValidationError = NeuralNetwork(myModelParameters(startWeightsForOptimalTraining, units_for_levels, activation, True, optimalKeys[0], 0.5, 100, optimalKeys[1], optimalKeys[2], True), debugMode)
        trainError, logVL, LogsTR = modelToBuildValidationError.train(xTrain, yTrain, 1000, 0.0001, "random", 1, True, xValid, yValid)

        result = modelToBuildValidationError.predict_class(xValid, False)
        valError = modelToBuildValidationError.classification_error(yValid, result)
        
        print(f"****************************************************************\n")
        print(f"****************************************************************\n")
        print(f"valError :\n")
        print(f"{valError}\n")
        


        print(f"****************************************************************\n")
        print(f"****************************************************************\n")


        print(f"logVL:\n")
        for kk in logVL : 
            print(f" {kk}")
        
        print(f"****************************************************************\n")
        print(f"****************************************************************\n")
        print(f"****************************************************************\n")
        print(f"****************************************************************\n")
        print(f"****************************************************************\n")

        print(f"LogsTR:\n")
        for kk in LogsTR : 
            print(f" {kk}")



        return optModel, resultOptIperParam, optimalKeys, optimalValue, LogsTR, logVL
        """
        return resultOptIperParam