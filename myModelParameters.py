from model import NeuralNetwork
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

        rangeEta = [0.8]
        rangeLambda = [0.01]
        rangeAlpha = [0.9]
        rangeEpochs = [500]



        optLogsTR = []
        startWeightsForOptimalTraining = []

        optModel = None
        # start with learning rate
        for eta in rangeEta:
            for Lambda in rangeLambda:
                for Alpha in rangeAlpha:
                    for epochs in rangeEpochs:

                        prm = myModelParameters(None, units_for_levels, activation, True, eta, 0.5, 100, Lambda, Alpha)
                        model = NeuralNetwork(prm, debugMode)
                        
                        trainError, LogsTR = model.train(xTrain, yTrain, epochs, 0.0001, "random", 1, False, xValid, yValid)

                        result = model.predict_class(xValid, False)
                        valError = model.classification_error(yValid, result)
                        

                        if debugMode :
                            print(f"classification error on VL set: {valError}")
                        
                        optWeights = model.getOptimalWeights()
                        resultOptIperParam[(eta, Lambda, Alpha, epochs)] = (trainError, valError, optWeights)

                        if valError < optimalValue[1] :
                            optimalValue = (trainError, valError, optWeights)
                            optimalKeys = (eta, Lambda, Alpha, epochs)
                            optModel = model
                            optLogsTR = LogsTR
                            startWeightsForOptimalTraining = optModel.get_list_init_weight_matrices()


        # retraining model with best hiperparameters
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