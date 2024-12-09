class myModelParameters:

    def __init__(self, weights, units_for_levels, activation, eta0=0.8, eta_tau=0.5, tau=100, lambda_reg=0.01, alpha = 0.9):
            
        self.weights = weights
        self.units_for_levels = units_for_levels
        self.activation = activation
        self.eta0 = eta0
        self.eta_tau = eta_tau
        self.tau = tau
        self.lambda_reg = lambda_reg
        self.alpha = alpha