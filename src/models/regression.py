import numpy as np
from sklearn.linear_model import LinearRegression
from utils import MAPE


def log_linear_regression(fit_y, test_y, config):
    num_regions = len(config.regions)
    Predictions = np.zeros((num_regions,config.size_PW))   
    scores_reg = np.zeros((num_regions))
    for j in range(0,num_regions):
        
        X_fit = np.arange(1,config.size_SW+1).reshape(-1, 1)
        Y_fit = fit_y[j]


        X_test = np.arange(config.size_SW+1,config.size_SW+1+config.size_PW).reshape(-1, 1)
        Y_test = test_y[j]
        model=LinearRegression() 
        model.fit(X_fit,Y_fit)
        Y_pred = model.predict(X_test)

        score_reg = MAPE(np.exp(Y_test),np.exp(Y_pred), log_linear=True)
        
        Predictions[j,:] = Y_pred[:]
        scores_reg [j] = score_reg

    return(np.exp(Predictions), scores_reg)