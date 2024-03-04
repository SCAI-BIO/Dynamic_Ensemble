import pmdarima as pm
from utils import MAPE
import numpy as np

def arima_prediction(config, train_data, test_data):
    model = pm.auto_arima(train_data, start_p=1, start_q=1,       
                      max_p=14, max_q=14, 
                      m=1,             
                      d=1,           
                      seasonal=config.seasonality, 
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    forecast, conf_int = model.predict(n_periods=config.size_PW, return_conf_int=True)

    
    score = MAPE(np.exp(test_data), np.exp(forecast),log_linear=True)

    return(np.exp(forecast), score)