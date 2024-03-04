import numpy as np
from utils import MAPE
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def random_forest_regression(config,train_x, train_y, test_x, test_y,test_y_norm=None, diff=False):
    model = RandomForestRegressor(n_estimators=config.rf_estimators,
                                  max_depth=config.rf_max_depth,
                                  min_samples_split=config.rf_min_samples_split,
                                  min_samples_leaf=config.rf_min_samples_leaf,
                                  random_state=config.seed ,verbose=False)
    
    train_x = train_x.to_numpy().reshape(-1,1)
    train_y = train_y.to_numpy().reshape(-1,1)
    model.fit(train_x,train_y)
    test_x = test_x.to_numpy().reshape(-1,1)
    pred_y = model.predict(test_x)
    if diff:
        pred_y = np.exp(np.cumsum(pred_y)+test_y_norm.iloc[-1])
        test_y = np.exp(np.cumsum(test_y)+test_y_norm.iloc[-1])
    else:
        pred_y = np.exp(pred_y)
        test_y = np.exp(test_y.to_numpy().reshape(-1,1))   

    score = MAPE(test_y, pred_y,log_linear=True )
    
    return(pred_y, score)
    
def xgboost_regression(config,train_x, train_y, test_x, test_y,test_y_norm=None, diff=False):
    model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=200,
                           objective='reg:squarederror',
                           max_depth=config.xg_max_depth,
                           learning_rate=config.xg_learning_rate,
                           reg_lambda=config.xg_reg_lambda,
                           reg_alpha=config.xg_reg_alpha
                           )                        
    train_x = train_x.to_numpy().reshape(-1,1)
    train_y = train_y.to_numpy().reshape(-1,1)
    model.fit(train_x, train_y,
            verbose=False)
    test_x = test_x.to_numpy().reshape(-1,1)
    pred_y = model.predict(test_x)

    if diff:
        pred_y = np.exp(np.cumsum(pred_y)+test_y_norm.iloc[-1])
        test_y = np.exp(np.cumsum(test_y)+test_y_norm.iloc[-1])
    else:
        pred_y = np.exp(pred_y)
        test_y = np.exp(test_y.to_numpy().reshape(-1,1))
    
    score = MAPE(test_y, pred_y,log_linear=True)
    
    
    return(pred_y, score)    
        
