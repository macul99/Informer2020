import numpy as np
# all inputs are numpy array

def RSE(pred:np.ndarray, true:np.ndarray):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred:np.ndarray, true:np.ndarray):
    '''
    Calculate the corrcoeff for each column pair of pred and true, then calculate mean of all corrcoeff
    '''
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    #d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0)) # this is the original implementation, seems wrong
    d = np.sqrt((((true-true.mean(0))**2).sum(0)*((pred-pred.mean(0))**2).sum(0)))
    return (u/d).mean(-1) #calculate mean for corrcoef of each pair

def MAE(pred:np.ndarray, true:np.ndarray):
    return np.mean(np.abs(pred-true))

def MSE(pred:np.ndarray, true:np.ndarray):
    return np.mean((pred-true)**2)

def RMSE(pred:np.ndarray, true:np.ndarray):
    return np.sqrt(MSE(pred, true))

def MAPE(pred:np.ndarray, true:np.ndarray):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred:np.ndarray, true:np.ndarray):
    return np.mean(np.square((pred - true) / true))

def metric(pred:np.ndarray, true:np.ndarray):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe