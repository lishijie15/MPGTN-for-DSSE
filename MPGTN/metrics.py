import numpy as np
from momentum import kurtosis_init, kurtosis_update

def evaluate(y_true, y_pred, precision=10):
    return MSE(y_true, y_pred), RMSE(y_true, y_pred), MAE(y_true, y_pred), MAPE(y_true, y_pred), skewness(y_true, y_pred), kurtosis(y_true, y_pred)

def MSE(y_true, y_pred):
    y_true[y_true < 1e-5] = 0
    y_pred[y_pred < 1e-5] = 0
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse
    
def RMSE(y_true, y_pred):
    y_true[y_true < 1e-5] = 0
    y_pred[y_pred < 1e-5] = 0
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse
        
def MAE(y_true, y_pred):
    y_true[y_true < 1e-5] = 0
    y_pred[y_pred < 1e-5] = 0
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae

def MAPE(y_true, y_pred, null_val=0):
    y_true[y_true < 1e-5] = 0
    y_pred[y_pred < 1e-5] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype('float32'), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def skewness(y_true, y_pred):
    y_true[y_true < 1e-5] = 0
    y_pred[y_pred < 1e-5] = 0
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        middle = y_pred - y_true
        n = middle.shape[0]
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        middle = np.nan_to_num(middle * mask)
        f_mean = np.mean(middle)
        f_std = np.std(middle)
        skew = (middle-f_mean)
        skew = np.divide(skew, f_std)
        skew = np.power(skew, 3)
        skew = np.mean(skew)
        return skew

def kurtosis(y_true, y_pred):
    y_true[y_true < 1e-5] = 0
    y_pred[y_pred < 1e-5] = 0
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        middle = y_pred - y_true
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        middle = np.nan_to_num(middle * mask)
        f_mean = np.mean(middle)
        f_std = np.std(middle)
        kurt = (middle-f_mean)
        kurt = np.divide(kurt, f_std)
        kurt = np.power(kurt, 4)
        kurt = np.mean(kurt)-3
        return kurt


