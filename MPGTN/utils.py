import pickle

import jpholiday
import numpy as np
import pandas as pd
import torch


def mape(y_pred, y):
    return torch.mean(torch.abs((y_pred - y) / y))


def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

# DCRNN
def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters.\n')
    return

def get_data(data_path, N_link, subdata_path, feature_list):
    data = pd.read_csv(data_path)[feature_list].values
    data = data.reshape(-1, N_link, data.shape[-1])
    data[data<0] = 0
    data[data>200.0] = 100.0
    sub_idx = np.loadtxt(subdata_path).astype(int)
    data = data[:, sub_idx, :]
    return data

def get_time(data_path, N_link, subdata_path):
    df = pd.read_csv(data_path)[['timestamp']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['weekdaytime'] = df['timestamp'].dt.weekday * 144 + (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute)//10
    df['weekdaytime'] = df['weekdaytime'] / df['weekdaytime'].max()
    data = df[['weekdaytime']].values
    data = data.reshape(-1, N_link, data.shape[-1])
    sub_idx = np.loadtxt(subdata_path).astype(int)
    data = data[:, sub_idx, :]
    return data

def get_timeinday(data_path, N_link, subdata_path):
    df = pd.read_csv(data_path)[['timestamp']]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timeinday'] = (df.timestamp - df.timestamp.astype("datetime64[D]")) / np.timedelta64(1, "D")
    data = df[['timeinday']].values
    data = data.reshape(-1, N_link, data.shape[-1])
    sub_idx = np.loadtxt(subdata_path).astype(int)
    data = data[:, sub_idx, :]
    return data

def get_adj(adj_path, subroad_path):
    A = np.load(adj_path)
    if subroad_path is not None:
        sub_idx = np.loadtxt(subroad_path).astype(int)
        A = A[sub_idx, :][:, sub_idx]
    return A

def get_seq_data(data, seq_len):
    seq_data = [data[i:i+seq_len, ...] for i in range(0, data.shape[0]-seq_len+1)]
    return np.array(seq_data)

def getXSYS_single(data_list, his_len, seq_len):
    XS, YS = [], []
    for data in data_list:
        seq_data = get_seq_data(data, seq_len + his_len)
        XS_, YS_ = seq_data[:, :his_len, ...], seq_data[:, -seq_len:-seq_len+1, ...]
        XS.append(XS_)
        YS.append(YS_)
    XS, YS = np.vstack(XS), np.vstack(YS)    
    return XS, YS

def getXSYS(data_list, his_len, seq_len):
    XS, YS = [], []
    for data in data_list:
        seq_data = get_seq_data(data, seq_len + his_len)
        XS_, YS_ = seq_data[:, :his_len, ...], seq_data[:, -seq_len:, ...]
        XS.append(XS_)
        YS.append(YS_)
    XS, YS = np.vstack(XS), np.vstack(YS)
    return XS, YS

def get_onehottime(data_path):
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    df = pd.DataFrame({'time':data['timestamp'].unique()})  
    df['dayofweek'] = df.time.dt.weekday
    df['hourofday'] = df.time.dt.hour
    df['intervalofhour'] = df.time.dt.minute//10 
    df['isholiday'] = df.apply(lambda x: int(jpholiday.is_holiday(x.time) | (x.dayofweek==5) | (x.dayofweek==6)), axis=1)
    tmp1 = pd.get_dummies(df.dayofweek)
    tmp2 = pd.get_dummies(df.hourofday)
    tmp3 = pd.get_dummies(df.intervalofhour)
    tmp4 = df[['isholiday']]
    df_dummy = pd.concat([tmp1, tmp2, tmp3, tmp4], axis=1)
    return df_dummy.values


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()




def masked_mse(preds, labels, null_val=1e-3):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=1e-3):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=1e-3):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=1e-3):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

# DCRNN
def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mape_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(torch.div(y_true - y_pred, y_true))
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_rmse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())

def masked_mse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_true - y_pred, 2)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()