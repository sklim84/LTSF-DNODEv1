import numpy as np
import os
import pandas as pd

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


def exists_metrics(save_path, args, arg_names):
    if not os.path.exists(save_path):
        return False

    df_results = pd.read_csv(save_path)
    # nan -> None
    df_results = df_results.replace({np.nan: None})
    for index, result in df_results.iterrows():
        existence_flag = True
        for arg_name in arg_names:
            # print(f'arg name: {arg_name}, {result[arg_name]}, {vars(args).get(arg_name)}')
            if result[arg_name] != vars(args).get(arg_name):
                existence_flag = False
                break

        if existence_flag == True:
            break

    return existence_flag