import argparse
import random
import sys

import numpy as np
import torch

from exp.exp_main import Exp_Main
from utils.metrics import exists_metrics

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')

# TINSRDLinear_SONODE_TRaug
parser.add_argument('--model', type=str, default='LTSF_DNODE_v1',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader

parser.add_argument('--model_id', type=str, default='ili_104_24', help='model id')
parser.add_argument('--data', type=str, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='national_illness.csv', help='data file')
# forecasting task
parser.add_argument('--seq_len', type=int, default=104, help='input sequence length')
parser.add_argument('--label_len', type=int, default=18, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='m',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# DLinear
parser.add_argument('--individual', action='store_true', default=False,
                    help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
parser.add_argument('--embed_type', type=int, default=0,
                    help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')

# national_illness: 7, exchange_rate: 8, ETTh1: 7, ETTh2: 7, ETTm1: 7, ETTm2: 7, electricity: 321, traffic: 862, weather: 21
parser.add_argument('--enc_in', type=int, default=7,
                    help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')

parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.05, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
# parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='1,2', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# ODE
parser.add_argument('--bias', action='store_true', help='', default=False)
parser.add_argument('--ode_solver', type=str, default='euler', help='ODE solver')  # DOP853 RK45
parser.add_argument('--ode_step_size', type=float, default=0.1, help='ODE step size')
parser.add_argument('--ode_step_size_T', type=float, default=0.1, help='Trend ODE step size')
parser.add_argument('--ode_step_size_S', type=float, default=0.1, help='Seasonality ODE step size')
parser.add_argument('--ode_step_size_R', type=float, default=0.1, help='Residual ODE step size')
parser.add_argument('--rtol_T', type=float, default=1e-3, help='ODE rtol for trend')
parser.add_argument('--atol_T', type=float, default=1e-5, help='ODE atol for trend')
parser.add_argument('--rtol_S', type=float, default=1e-7, help='ODE rtol for seasonal')
parser.add_argument('--atol_S', type=float, default=1e-9, help='ODE atol for seasonal')
parser.add_argument('--rtol_R', type=float, default=1e-7, help='ODE rtol for residual')
parser.add_argument('--atol_R', type=float, default=1e-9, help='ODE atol for residual')

# decomposition
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--num_fourier_decomp', type=int, default=4, help='gpu')

# loss
# https://github.com/Maghoumi/pytorch-softdtw-cuda
parser.add_argument('--use_sdtw_loss', action='store_true', help='', default=False)

# transformation(preprocessing)
parser.add_argument('--tsr_freq', type=str, default='D', help='for TSRDLinear')
parser.add_argument('--tsr_period', type=int, default=7, help='for TSRDLinear')
parser.add_argument('--use_trend_norm', action='store_true', help='', default=False)
parser.add_argument('--use_resid_norm', action='store_true', help='', default=False)
parser.add_argument('--sep_seasonality', action='store_true', help='', default=False)
parser.add_argument('--num_resid_sampling', type=int, default=100, help='multiple residual sampling')
parser.add_argument('--num_ode_func_layers', type=int, default=3, help='number of layers in ODE function')
parser.add_argument('--num_enc_layers', type=int, default=1, help='number of layers in encoder')
parser.add_argument('--num_dec_layers', type=int, default=1, help='number of layers in decoder')
parser.add_argument('--act_func_ode_func', type=str, help='', default='ReLU')
parser.add_argument('--act_func_enc', type=str, help='', default=None)
parser.add_argument('--act_func_dec', type=str, help='', default=None)

parser.add_argument('--ode_hidden_size_T', type=int, default=32)
parser.add_argument('--ode_hidden_size_S', type=int, default=16)
parser.add_argument('--ode_hidden_size_R', type=int, default=16)

# kinetic
parser.add_argument('--kinetic', type=float, default=1., help='kinetic')
parser.add_argument('--jacobian', type=float, default=1., help='jacobian')

# sde
parser.add_argument('--sde_solver', type=str, default='reversible_heun')

# etc
parser.add_argument('--metric_save_path', type=str, default='./results/results_default.csv', help='')
parser.add_argument('--loss_save_path', type=str, default='./results/results_loss_default.csv', help='')
parser.add_argument('--save_results', action='store_true', help='', default=False)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

# check existence of experiments results
metric_save_path = args.metric_save_path
arg_names = ['model_id', 'model', 'data',
             'kinetic', 'jacobian',
             'num_enc_layers', 'act_func_enc',
             'num_ode_func_layers', 'act_func_ode_func',
             'ode_hidden_size_T', 'ode_hidden_size_S', 'ode_hidden_size_R',
             'num_dec_layers', 'act_func_dec',
             'ode_solver',
             'ode_step_size_trend', 'ode_step_size_seasonal', 'ode_step_size_resid',
             'rtol_trend', 'atol_trend',
             'rtol_seasonal', 'atol_seasonal',
             'rtol_resid', 'atol_resid',
             'features', 'batch_size', 'learning_rate',
             'lradj',
             'seq_len', 'pred_len',
             'train_epochs', 'patience',
             'moving_avg', 'tsr_period',
             'des']

# if exists_metrics(metric_save_path, args, arg_names):
#     print(f'There exist experiments results! - {args}')
#     sys.exit()

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}_{}_bi{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
            args.batch_size,
            args.learning_rate,
            args.bias)

        args.setting = setting

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

        # exp.train(setting)
        if args.model not in ['Linear_ODE_solver', 'DLinear_ODE_solver', 'NLinear_ODE_solver', 'Linear_ODE_solver2']:
            exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, args=args)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
