import torch
import torch.nn as nn
import numpy as np
from models.TorchDiffEqPack_src.TorchDiffEqPack import odesolve as odeint
from kinetic.kinetic_wrapper_class import KineticWrapper

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        if self.kernel_size % 2 == 0:
            front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        else:
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class ODEFunc(nn.Module):

    def __init__(self, in_seq, out_seq, n_layers=1):
        super(ODEFunc, self).__init__()
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.net = nn.Linear(self.in_seq, self.out_seq)

        self.net = nn.ModuleList(nn.Linear(self.in_seq, self.in_seq) for _ in range(n_layers - 1))
        self.net.append(nn.Linear(self.in_seq, self.out_seq))

    def forward(self, t, x):
        x = x.permute(0, 2, 1)

        # y = self.net(x)
        for layer in self.net:
            x = layer(x)
        y = x

        y = y.permute(0, 2, 1)

        return y


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.batch_size = configs.batch_size
        self.rtol_T = configs.rtol_T
        self.atol_T = configs.atol_T
        self.rtol_S = configs.rtol_S
        self.atol_S = configs.atol_S
        self.rtol_R = configs.rtol_R
        self.atol_R = configs.atol_R
        self.ode_solver = configs.ode_solver
        # self.ode_step_size = configs.ode_step_size
        self.ode_step_size_T = configs.ode_step_size_T
        self.ode_step_size_S = configs.ode_step_size_S
        self.ode_step_size_R = configs.ode_step_size_R

        self.kernel_size = configs.moving_avg
        self.period = configs.tsr_period
        self.use_trend_norm = configs.use_trend_norm
        self.use_resid_norm = configs.use_resid_norm
        self.sep_seasonality = configs.sep_seasonality

        self.kinetic = configs.kinetic
        self.jacobian = configs.jacobian

        print(f'use_trend_norm: {self.use_trend_norm}, use_resid_norm: {self.use_resid_norm}, sep_seasonality: {self.sep_seasonality}')

        self.series_decomp = series_decomp(self.kernel_size)

        self.num_ode_func_layers = configs.num_ode_func_layers
        self.ode_func_T = ODEFunc(self.seq_len, self.seq_len, self.num_ode_func_layers)
        self.ode_func_S = ODEFunc(self.seq_len, self.seq_len, self.num_ode_func_layers)
        self.ode_func_R = ODEFunc(self.seq_len, self.seq_len, self.num_ode_func_layers)

        self.ode_func_T = KineticWrapper(self.ode_func_T, self.kinetic, self.jacobian, 1)
        self.ode_func_S = KineticWrapper(self.ode_func_S, self.kinetic, self.jacobian, 1)
        self.ode_func_R = KineticWrapper(self.ode_func_R, self.kinetic, self.jacobian, 1)

        self.readout_T = nn.Linear(self.seq_len, self.pred_len)
        self.readout_S = nn.Linear(self.seq_len, self.pred_len)
        self.readout_R = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, t):
        # x: [Batch, Input length, Channel]
        device = x.get_device()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.Tensor([0, 1])
        # print(x.shape)

        options_trend = {}
        options_trend.update({'method': self.ode_solver})
        options_trend.update({'h': self.ode_step_size_T})
        options_trend.update({'t0': 0.0})
        options_trend.update({'t1': 1.0})
        options_trend.update({'rtol': self.rtol_T})
        options_trend.update({'atol': self.atol_T})
        options_trend.update({'print_neval': False})  # nfe counts
        options_trend.update({'neval_max': 1000000})
        options_resid = {}
        options_resid.update({'method': self.ode_solver})
        options_resid.update({'h': self.ode_step_size_R})
        options_resid.update({'t0': 0.0})
        options_resid.update({'t1': 1.0})  # when t_eval is None(default), odesolve only returns values at t1
        options_resid.update({'rtol': self.rtol_R})
        options_resid.update({'atol': self.atol_R})
        options_resid.update({'print_neval': False})  # nfe counts
        options_resid.update({'neval_max': 1000000})

        trend_init, seasonal_init, resid_init = self.tsr_decomp(x, device)
        # trend_init = (trend_init,torch.zeros(trend_init.size(0)).to(trend_init), torch.zeros(trend_init.size(0)).to(trend_init))
        # resid_init = (resid_init,torch.zeros(resid_init.size(0)).to(resid_init), torch.zeros(resid_init.size(0)).to(resid_init))

        # normal (normal o/x)
        if self.use_trend_norm:
            mean_trend, stdev_trend = self._get_statistics(trend_init)
            trend_init = self._normalize(trend_init, mean_trend, stdev_trend)
            trend_init = (trend_init,torch.zeros(trend_init.size(0)).to(trend_init), torch.zeros(trend_init.size(0)).to(trend_init))
            y_trend, y_trend_kinetic, y_trend_jacobian = odeint(func=self.ode_func_T, y0=trend_init, options=options_trend)
            y_out = self.readout_T(y_trend.permute(0, 2, 1)).permute(0, 2, 1)
            y_out = self._denormalize(y_out, mean_trend, stdev_trend)
            reg = self.kinetic * y_trend_kinetic.mean() + self.jacobian * y_trend_jacobian.mean()
        else:
            trend_init = (trend_init,torch.zeros(trend_init.size(0)).to(trend_init), torch.zeros(trend_init.size(0)).to(trend_init))
            y_trend, y_trend_kinetic, y_trend_jacobian = odeint(func=self.ode_func_T, y0=trend_init, options=options_trend)
            y_out = self.readout_T(y_trend.permute(0, 2, 1)).permute(0, 2, 1)
            reg = self.kinetic * y_trend_kinetic.mean() + self.jacobian * y_trend_jacobian.mean()

        # Seasonality
        if self.sep_seasonality == True:
            options_seasonal = {}
            options_seasonal.update({'method': self.ode_solver})
            options_seasonal.update({'h': self.ode_step_size_S})
            options_seasonal.update({'t0': 0.0})
            options_seasonal.update({'t1': 1.0})  # when t_eval is None(default), odesolve only returns values at t1
            options_seasonal.update({'rtol': self.rtol_S})
            options_seasonal.update({'atol': self.atol_S})
            options_seasonal.update({'print_neval': False})  # nfe counts
            options_seasonal.update({'neval_max': 1000000})

            seasonal_init = (seasonal_init,torch.zeros(seasonal_init.size(0)).to(seasonal_init), torch.zeros(seasonal_init.size(0)).to(seasonal_init))
            y_seasonal, y_seasonal_kinetic, y_seasonal_jacobian = odeint(func=self.ode_func_S, y0=seasonal_init, options=options_seasonal)
            seasonal_output = self.readout_S(y_seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            reg_seasonal = self.kinetic * y_seasonal_kinetic.mean() + self.jacobian * y_seasonal_jacobian.mean()

            y_out = y_out + seasonal_output
            reg = reg + reg_seasonal

        # resid
        if self.use_resid_norm:
            mean_resid, stdev_resid = self._get_statistics(resid_init)
            resid_init = self._normalize(resid_init, mean_resid, stdev_resid)
            resid_init = (resid_init,torch.zeros(resid_init.size(0)).to(resid_init), torch.zeros(resid_init.size(0)).to(resid_init))
            y_resid, y_resid_kinetic, y_resid_jacobian= odeint(func=self.ode_func_R, y0=resid_init, options=options_resid)
            resid_output = self.readout_R(y_resid.permute(0, 2, 1)).permute(0, 2, 1)
            resid_output = self._denormalize(resid_output, mean_resid, stdev_resid)
            reg_resid = self.kinetic * y_resid_kinetic.mean() + self.jacobian * y_resid_jacobian.mean()
        else:
            resid_init = (resid_init,torch.zeros(resid_init.size(0)).to(resid_init), torch.zeros(resid_init.size(0)).to(resid_init))
            y_resid, y_resid_kinetic, y_resid_jacobian= odeint(func=self.ode_func_R, y0=resid_init, options=options_resid)
            resid_output = self.readout_R(y_resid.permute(0, 2, 1)).permute(0, 2, 1)
            reg_resid = self.kinetic * y_resid_kinetic.mean() + self.jacobian * y_resid_jacobian.mean()

        # if kinetic_energy_coef is not None or jacobian_norm2_coef is not None:
        #     reg = self.kinetic * real_output[2].mean() + self.jacobian * real_output[3].mean()
        #     return real_output[:2], reg
        # else:
        #     return real_output, 0

        y_T = y_out + resid_output
        reg_T = reg + reg_resid

        return y_T, reg_T  # to [Batch, Output length, Channel]

    def tsr_decomp(self, x, device):

        res, trend_init = self.series_decomp(x)
        if self.sep_seasonality == False:
            return trend_init, None, res

        seasonal = torch.Tensor().to(device)
        for j in range(self.period):
            period_average = torch.unsqueeze(torch.mean(res[:, j::self.period, :], axis=1), dim=1)
            seasonal = torch.concat([seasonal, period_average], dim=1)
        seasonal = seasonal - torch.unsqueeze(torch.mean(seasonal, dim=1), dim=1)
        seasonal_init = torch.tile(seasonal.T, (1, x.shape[1] // self.period + 1, 1)).T[:, :x.shape[1], :]
        resid_init = res - seasonal_init

        return trend_init, seasonal_init, resid_init

    def _get_statistics(self, x, eps=1e-5):
        dim2reduce = tuple(range(1, x.ndim - 1))

        mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + eps).detach()
        return mean, stdev

    def _normalize(self, x, mean, stdev):
        x = x - mean
        x = x / stdev

        return x

    def _denormalize(self, x, mean, stdev):
        x = x * stdev
        x = x + mean

        return x
