import torch
import torch.nn as nn

from kinetic.kinetic_wrapper_class import KineticWrapper
from RevIN.RevIN import RevIN
from models.TorchDiffEqPack_src.TorchDiffEqPack import odesolve as odeint
from models.torchsde._core.adjoint import sdeint_adjoint as sdeint


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


class Encoder(nn.Module):
    """
        act_func : 'ReLU', 'Tanh', ...
    """

    def __init__(self, in_seq, out_seq, n_layers=1, act_func=None):
        super(Encoder, self).__init__()
        self.in_seq = in_seq
        self.out_seq = out_seq

        self.act_func = act_func
        if self.act_func is not None:
            self.act_func = getattr(nn, self.act_func)()

        self.net = nn.ModuleList(nn.Linear(self.in_seq, self.in_seq) for _ in range(n_layers - 1))
        self.net.append(nn.Linear(self.in_seq, self.out_seq))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.net:
            x = layer(x)
            if self.act_func is not None:
                x = self.act_func(x)
        x = x.permute(0, 2, 1)

        return x


class Decoder(nn.Module):
    def __init__(self, in_seq, out_seq, n_layers=1, act_func=None):
        super(Decoder, self).__init__()
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.act_func = act_func
        if self.act_func is not None:
            self.act_func = getattr(nn, self.act_func)()

        self.net = nn.ModuleList(nn.Linear(self.in_seq, self.in_seq) for _ in range(n_layers - 1))
        self.net.append(nn.Linear(self.in_seq, self.out_seq))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.net:
            x = layer(x)
            if self.act_func is not None:
                x = self.act_func(x)
        x = x.permute(0, 2, 1)

        return x


class ODEFunc(nn.Module):

    def __init__(self, hidden_size, n_layers=1, act_func=None):
        super(ODEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.act_func = act_func
        if self.act_func is not None:
            self.act_func = getattr(nn, self.act_func)()

        self.net = nn.ModuleList(nn.Linear(self.hidden_size, self.hidden_size) for _ in range(n_layers - 1))
        self.net.append(nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, t, x):
        x = x.permute(0, 2, 1)

        for layer in self.net:
            x = layer(x)
            if self.act_func is not None:
                x = self.act_func(x)
        y = x
        y = y.permute(0, 2, 1)

        return y


class SDEFunc(nn.Module):
    noise_type = 'scalar'
    sde_type = 'stratonovich'

    def __init__(self, hidden_size, batch_size, enc_in):
        super(SDEFunc, self).__init__()
        self.hidden_size = hidden_size
        self.enc_in = enc_in
        self.brownian_size = 1  # Scalar noise must have only one channel
        self.batch_size = batch_size
        self.mu = nn.Linear(self.hidden_size, self.hidden_size)
        self.sigma = nn.Linear(self.hidden_size, self.hidden_size * self.brownian_size)

    # Drift
    def f(self, t, y):  # y shape [B, C, S]
        return self.mu(y)  # shape (B,C,S)

    # Diffusion
    def g(self, t, y):  # y shape [B, C, S] / self.c = (B,C,S,D)
        return self.sigma(y).reshape(self.batch_size, self.enc_in, self.hidden_size, self.brownian_size)  # shape (B,C,S,D)


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
        self.sde_solver = configs.sde_solver
        self.ode_step_size_T = configs.ode_step_size_T
        self.ode_step_size_S = configs.ode_step_size_S
        self.ode_step_size_R = configs.ode_step_size_R

        self.kernel_size = configs.moving_avg
        self.period = configs.tsr_period

        self.kinetic = configs.kinetic
        self.jacobian = configs.jacobian

        self.series_decomp = series_decomp(self.kernel_size)

        self.ode_hidden_size_T = configs.ode_hidden_size_T
        self.ode_hidden_size_S = configs.ode_hidden_size_S
        self.ode_hidden_size_R = configs.ode_hidden_size_R

        self.act_func_enc = configs.act_func_enc
        self.enc_T = Encoder(self.seq_len, self.ode_hidden_size_T, act_func=self.act_func_enc)
        self.enc_S = Encoder(self.seq_len, self.ode_hidden_size_S, act_func=self.act_func_enc)
        self.enc_R = Encoder(self.seq_len, self.ode_hidden_size_R, act_func=self.act_func_enc)

        self.num_ode_func_layers = configs.num_ode_func_layers
        self.act_func_ode_func = configs.act_func_ode_func
        self.ode_func_T = ODEFunc(self.ode_hidden_size_T, self.num_ode_func_layers, act_func=self.act_func_ode_func)
        self.ode_func_S = ODEFunc(self.ode_hidden_size_S, self.num_ode_func_layers, act_func=self.act_func_ode_func)
        self.sde_func_R = SDEFunc(self.ode_hidden_size_R, self.batch_size, self.enc_in)

        self.ode_func_T = KineticWrapper(self.ode_func_T, self.kinetic, self.jacobian, 1)
        self.ode_func_S = KineticWrapper(self.ode_func_S, self.kinetic, self.jacobian, 1)

        self.act_func_dec = configs.act_func_dec
        self.dec_T = Decoder(self.ode_hidden_size_T, self.pred_len, act_func=self.act_func_dec)
        self.dec_S = Decoder(self.ode_hidden_size_S, self.pred_len, act_func=self.act_func_dec)
        self.dec_R = Decoder(self.ode_hidden_size_R, self.pred_len, act_func=self.act_func_dec)

        self.revin_T = RevIN(self.enc_in)
        self.revin_S = RevIN(self.enc_in)
        self.revin_R = RevIN(self.enc_in)

    def forward(self, x, t):
        # x: [Batch, Input length, Channel]
        device = x.get_device()
        # print(x.shape)

        init_T, init_S, init_R = self.tsr_decomp(x, device)

        # Trend
        opt_T = {}
        opt_T.update({'method': self.ode_solver})
        opt_T.update({'h': self.ode_step_size_T})
        opt_T.update({'t0': 0.0})
        opt_T.update({'t1': 1.0})
        opt_T.update({'rtol': self.rtol_T})
        opt_T.update({'atol': self.atol_T})
        opt_T.update({'print_neval': False})  # nfe counts
        opt_T.update({'neval_max': 1000000})

        init_T = self.revin_T(init_T, 'norm')
        init_T = self.enc_T(init_T)
        init_T = (init_T, torch.zeros(init_T.size(0)).to(init_T), torch.zeros(init_T.size(0)).to(init_T))
        y_T, y_T_kinetic, y_T_jacobian = odeint(func=self.ode_func_T, y0=init_T, options=opt_T)
        y_T = self.dec_T(y_T)
        y_T = self.revin_T(y_T, 'denorm')
        reg_T = self.kinetic * y_T_kinetic.mean() + self.jacobian * y_T_jacobian.mean()

        # Seasonality
        opt_S = {}
        opt_S.update({'method': self.ode_solver})
        opt_S.update({'h': self.ode_step_size_S})
        opt_S.update({'t0': 0.0})
        opt_S.update({'t1': 1.0})
        opt_S.update({'rtol': self.rtol_S})
        opt_S.update({'atol': self.atol_S})
        opt_S.update({'print_neval': False})  # nfe counts
        opt_S.update({'neval_max': 1000000})

        init_S = self.revin_S(init_S, 'norm')
        init_S = self.enc_S(init_S)
        init_S = (init_S, torch.zeros(init_S.size(0)).to(init_S),
                  torch.zeros(init_S.size(0)).to(init_S))
        y_S, y_S_kinetic, y_S_jacobian = odeint(func=self.ode_func_S, y0=init_S, options=opt_S)
        y_S = self.dec_S(y_S)
        y_S = self.revin_S(y_S, 'denorm')
        reg_S = self.kinetic * y_S_kinetic.mean() + self.jacobian * y_S_jacobian.mean()

        # Residual
        init_R = self.revin_R(init_R, 'norm')
        init_R = self.enc_R(init_R)
        t = torch.Tensor([0, 1])
        init_R = init_R.permute(0, 2, 1)
        y_R = sdeint(sde=self.sde_func_R, y0=init_R, ts=t, method=self.sde_solver,
                     options={'step_size': self.ode_step_size_R})
        y_R = y_R[-1, :, :, :].permute(0, 2, 1).to(device)
        y_R = self.dec_R(y_R)
        y_R = self.revin_R(y_R, 'denorm')

        y = y_T + y_S + y_R
        reg = reg_T + reg_S

        return y, reg  # to [Batch, Output length, Channel]

    def tsr_decomp(self, x, device):
        res, trend_init = self.series_decomp(x)
        seasonal = torch.Tensor().to(device)
        for j in range(self.period):
            period_average = torch.unsqueeze(torch.mean(res[:, j::self.period, :], axis=1), dim=1)
            seasonal = torch.concat([seasonal, period_average], dim=1)
        seasonal = seasonal - torch.unsqueeze(torch.mean(seasonal, dim=1), dim=1)
        seasonal_init = torch.tile(seasonal.T, (1, x.shape[1] // self.period + 1, 1)).T[:, :x.shape[1], :]
        resid_init = res - seasonal_init

        return trend_init, seasonal_init, resid_init
