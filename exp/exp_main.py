import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

parent_path = os.path.dirname(
    os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))))
sys.path.append(parent_path)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic

from models import LTSF_DNODE_v1, LTSF_DNODE_v2, LTSF_DNODE_v3

from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, count_parameters
from utils.soft_dtw_cuda import SoftDTW

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'LTSF_DNODE_v1': LTSF_DNODE_v1,
            'LTSF_DNODE_v2': LTSF_DNODE_v2,
            'LTSF_DNODE_v3': LTSF_DNODE_v3
        }

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.use_sdtw_loss:
            criterion = SoftDTW(use_cuda=True, gamma=0.1)
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'LTSF_DNODE' in self.args.model:
                        outputs, _ = self.model(batch_x, batch_x_mark)
                    elif 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if self.args.use_sdtw_loss:
                    pred = outputs
                    true = batch_y
                    loss = criterion(pred, true)
                    loss = loss.mean()
                    loss = loss.detach().cpu()
                else:
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def train(self, setting):

        total_params, trainable_params = count_parameters(self.model)
        print(f'##### total model parameters: {total_params}')
        print(f'##### trainable model parameters: {trainable_params}')

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                # print(f'[{i}/{len(train_loader)} batch] start training...')

                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)
                        if self.args.use_sdtw_loss:
                            loss = loss.mean()

                        train_loss.append(loss.item())

                else:
                    if 'LTSF_DNODE' in self.args.model:
                        outputs, reg = self.model(batch_x, batch_x_mark)
                    elif 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    if self.args.use_sdtw_loss:
                        loss = loss.mean()

                    if 'LTSF_DNODE' in self.args.model:
                        loss = loss + reg

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST' or self.args.lradj == 'DNODE':
                    adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # save loss
            # loss_path = os.path.abspath(self.args.loss_save_path)
            # if os.path.exists(loss_path):
            #     df_loss = pandas.read_csv(loss_path)
            # else:
            #     df_loss = pandas.DataFrame(
            #         columns=['timestamp', 'model_id', 'model', 'data',
            #                  'batch_size', 'lr', 'pred_len',
            #                  'epoch', 'train_loss', 'valid_loss', 'test_loss'])
            # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # df_loss.loc[len(df_loss)] = [timestamp, self.args.model_id, self.args.model, self.args.data,
            #                              self.args.batch_size, self.args.learning_rate, self.args.pred_len,
            #                              epoch, train_loss, vali_loss, test_loss]
            # df_loss.to_csv(self.args.loss_save_path, index=False)

            early_stopping(vali_loss, self.model, path, epoch=epoch)
            self.best_epoch = early_stopping.best_epoch
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj == 'TST' or self.args.lradj == 'DNODE':
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler)
            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, args=None, seasonal_predict_fn=None, trend_predict_fn=None):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        time_now = time.time()

        print(f'length of test loader: {len(test_loader)}')

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'LTSF_DNODE' in self.args.model:
                        outputs, _ = self.model(batch_x, batch_x_mark)
                    elif 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / 100
                    print('\titers: {0}, speed: {1:.4f}s/iter'.format(i + 1, speed))
                    time_now = time.time()

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path) and self.args.save_results:
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        if self.args.save_results:
            np.save(folder_path + 'pred.npy', preds)

        if args is not None:
            results_path = os.path.abspath(args.metric_save_path)
            # 'train_epochs'
            if os.path.exists(results_path):
                df_results = pandas.read_csv(results_path)
            else:
                columns = ['timestamp', 'model_id', 'model', 'data',
                           'kinetic', 'jacobian',
                           'num_enc_layers', 'act_func_enc',
                           'num_ode_func_layers', 'act_func_ode_func',
                           'ode_hidden_size_T', 'ode_hidden_size_S', 'ode_hidden_size_R',
                           'num_dec_layers', 'act_func_dec',
                           'ode_solver',
                           'ode_step_size_T', 'ode_step_size_S', 'ode_step_size_R',
                           'rtol_T', 'atol_T',
                           'rtol_S', 'atol_S',
                           'rtol_R', 'atol_R',
                           'features', 'batch_size', 'learning_rate',
                           'lradj',
                           'seq_len', 'pred_len',
                           'train_epochs', 'best_epoch', 'patience',
                           'moving_avg', 'tsr_period',
                           'des',
                           'mse', 'mae', 'mape', 'rse']
                df_results = pandas.DataFrame(columns=columns)

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            addt_row = [timestamp, args.model_id, args.model, args.data,
                        args.kinetic, args.jacobian,
                        args.num_enc_layers, args.act_func_enc,
                        args.num_ode_func_layers, args.act_func_ode_func,
                        args.ode_hidden_size_T, args.ode_hidden_size_S, args.ode_hidden_size_R,
                        args.num_dec_layers, args.act_func_dec,
                        args.ode_solver,
                        args.ode_step_size_T, args.ode_step_size_S, args.ode_step_size_R,
                        args.rtol_T, args.atol_T,
                        args.rtol_S, args.atol_S,
                        args.rtol_R, args.atol_R,
                        args.features, args.batch_size, args.learning_rate,
                        args.lradj,
                        args.seq_len, args.pred_len,
                        args.train_epochs, self.best_epoch, args.patience,
                        args.moving_avg, args.tsr_period,
                        args.des,
                        mse, mae, mape, rse]
            print(addt_row)
            df_results.loc[len(df_results)] = addt_row

            # df_results.sort_values(by="mse", ascending=True, inplace=True)
            print(df_results)
            df_results.to_csv(results_path, index=False)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
