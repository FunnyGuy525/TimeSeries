import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
class moving_avg(nn.Module):

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        # self.individual = individual
        self.channels = configs.enc_in




        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        if self.seq_len > self.channels:
            # self.Linear_S = nn.Linear(self.seq_len, self.seq_len)
            self.Conv = nn.Conv1d(in_channels=self.seq_len, out_channels=self.channels, kernel_size=3,padding=1)
        else:
            # self.Linear_S = nn.Linear(self.channels, self.channels)
            self.Conv = nn.Conv1d(in_channels=self.seq_len, out_channels=self.seq_len, kernel_size=3,padding=1)
        
        self.Linear_C = nn.Linear(self.channels, self.channels)

        self.Linear.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_C.weight = nn.Parameter(
            (1 / self.channels) * torch.ones([self.channels, self.channels]))
        

        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        B, L, D = x.shape

        seasonal_init, trend_init = self.decompsition(x)


        U, S, V = torch.svd(seasonal_init, some=False)
        # U = [u1,u2,u3...]: [L, L ]，为 x 左特征向量，时序信息
        # V = [v1,v2,v3...]: [D, D] ，为 x 右特征向量，通道信息

        U_p = self.Linear(U.permute(0, 2, 1)) # U_p : [seq, pre]
        U_new = self.Conv(U_p).permute(0, 2, 1) # U_new : [pre, channels] or [pre, seq]

        V_new = self.Linear_C(V.permute(0, 2, 1))
        

        if L > D:
            U = F.normalize(U_p[:, :D, :].permute(0, 2, 1) + U_new, p=2, dim=1)
            V = F.normalize(V_new.permute(0, 2, 1) + V, p=2, dim = 1)
        else:
            U = F.normalize(U_p.permute(0, 2, 1) + U_new, p=2, dim=1)
            V = F.normalize(V_new.permute(0, 2, 1) + V, p=2, dim = 1)[:,:,:L]

        seasonal_output = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(1, 2))

        trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1))

        x = seasonal_output + trend_output.permute(0, 2, 1)
        return x

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None

