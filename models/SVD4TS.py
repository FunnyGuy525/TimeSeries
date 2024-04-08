import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):

    def __init__(self, configs):
        # """
        # individual: Bool, whether shared model among different variates.
        # """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
        # self.decompsition = series_decomp(configs.moving_avg)
            
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
        
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        B, L, D = x.shape
        U, S, V = torch.svd(x, some=False)
        # U = [u1,u2,u3...]: [L, L ]，为 x 左特征向量，时序信息
        # V = [v1,v2,v3...]: [D, D] ，为 x 右特征向量，通道信息
        # print('U:',U.shape)

        U_p = self.Linear(U.permute(0, 2, 1)) # U_p : [seq, pre]
        U_new = self.Conv(U_p).permute(0, 2, 1) # U_new : [pre, channels] or [pre, seq]

        V_new = self.Linear_C(V.permute(0, 2, 1))
        

        if L > D:
            U = F.normalize(U_p[:, :D, :].permute(0, 2, 1) + U_new, p=2, dim=1)
            V = F.normalize(V_new.permute(0, 2, 1) + V, p=2, dim = 1)
        else:
            U = F.normalize(U_p.permute(0, 2, 1) + U_new, p=2, dim=1)
            V = F.normalize(V_new.permute(0, 2, 1) + V, p=2, dim = 1)[:,:,:L]
        # print('U_new:',U_new.shape)
        # print('torch.diag_embed(S):',torch.diag_embed(S).shape)


        x_new = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(1, 2))
        
        return x_new

    def forecast(self, x_enc):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        dec_out = self.encoder(x_enc)

        # print('dec_out的形状：', dec_out.shape)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

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

