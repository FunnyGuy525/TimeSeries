import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from Embed import DataEmbedding



def FFT_for_Periods(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)

    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size= 2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        self.Inception = nn.Sequential(
            InceptionBlock(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            InceptionBlock(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        # 进入TimesBlock时x的维度： torch.Size([32, 192, 32])
        # print('进入TimesBlock时x的维度：',x.shape)

        # 1D-Variations into 2D-Variations
        periods, period_weight = FFT_for_Periods(x, k=self.k)

        lis = []
        for i in range(self.k):
            period = periods[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                pad_len = period - ((self.seq_len + self.pred_len) % period)
                padding = torch.zeros([x.shape[0], pad_len, x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim = 1)
            else:
                out = x
                # out: [B,T,N]
            out = out.reshape(out.shape[0], out.shape[1] // period, period, out.shape[2]).permute(0,3,1,2).contiguous()
            # out: [B, N, T // period, period]
            
            # print('patch后x形状：',out.shape)

            # Capturing temporal 2D-variations
            out = self.Inception(out)
            # Transform 2D representation to 1D
            out = out.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[-1])
            lis.append(out[:, :(self.seq_len + self.pred_len), :])
        lis = torch.stack(lis, dim = -1)

        # Adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, x.shape[1], x.shape[2], 1)

        out = torch.sum(lis * period_weight, dim = -1)
        # print('输出x形状：',x.shape)
        return out + x

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.e_layers = configs.e_layers

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])

        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

            
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 标准化
        means = x_enc.mean(1, keepdim=True).detach() # x: [B,T]
        x_enc = x_enc - means
        std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+1e-5)
        x_enc /= std

        enc_out = self.enc_embedding(x_enc, x_mark_enc) # [B, T, d_model]
        # 映射到 [B, seq_len + pred_len, d_model]
        enc_out = self.predict_linear(enc_out.permute(0,2,1)).permute(0,2,1)

        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))

        dec_out = self.projection(enc_out)
        dec_out = dec_out * (std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 标准化
        means = x_enc.mean(1, keepdim=True).detach() # x: [B,T]
        x_enc = x_enc - means
        std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+1e-5)
        x_enc /= std

        enc_out = self.enc_embedding(x_enc, x_mark_enc) # [B, T, d_model]
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)
        dec_out = dec_out * (std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out
    
    def anomaly_detection(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 标准化
        means = x_enc.mean(1, keepdim=True).detach() # x: [B,T]
        x_enc = x_enc - means
        std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+1e-5)
        x_enc /= std

        enc_out = self.enc_embedding(x_enc, None) # [B, T, d_model]
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        
        dec_out = self.projection(enc_out)
        dec_out = dec_out * (std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        return dec_out


    def classification(self, x_enc, x_mark_enc):

        enc_out = self.enc_embedding(x_enc, None)  # [B,T,d_model]

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)


        # x_mark_enc: 用于标记填充位置的张量
        # 与输出张量进行逐元素乘法运算，从而将填充位置的嵌入置零。
        output = output * x_mark_enc.unsqueeze(-1)
        
        # (B, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (B, num_classes)
        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :] # 返回预测部分
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # 返回丢失补全的全部序列
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # 返回正确序列
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # 返回分类结果
        return None
    