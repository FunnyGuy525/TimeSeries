import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from Embed import iT_PatchEmbedding



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

        out = self.Inception(x)
        # print('输出x形状：',x.shape)
        return out + x

class Model(nn.Module):
    
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.e_layers = configs.e_layers
        padding = stride
        self.patch_embedding = iT_PatchEmbedding(configs.enc_in, patch_len, stride, padding, configs.dropout)

        self.Num_of_Patch = int((self.seq_len - patch_len) / stride + 2)

        self.conv1 = nn.Conv2d(self.Num_of_Patch, configs.d_model, kernel_size=3, padding=1)
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.conv2 = nn.Conv2d(configs.d_model, self.pred_len, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.layer_norm = nn.LayerNorm([configs.enc_in, patch_len])

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(patch_len, 1, bias=True)
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

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc) #  # enc_out:[B, C, N, P]
        

        enc_out = enc_out.permute(0, 2, 1, 3)
        enc_out = self.conv1(enc_out)
        B, _, C_in, P = enc_out.shape
        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.model[i](enc_out).view(-1, C_in, P)).view(B, -1, C_in, P)

        enc_out = self.conv2(enc_out)
        B, p_len, _, _ = enc_out.shape

        dec_out = self.projection(enc_out).view(B, p_len, -1)
        dec_out = dec_out * (std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

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