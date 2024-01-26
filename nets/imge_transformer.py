#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hid_dim, n_heads):
        super(ScaledDotProductAttention, self).__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert self.hid_dim % self.n_heads == 0

        self.d_k = self.hid_dim // self.n_heads

    def forward(self, Q, K, V):
        '''
        输入进来的维度：
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        assert self.d_model % self.n_heads == 0

        self.d_k = self.d_v = self.d_model // self.n_heads
        self.scaledDotProductAttention = ScaledDotProductAttention(self.d_model, self.n_heads)

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)

        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

        self.normlayer = nn.LayerNorm(self.d_model)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        context = self.scaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return self.normlayer(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()

        self.d_model, self.d_ff = d_model, d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )

        self.normlayer = nn.LayerNorm(self.d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)

        return self.normlayer(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(EncoderLayer, self).__init__()

        self.d_model, self.n_heads, self.d_ff = d_model, n_heads, d_ff
        self.enc_self_attn = MultiHeadAttention(self.d_model, self.n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(self.d_model, self.d_ff)

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, n_layers=6, d_model=256 * 256, n_heads=8, d_ff=256 * 256 * 4):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.d_model, self.n_heads, self.d_ff = d_model, n_heads, d_ff
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.n_heads, self.d_ff) for _ in
                                     range(self.n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = enc_inputs
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)

        return enc_outputs


if __name__ == '__main__':
    d_model = 512  # Embedding Size
    n_heads = 8  # number of heads in Multi-Head Attention
    d_ff = 2048  # FeedForward dimension
    n_layers = 6  # number of Encoder of Decoder Layer
    d_k = d_v = d_model / n_heads  # dimension of K(=Q), V

    assert d_model % n_heads == 0

    mode = Encoder(n_layers, d_model, n_heads, d_ff)

    print(mode)
