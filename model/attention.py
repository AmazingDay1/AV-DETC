#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : attention.py
import torch.nn as nn


class ConvergeAttention(nn.Module):
    def __init__(self, dim=192, head=6, proj_drop=0.1):
        super().__init__()
        self.heads = head
        head_dim = dim // head
        self.scale = head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, 2 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(proj_drop)

    def forward(self, t, f):
        residual_x = t
        Bt, Nt, Ct = t.shape
        Bf, Nf, Cf = f.shape

        f = self.norm(f)
        t = self.norm(t)

        q = self.q(t).reshape(Bt, Nt, self.heads, Ct // self.heads).permute(0, 2, 1, 3).contiguous()

        k, v = self.kv(f).chunk(2, dim=-1)
        k = k.reshape(Bf, Nf, self.heads, Cf // self.heads).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(Bf, Nf, self.heads, Cf // self.heads).permute(0, 2, 1, 3).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(Bt, Nt, -1).contiguous()
        x = self.proj(x)
        x = self.attn_drop(x)

        x = x + residual_x
        return x
