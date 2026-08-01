#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : zy.xiao
# @File     : mamba.py
import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from functools import partial

try:
    import selective_scan_cuda_oflex
except ImportError:
    selective_scan_cuda_oflex = None
try:
    import selective_scan_cuda_core
except ImportError:
    selective_scan_cuda_core = None
try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None

from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn
from timm.models.layers import DropPath


class MambaBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_state=16,
                 norm_epsilon=1e-5,
                 d_conv=4,
                 expand=2,
                 dt_rank="auto",
                 fused_add_norm=True,
                 layer_idx=None,
                 drop_path=None,
                 t_f_kind=""
                 ):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.norm_cls = partial(RMSNorm, eps=norm_epsilon)
        self.norm = self.norm_cls(self.d_model)

        self.fused_add_norm_fn = rms_norm_fn
        self.layer_idx = layer_idx

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.activate = nn.SiLU()
        self.t_f_kind = t_f_kind
        if t_f_kind == "time":
            self.con2d = nn.Conv2d(self.d_model * 2, self.d_model * 2, kernel_size=(3, 1), stride=(1, 1),
                                   padding=(1, 0), groups=self.d_model * 2)
            self.ssm = SSM(self.d_model, self.d_state, self.d_inner, self.dt_rank)

        elif t_f_kind == "frequency":
            self.con2d = nn.Conv2d(self.d_model * 2, self.d_model * 2, kernel_size=(1, 3), stride=(1, 1),
                                   padding=(0, 1), groups=self.d_model * 2)
            self.ssm = SSM(self.d_model, self.d_state, self.d_inner, self.dt_rank)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x, residual):
        if residual is None:
            x, residual = self.fused_add_norm_fn(x,
                                                 self.norm.weight,
                                                 self.norm.bias,
                                                 residual=residual,
                                                 prenorm=True,
                                                 residual_in_fp32=True,
                                                 eps=self.norm.eps)
        else:
            x, residual = self.fused_add_norm_fn(self.drop_path(x),
                                                 self.norm.weight,
                                                 self.norm.bias,
                                                 residual=residual,
                                                 prenorm=True,
                                                 residual_in_fp32=True,
                                                 eps=self.norm.eps)

        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)

        if self.t_f_kind == "time":
            x = x.unsqueeze(2).permute(0, 3, 1, 2).contiguous()
            x = self.con2d(x)
            x = x.squeeze(3)

        elif self.t_f_kind == "frequency":
            x = x.unsqueeze(1).permute(0, 3, 1, 2).contiguous()
            x = self.con2d(x)
            x = x.squeeze(2)

        x = self.activate(x)
        y = self.ssm.forward_fn(x)
        y = y * self.activate(z)

        out = self.out_proj(y)
        return out, residual


class ImgMambaBlock(nn.Module):
    def __init__(self,
                 d_model,
                 d_state=16,
                 norm_epsilon=1e-5,
                 d_conv=4,
                 expand=2,
                 dt_rank="auto",
                 fused_add_norm=True,
                 layer_idx=None,
                 drop_path=None,
                 ):
        super(ImgMambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.norm_cls = partial(RMSNorm, eps=norm_epsilon)
        self.norm = self.norm_cls(self.d_model)

        self.fused_add_norm_fn = rms_norm_fn
        self.layer_idx = layer_idx

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.activate = nn.SiLU()
        # forward scan
        self.con2d_f = nn.Conv2d(self.d_model * 2, self.d_model * 2, kernel_size=(3, 1), stride=(1, 1),
                                 padding=(1, 0), groups=self.d_model * 2)
        self.ssm_f = SSM(self.d_model, self.d_state, self.d_inner, self.dt_rank)

        # backward scan
        self.con2d_b = nn.Conv2d(self.d_model * 2, self.d_model * 2, kernel_size=(1, 3), stride=(1, 1),
                                 padding=(0, 1), groups=self.d_model * 2)
        self.ssm_b = SSM(self.d_model, self.d_state, self.d_inner, self.dt_rank)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x, residual):
        if residual is None:
            x, residual = self.fused_add_norm_fn(x,
                                                 self.norm.weight,
                                                 self.norm.bias,
                                                 residual=residual,
                                                 prenorm=True,
                                                 residual_in_fp32=True,
                                                 eps=self.norm.eps)
        else:
            x, residual = self.fused_add_norm_fn(self.drop_path(x),
                                                 self.norm.weight,
                                                 self.norm.bias,
                                                 residual=residual,
                                                 prenorm=True,
                                                 residual_in_fp32=True,
                                                 eps=self.norm.eps)

        x = self.in_proj(x)

        # forward
        x_f, z_f = x.chunk(2, dim=-1)
        x_f = x_f.unsqueeze(2).permute(0, 3, 1, 2).contiguous()
        x_f = self.con2d_f(x_f)
        x_f = x_f.squeeze(3)
        x_f = self.activate(x_f)
        y_f = self.ssm_f.forward_fn(x_f)
        y_f = y_f * self.activate(z_f)

        # backward
        x_b, z_b = x.flip([-1]).chunk(2, dim=-1)
        x_b = x_b.unsqueeze(2).permute(0, 3, 1, 2).contiguous()
        x_b = self.con2d_b(x_b)
        x_b = x_b.squeeze(3)
        x_b = self.activate(x_b)
        y_b = self.ssm_b.forward_fn(x_b)
        y_b = y_b * self.activate(z_b)

        y = (y_f + y_b.flip([-1])) / 2
        out = self.out_proj(y)

        return out, residual


class SSM(nn.Module):
    def __init__(self,
                 d_mode,
                 d_state,
                 d_inner,
                 rank,
                 dt_scale=1.0,
                 dt_max=0.1,
                 dt_min=0.001,
                 dt_init_floor=0.0001,
                 device=None):
        super(SSM, self).__init__()
        self.dt_max = dt_max

        self.d_model = d_mode
        self.d_state = d_state
        self.d_inner = d_inner
        self.dt_rank = rank

        self.out_norm = nn.LayerNorm(self.d_model * 2)

        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n",
                   d=self.d_inner).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj_weight = nn.Parameter(self.dt_proj.weight)
        self.dt_proj_bias = nn.Parameter(self.dt_proj.bias)
        del self.dt_proj

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

    def forward_fn(self, xs):
        B, D, L = xs.shape
        D, N = self.A_log.shape
        D, R = self.dt_proj_weight.shape

        x_dbl = self.x_proj(rearrange(xs, "b d l -> b l d"))
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b l r, d r -> b d l", dts, self.dt_proj_weight)
        xs = xs.view(B, -1, L).to(torch.float32)
        dts = dts.contiguous().view(B, -1, L).to(torch.float32)
        As = -self.A_log.to(torch.float).exp()
        Ds = self.D.to(torch.float)
        Bs = Bs.contiguous().view(B, N, L).to(torch.float32)
        Cs = Cs.contiguous().view(B, N, L).to(torch.float32)
        delta_bias = self.dt_proj_bias.view(-1).to(torch.float)

        Bs = Bs.unsqueeze(1)
        Cs = Cs.unsqueeze(1)
        y = selective_scan_fn(xs, dts, As, Bs, Cs, Ds, delta_bias=delta_bias, delta_softplus=True,
                              ssoflex=True, backend="mamba")

        y = y.permute(0, 2, 1).contiguous()
        y = self.out_norm(y)

        return y.to(xs.dtype)


def selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=None):
    fn = selective_scan_torch if backend == "torch" else SelectiveScanCuda.apply
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend)


class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        ctx.backend = backend
        if backend == "oflex":
            out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        elif backend == "core":
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        elif backend == "mamba":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        backend = ctx.backend
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if backend == "oflex":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "core":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "mamba":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False
            )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_torch(u,  # [B C L]
                         delta,  # [B C L]
                         A,  # [C N]
                         B,  # [B 1 N L]
                         C,  # [B 1 N L]
                         D,  # [C]
                         delta_bias,
                         delta_softplus,
                         ssoflex=None,
                         backend=None):
    dtyper_in = u.dtype
    _, inner_dim, _ = u.shape
    Bs, G, N, L = B.shape
    dim = u.shape[1]

    assert u.shape == (Bs, inner_dim, L)
    assert delta.shape == (Bs, inner_dim, L)
    assert A.shape == (inner_dim, N)
    assert C.shape == B.shape

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Bs, 1, N, L).repeat(1, dim, 1, 1).view(Bs, dim, N, L)
    C = C.view(Bs, 1, N, L).repeat(1, dim, 1, 1).view(Bs, dim, N, L)
    deltaA = torch.exp(torch.einsum("bdl, dn->bdln", delta, A))
    deltaB = torch.exp(torch.einsum("bdl, bdnl->bdln", delta, B))
    deltaBu = torch.einsum("bdln, bdl ->bdln", deltaB, u)

    hidden_state = A.new_zeros((Bs, dim, N))
    ys = []
    for i in range(L):
        hidden_state = deltaA[:, :, i, :] * hidden_state + deltaBu[:, :, i, :]
        y = torch.einsum("bdn, bdn->bd", hidden_state, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)

    out = y + u * D.unsqueeze(-1)
    return out.to(dtype=dtyper_in)
