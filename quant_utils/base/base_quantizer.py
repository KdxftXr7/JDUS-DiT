import logging
import warnings
from typing import Union
import time
import math
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
import transformers
from tqdm import tqdm


logger = logging.getLogger(__name__)


class BaseQuantizer(nn.Module):
    def __init__(self, quant_config):
        super(BaseQuantizer, self).__init__()
        
        # unpack the quant configurations
        self.n_bits = quant_config['n_bits']
        # self.group = quant_config['group']
        self.sym = quant_config.get('sym', False)
        
        # self.gptq = quant_config.get('gptq', False)
        self.group_size = quant_config.get('group_size', 0)
        
        self.gptq = quant_config.get('gptq', False)

        if isinstance(self.n_bits, list):
            raise AssertionError("when multiple n_bits are adopted, use the MixedPrecisionBaseQuantizer")
        # assert self.group in ['token','tensor','channel']

        self.register_buffer('delta', None)
        self.register_buffer('zero_point', None)

        # INFO: for mixed_precision, the n_bits could be a ListConfig, and need to be initialized in subclass init
        if not isinstance(self.n_bits, ListConfig):
            self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1

        self.init_done = False
        
    def forward(self, x: torch.Tensor):
        raise NotImplementedError("should be implemented in subclass.")

    def init_quant_params(self, x):
        raise NotImplementedError("should be implemented in subclass.")
    
    
class StaticQuantizer(BaseQuantizer):
    """
    the input shape should be [Group,-1]
    store the quant params (delta, zp) offline with init_quant_params
    """
    def __init__(self, quant_config):
        super().__init__(quant_config)

        if self.sym:
            self.x_absmax = None
        else:
            self.x_max = None
            self.x_min = None
        
        if self.gptq:
            self.gptq_quantizer = GPTQQuantizer()
            self.gptq_H = None
            self.gptq_nsamples = 0
            
            gptq_config = quant_config.get('gptq_config', {})
            # self.gptq_bits = gptq_config.get('n_bits')
            self.gptq_bits = self.n_bits
            self.gptq_perchannel = gptq_config.get('perchannel', True)
            self.gptq_channel_group = gptq_config.get('channel_group', 1)
            self.gptq_sym = gptq_config.get('sym', False)
            self.gptq_mse = gptq_config.get('mse', True)
            self.gptq_norm = gptq_config.get('norm', 2.4)
            self.gptq_grid = gptq_config.get('grid', 100)
            self.gptq_maxshrink = gptq_config.get('maxshrink', 0.8)
            self.gptq_clip_ratio = gptq_config.get('clip_ratio', 1.0)
            # self.gptq_blocksize = gptq_config.get('blocksize', 128)
            self.gptq_blocksize = self.group_size
            if self.gptq_blocksize <= 0:
                raise NotImplementedError("gptq_blocksize should be larger than 0")
            self.gptq_percdamp = gptq_config.get('percdamp', 0.01)
            self.gptq_groupsize = self.group_size
            
            self.gptq_quantizer.configure(
                bits=self.gptq_bits,
                perchannel=self.gptq_perchannel,
                channel_group=self.gptq_channel_group,
                sym=self.gptq_sym,
                mse=self.gptq_mse,
                norm=self.gptq_norm,
                grid=self.gptq_grid,
                maxshrink=self.gptq_maxshrink,
                clip_ratio=self.gptq_clip_ratio,
            )
        
        self.init_finish = True
            
            
    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 2
        # import pdb; pdb.set_trace()
        
        if self.gptq and hasattr(self, 'gptq_quantizer'):
            x_dequant = self.quantize(x)
            return x_dequant
        else:
            if self.group_size == 0:
                x_quant = self.quantize(x)
                x_dequant = (x_quant - self.zero_point) * self.delta
            else:
                x_quant = self.quantize(x)
                assert x.shape[1] % self.group_size == 0, "the input shape should be divisible by group_size"
                # n_groups = math.ceil(x.shape[1] / self.group_size)
                n_groups = x.shape[1] // self.group_size
                x_dequant = torch.zeros_like(x, device=x.device)
                
                for g_idx in range(n_groups):
                    g_start = g_idx * self.group_size
                    g_end = min((g_idx + 1) * self.group_size, x.shape[1])
                    
                    g_x_quant = x_quant[:, g_start:g_end]  # [C_out, group_size]
                    
                    g_delta = self.delta[:, g_idx].unsqueeze(-1)  # [C_out, 1]
                    g_zero_point = self.zero_point[:, g_idx].unsqueeze(-1)  # [C_out, 1]
                    g_delta = g_delta.to(g_x_quant.device)  # ensure the same device
                    g_zero_point = g_zero_point.to(g_x_quant.device)  # ensure the same device
                    
                    g_x_dequant = (g_x_quant - g_zero_point) * g_delta
                    
                    x_dequant[:, g_start:g_end] = g_x_dequant
                    
            return x_dequant
    
    def quantize(self, x: torch.Tensor):
        assert len(x.shape) == 2
        
        if self.gptq and hasattr(self, 'gptq_quantizer'):
            # import pdb; pdb.set_trace()
            return self.gptq_quantize(x)
        
        if self.init_done is not True:  # set as True in ptq.py
            self.init_quant_params(x)
            
        
        
        if self.group_size == 0:
            x_int = torch.round(x / self.delta) + self.zero_point
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        else:
            assert x.shape[1] % self.group_size == 0, "the input shape should be divisible by group_size"
            # n_groups = math.ceil(x.shape[1] / self.group_size)
            n_groups = x.shape[1] // self.group_size
            x_int = torch.zeros_like(x, device=x.device)
            x_quant = torch.zeros_like(x, device=x.device)
            
            for g_idx in range(n_groups):
                g_start = g_idx * self.group_size
                g_end = min((g_idx + 1) * self.group_size, x.shape[1])
                
                g_x = x[:, g_start:g_end]  # [C_out, group_size]
                
                g_delta = self.delta[:, g_idx].unsqueeze(-1)  # [C_out, 1]
                g_zero_point = self.zero_point[:, g_idx].unsqueeze(-1)  # [C_out, 1]
                g_delta = g_delta.to(g_x.device)  # ensure the same device
                g_zero_point = g_zero_point.to(g_x.device)  # ensure the same device
                
                # quantize the group
                g_x_int = torch.round(g_x / g_delta) + g_zero_point
                if self.sym:
                    g_x_quant = torch.clamp(g_x_int, -self.n_levels - 1, self.n_levels)
                else:
                    g_x_quant = torch.clamp(g_x_int, 0, self.n_levels - 1)
                
                # store the quantized values back to the original tensor
                x_int[:, g_start:g_end] = g_x_int
                x_quant[:, g_start:g_end] = g_x_quant
            
        return x_quant
    
    def init_quant_params(self, x):
        assert len(x.shape) == 2  # [C_out, C_in]
        if self.group_size == 0:
            if self.sym:
                x_absmax = x.abs().max(dim=1)[0]
                self.x_absmax = (torch.max(self.x_absmax, x_absmax) if self.x_absmax is not None else x_absmax).to("cuda")  # update
                delta = x_absmax / self.n_levels
                zero_point = torch.zeros_like(delta, device=delta.device)
            else:
                x_max = x.max(dim=1)[0]
                x_max = torch.maximum(x_max, torch.zeros_like(x_max))  # set negative values to 0
                # sometimes the weight are init on CPU, but new data on GPU needed for update quant_params (quarot)
                self.x_max = torch.max(self.x_max.to(x_max.device), x_max) if self.x_max is not None else x_max

                x_min = x.min(dim=1)[0]
                x_min = torch.minimum(x_min, torch.zeros_like(x_min))  # set positive values to 0
                self.x_min = torch.min(self.x_min.to(x_min.device), x_min) if self.x_min is not None else x_min

                delta = (x_max - x_min)/(self.n_levels-1)
                zero_point = torch.round(-x_min/delta).clamp(0, self.n_levels-1)
            
            try:
                assert torch.all(delta > 1.e-6), "unexpected small delta exists"
            except:
                import pdb; pdb.set_trace()

            self.delta = delta.unsqueeze(-1)  # [C_out] -> [C_out, 1]
            self.zero_point = zero_point.unsqueeze(-1)  # [C_out] -> [C_out, 1]
        else:
            assert x.shape[1] % self.group_size == 0, "the input shape should be divisible by group_size"
            # n_groups = math.ceil(x.shape[1] / self.group_size)
            n_groups = x.shape[1] // self.group_size
            
            if self.sym:
                if self.x_absmax is None:
                    self.x_absmax = torch.full([x.shape[0], n_groups], float('-inf'), device=x.device)
                
                for g_idx in range(n_groups):
                    g_start = g_idx * self.group_size
                    g_end = min((g_idx + 1) * self.group_size, x.shape[1])
                    g_x = x[:, g_start:g_end]  # [C_out, group_size]
                    g_absmax = g_x.abs().max(dim=1)[0]
                    self.x_absmax[:, g_idx] = torch.max(self.x_absmax[:, g_idx].to(g_absmax.device), g_absmax)
                
                delta = self.x_absmax.view(x.shape[0], n_groups) / self.n_levels
                zero_point = torch.zeros_like(delta, device=delta.device)
            else:
                delta = torch.zeros((x.shape[0], n_groups), device=x.device)
                zero_point = torch.zeros((x.shape[0], n_groups), device=x.device)
                if self.x_max is None:
                    self.x_max = torch.full([x.shape[0], n_groups], float('-inf'), device=x.device)
                if self.x_min is None:
                    self.x_min = torch.full([x.shape[0], n_groups], float('inf'), device=x.device)
                
                for g_idx in range(n_groups):
                    g_start = g_idx * self.group_size
                    g_end = min((g_idx + 1) * self.group_size, x.shape[1])
                    g_x = x[:, g_start:g_end]  # [C_out, group_size]
                    
                    g_max = g_x.max(dim=1)[0]
                    g_max = torch.maximum(g_max, torch.zeros_like(g_max))  # set negative values to 0
                    self.x_max[:, g_idx] = torch.max(self.x_max[:, g_idx].to(g_max.device), g_max)
                    
                    g_min = g_x.min(dim=1)[0]
                    g_min = torch.minimum(g_min, torch.zeros_like(g_min))  # set positive values to 0
                    self.x_min[:, g_idx] = torch.min(self.x_min[:, g_idx].to(g_min.device), g_min)
                    
                    g_delta = (g_max - g_min) / (self.n_levels-1)
                    g_zero_point = torch.round(-g_min / g_delta).clamp(0, self.n_levels-1)
                    delta[:, g_idx] = g_delta
                    zero_point[:, g_idx] = g_zero_point
            try:
                assert torch.all(delta > 1.e-6), "unexpected small delta exists"
            except:
                import pdb; pdb.set_trace()
            
            self.delta = delta
            self.zero_point = zero_point
    
    def gptq_add_batch(self, x, out=None):
        if not self.gptq:
            return

        if self.gptq_H is None:
            self.gptq_H = torch.zeros((x.shape[-1], x.shape[-1]), device=x.device)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        if len(x.shape) == 3:
            x = x.reshape((-1, x.shape[-1]))
            x = x.t()
        
        self.gptq_H *= self.gptq_nsamples / (self.gptq_nsamples + batch_size)
        self.gptq_nsamples += batch_size
        
        x_scaled = math.sqrt(2 / self.gptq_nsamples) * x.float()
        self.gptq_H += x_scaled.matmul(x_scaled.t())
    
    def gptq_quantize(self, x):
        if not self.gptq or self.gptq_H is None:
            # raise RuntimeError("GPTQ is not enabled or Hessian matrix is not initialized.")
            return x

        orig_shape = x.shape
        orig_dtype = x.dtype
        if len(x.shape) > 2:
            x = x.reshape(orig_shape[0], -1)
        
        W = x.clone().float()
        columns = W.shape[1]
        
        # import pdb; pdb.set_trace()
        
        if not self.gptq_quantizer.ready():
            self.gptq_quantizer.find_params(W, weight=True)
            
        H = self.gptq_H.clone()
        del self.gptq_H
        
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        Q = torch.zeros_like(W)
        
        damp = self.gptq_percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=H.device)
        H[diag, diag] += damp
        
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H.to("cuda")
        
        # import pdb; pdb.set_trace()
        
        n_nonout = columns
        for i1 in range(0, n_nonout, self.gptq_blocksize):
            i2 = min(i1 + self.gptq_blocksize, n_nonout)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                
                if self.gptq_groupsize > 0:
                    if (i1 + i) % self.gptq_groupsize == 0:
                        self.gptq_quantizer.find_params(
                            W[:, (i1 + i):min((i1 + i + self.gptq_groupsize), n_nonout)],
                            weight=True
                        )
                
                q = self.gptq_quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
                
            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            
        Q = Q.reshape(orig_shape).to(orig_dtype)
        
        return Q

class DynamicQuantizer(BaseQuantizer):
    """
    the input shape should be [Group,-1]
    store the quant params (delta, zp) offline with init_quant_params
    """
    def __init__(self, quant_config):
        super().__init__(quant_config)
    
    def quantize(self, x: torch.Tensor):
        # get the quant_params online
        assert len(x.shape) == 2  # [N_group, -1]
        assert torch.isnan(x).sum() == 0  # no nan exists

        if self.group_size == 0:
            if self.sym:
                x_absmax = x.abs().max(dim=1)[0]
                self.x_absmax = x_absmax
                
                delta = x_absmax / self.n_levels
                zero_point = torch.zeros_like(delta, device=delta.device)
                
                eps = 1.e-6
                try:
                    assert torch.all(delta.abs() > eps)
                except:
                    delta[delta < eps] = eps
                    logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(delta.abs().min(), self.module_name))
                    
            else:
                x_max = x.max(dim=1)[0]
                x_max[x_max<0] = 0. 
                self.x_max = x_max

                x_min = x.min(dim=1)[0]
                x_min[x_min>0] = 0.
                self.x_min = x_min

                delta = (x_max - x_min)/(self.n_levels-1)
                # INFO: check small values for delta, close to zero delta, would cause nan in zero_point
                eps = 1.e-8
                try:
                    assert torch.all(delta.abs() > eps)
                except:
                    import pdb; pdb.set_trace()
                    
                    delta[delta < eps] = eps
                    logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(delta.abs().min(), self.module_name))
                zero_point = torch.round(-x_min/delta).clamp(0, self.n_levels-1)

            self.delta = delta.unsqueeze(-1)  # [G] -> [G,1]
            self.zero_point = zero_point.unsqueeze(-1)

            # quantize model with quant params
            x_int = torch.round(x / self.delta) + self.zero_point
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            return x_quant
        else:
            assert x.shape[1] % self.group_size == 0, "the input shape should be divisible by group_size"
            n_groups = math.ceil(x.shape[1] / self.group_size)
            x_quant = torch.zeros_like(x, device=x.device)
            
            delta = torch.zeros((x.shape[0], n_groups), device=x.device)
            zero_point = torch.zeros((x.shape[0], n_groups), device=x.device)
            
            for g_idx in range(n_groups):
                g_start = g_idx * self.group_size
                g_end = min((g_idx + 1) * self.group_size, x.shape[1])
                
                g_x = x[:, g_start:g_end]
                
                if self.sym:
                    g_absmax = g_x.abs().max(dim=1)[0]
                    g_delta = g_absmax / self.n_levels
                    g_zero_point = torch.zeros_like(g_delta, device=g_delta.device)

                    eps = 1.e-6
                    try:
                        assert torch.all(g_delta.abs() > eps)
                    except:
                        delta[g_delta < eps] = eps
                        logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(g_delta.abs().min(), self.module_name))
                else:
                    g_max = g_x.max(dim=1)[0]
                    g_max[g_max<0] = 0. 
                    g_min = g_x.min(dim=1)[0]
                    g_min[g_min>0] = 0.

                    g_delta = (g_max - g_min) / (self.n_levels-1)
                    eps = 1.e-6
                    try:
                        assert torch.all(g_delta.abs() > eps)
                    except:
                        import pdb; pdb.set_trace()
                        g_delta[g_delta < eps] = eps
                        logger.info("unexpected small delta: {:.3e} exists in {}, set as eps".format(g_delta.abs().min(), self.module_name))
                    # g_zero_point = torch.round(g_min / g_delta) + (self.n_levels / 2)
                    g_zero_point = torch.round(-g_min / g_delta).clamp(0, self.n_levels-1)

                delta[:, g_idx] = g_delta
                zero_point[:, g_idx] = g_zero_point
                
                g_x_int = torch.round(g_x / g_delta.unsqueeze(-1)) + g_zero_point.unsqueeze(-1)
                if self.sym:
                    g_x_quant = torch.clamp(g_x_int, -self.n_levels - 1, self.n_levels)
                else:
                    g_x_quant = torch.clamp(g_x_int, 0, self.n_levels - 1)
                
                x_quant[:, g_start:g_end] = g_x_quant
                if g_x_quant.max() == 0 and g_x_quant.min() == 0:
                    import pdb; pdb.set_trace()
            
            self.delta = delta
            self.zero_point = zero_point
            return x_quant
                    
    
    def forward(self, x: torch.Tensor):
        x_quant = self.quantize(x)
        
        if self.group_size == 0:
            x_dequant = (x_quant - self.zero_point) * self.delta
        else:
            assert x.shape[1] % self.group_size == 0, "the input shape should be divisible by group_size"
            n_groups = math.ceil(x.shape[1] / self.group_size)
            x_dequant = torch.zeros_like(x, device=x.device)
            
            for g_idx in range(n_groups):
                g_start = g_idx * self.group_size
                g_end = min((g_idx + 1) * self.group_size, x.shape[1])
                
                g_x_quant = x_quant[:, g_start:g_end]
                
                g_delta = self.delta[:, g_idx].unsqueeze(-1)  # [G, 1]
                g_zero_point = self.zero_point[:, g_idx].unsqueeze(-1)  # [G, 1]
                
                g_x_dequant = (g_x_quant - g_zero_point) * g_delta
                
                x_dequant[:, g_start:g_end] = g_x_dequant
            
        return x_dequant
    
    def forward_with_quant_params(self, x, delta, mixed_precision=None):
        # INFO: used for attn block-wise quant, with precomputed delta
        # take in the x and delta with the same shape
        assert self.sym

        # INFO: meant to check attn_map only, but we use this for qk quant pre_softmax also 
        # try:
            # assert x.min()>=0 and x.max()<=1   # attn_map: the input is within [0,1] attn_map«
        # except:
            # import ipdb; ipdb.set_trace()

        if self.group_size == 0:
            if mixed_precision is not None:
                n_levels = torch.pow(2,mixed_precision) -  1 # 8bit: -> 255
                # aditional handling of 0-bit, since divide by 0 cause na
                zero_bit_mask = (n_levels != 0).int()
                n_levels[n_levels == 0] = 255  # temporarily set as 8-bit, masked anyway

            # INFO: check abnormally small delta_
            eps = 1.e-6
            try:
                assert torch.all(delta.abs() > eps)
            except:
                # import ipdb; ipdb.set_trace()  
                # safe to set it is eps.
                delta[delta < eps] = eps
                # logger.info("unexpected small delta: {:.3f} exists in attn_map, set as eps".format(delta.abs().min()))

            if mixed_precision is not None:
                delta = delta / n_levels
                x_int = torch.round(x / delta)
                # INFO: the torch.clamp takes single max value, but we want the same shape as x
                x_quant = torch.where(x_int>n_levels, n_levels, x_int)
            else:
                delta = delta/ (self.n_levels*2+1)
                x_int = torch.round(x / delta)
                x_quant = torch.clamp(x_int, 0, self.n_levels*2+1)

            x_dequant = (x_quant) * delta

            if mixed_precision is not None:  # apply the mask of elements of 0-bit
                x_dequant = x_dequant*zero_bit_mask

            return x_dequant
        else:
            assert x.shape[1] % self.group_size == 0, "the input shape should be divisible by group_size"
            n_groups = math.ceil(x.shape[1] / self.group_size)
            x_dequant = torch.zeros_like(x, device=x.device)
            
            for g_idx in range(n_groups):
                g_start = g_idx * self.group_size
                g_end = min((g_idx + 1) * self.group_size, x.shape[1])
                
                g_x = x[:, g_start:g_end]
                
                g_delta = delta[:, g_idx].unsqueeze(-1)  # [G, 1]
                
                if mixed_precision is not None:
                    raise NotImplementedError("mixed precision not supported for group quantization")
                
                eps = 1.e-6
                try:
                    assert torch.all(g_delta.abs() > eps)
                except:
                    g_delta[g_delta.abs() < eps] = eps
                
                g_delta = g_delta / (self.n_levels * 2 + 1)
                g_x_int = torch.round(g_x / g_delta)
                g_x_quant = torch.clamp(g_x_int, 0, self.n_levels * 2 + 1)
                g_x_dequant = g_x_quant * g_delta
                
                x_dequant[:, g_start:g_end] = g_x_dequant
            return x_dequant


class GPTQQuantizer(nn.Module):
    def __init__(self, shape=1):
        super(GPTQQuantizer, self).__init__()
        self.maxq = None
        self.scale = None
        self.zero = None
        
    def configure(
        self,
        bits, perchannel=False, channel_group=1, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        clip_ratio=1.0,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
    
        self.perchannel = perchannel
        self.channel_group = channel_group
        if self.channel_group > 1:
            assert self.perchannel is True, "set perchannel to True when using multiple channel group"
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.clip_ratio = clip_ratio
        if trits:
            self.maxq = torch.tensor(-1)
    
    
    def find_params(self, x, weight=False):
        # import pdb; pdb.set_trace()
        assert len(x.shape) == 2  # [C_out, C_in]
        
        dev = x.device
        
        if self.sym:
            raise NotImplementedError("sym not supported for gptq")
        else:
            x_max = x.max(dim=1)[0]
            x_max = torch.maximum(x_max, torch.zeros_like(x_max))  # set negative values to 0
            
            x_min = x.min(dim=1)[0]
            x_min = torch.minimum(x_min, torch.zeros_like(x_min))  # set positive values to 0
            
        tmp = torch.zeros(x.shape[0], device=dev)
        tmp = (x_min == 0) & (x_max == 0)
        x_min[tmp] = -1
        x_max[tmp] = +1
        if self.maxq < 0:
            raise NotImplementedError("maxq < 0 not supported")
        else:
            # shrink the range based on clip ratio
            self.scale = (x_max - x_min) * self.clip_ratio / self.maxq
            if self.sym:
                self.zero = torch.zeros_like(self.scale, device=dev)
            else:
                self.zero = torch.round(-x_min / self.scale).to(dev)
        
        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            total_iters = int(self.grid * self.maxshrink)
            with tqdm(total=total_iters, desc="GPTQ MSE Search", ncols=80) as pbar:
                for i in range(total_iters):
                    p = 1 - i / self.grid
                    x_min1 = p * x_min
                    x_max1 = p * x_max
                    scale1 = (x_max1 - x_min1) / self.maxq
                    zero1 = torch.round(-x_min1 / scale1) if not self.sym else self.zero
                    q = quantize_gptq(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq, channel_group=self.channel_group)
                    q -= x
                    q.abs_()
                    q.pow_(self.norm)
                    err = torch.sum(q, 1)
                    tmp = err < best
                    if torch.any(tmp):
                        best[tmp] = err[tmp]
                        self.scale[tmp] = scale1[tmp]
                        self.zero[tmp] = zero1[tmp]
                    
                    if i % 10 == 0:
                        best_mean = best.mean().item()
                        pbar.set_postfix_str({
                            'shrink_ratio': f'{p:.3f}',
                            'best_err': f'{best_mean:.6f}',
                        })
                    pbar.update(1)
        
        self.scale = self.scale.unsqueeze(-1)
        self.zero = self.zero.unsqueeze(-1)
        eps = 1.e-6
        try:
            assert torch.all(self.scale.abs() > eps)
            assert torch.isnan(self.scale).sum() == 0
            assert torch.isinf(self.scale).sum() == 0
            assert torch.isnan(self.zero).sum() == 0
            assert torch.isinf(self.zero).sum() == 0
        except:
            import pdb; pdb.set_trace()
            self.scale[self.scale.abs() < eps] = eps
            logger.info("unexpected small scale: {:.3e} exists in {}, set as eps".format(self.scale.abs().min(), self.module_name))
    
    def quantize(self, x):
        if self.ready():
            assert len(self.scale.shape) == 2, "scale should be 2D tensor"
            assert len(self.zero.shape) == 2, "zero should be 2D tensor"
            assert len(x.shape) == 2, "x should be 2D tensor"
            return quantize_gptq(x, self.scale, self.zero, self.maxq, self.channel_group)
        return x
    
    def enabled(self):
        return self.maxq > 0
    
    def ready(self):
        if self.scale is None:
            return False
        return torch.all(self.scale != 0)


def quantize_gptq(x, scale, zero, maxq, channel_group):
    assert len(x.shape) == 2, "only support 2D input"
    assert len(scale.shape) == 2, "only support 2D scale"
    assert len(zero.shape) == 2, "only support 2D zero"
    if maxq < 0:
        raise NotImplementedError("maxq < 0 not supported")
    shape = x.shape
    if channel_group > 1:
        assert len(shape) == 2, "only support 2D input when using multiple channel group"
        shape = x.shape
        x = x.reshape((int(x.shape[0] / channel_group), -1))
    
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    q = scale * (q - zero)
    return q.reshape(shape)