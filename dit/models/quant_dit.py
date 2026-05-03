"""
conduct model PTQ process
take in the orginal model and the calib data
save the quantized model checkpoint
"""
import os
import sys
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from omegaconf import OmegaConf
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, sys.path[0] + '/../../')
from models.models import DiT, DiT_models
from models.download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from quant_utils.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from quant_utils.base.quant_layer import QuantizedLinear
from quant_utils.base.quant_model import quant_layer_refactor_, bitwidth_refactor_, load_quant_param_dict_, save_quant_param_dict_, set_init_done_
from quant_utils.utils import apply_func_to_submodules


logger = logging.getLogger(__name__)


# ------- some layers with cuda kernel forward ---------
# from viditq_extension.nn.qlinear import W8A8OF16LinearDynamicInputScale
# from viditq_extension.nn.layernorm import LayerNormGeneral
# import viditq_extension.fused as fused_kernels


def quantize_and_save_weight_(submodule, full_name):
    fp_weight = submodule.fp_module.weight.to(torch.float16)
    # the viditq_extension.nn.qlinear use [C] as the scale shape, but the qdiff simulation code use [C, 1]

    submodule.w_quantizer.delta = submodule.w_quantizer.delta.view(-1).to(torch.float16)
    submodule.w_quantizer.zero_point = submodule.w_quantizer.zero_point.view(-1).to(torch.float16)
    scale = submodule.w_quantizer.delta
    zero_point = submodule.w_quantizer.zero_point  # the cuda kernel code uses 128+zero_point

    # INFO: the orginal module weight is the FP16 quantized dequant weight, 
    # replace with INT weight, should update the state_dict
    int_weight = torch.clamp(
            torch.round(fp_weight / scale.view(-1,1)) - zero_point.view(-1,1),
            -128, 127).to(torch.int8)  # kernel supports W8A8 only for now
    submodule.weight.data = int_weight

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class QuantDiT(DiT):
    def __init__(self, quant_config: dict, ckpt_path, input_size=32, patch_size=2, in_channels=4,
                 hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1,
                 num_classes=1000, learn_sigma=True, **kwargs):
        super().__init__(input_size, patch_size, in_channels, hidden_size, depth, num_heads,
                         mlp_ratio, class_dropout_prob, num_classes, learn_sigma)

        state_dict = find_model(ckpt_path)
        self.quant_config=quant_config
        self.load_state_dict(state_dict)
        self.quant_param_dict = {}
        self.quant_layer_refactor()
    
    def quant_layer_refactor(self):
        apply_func_to_submodules(self, class_type=nn.Linear, function=quant_layer_refactor_,
                                 name=None, parent_module=None, quant_config=self.quant_config,
                                 full_name=None, remain_fp_regex=self.quant_config.remain_fp_regex if hasattr(self.quant_config, 'remain_fp_regex') else None)
    
    def save_quant_param_dict(self):
        apply_func_to_submodules(self, class_type=BaseQuantizer, function=save_quant_param_dict_,
                                 full_name=None, parent_module=None, model=self)
    
    def load_quant_param_dict(self, quant_param_dict):
        apply_func_to_submodules(self, class_type=BaseQuantizer, function=load_quant_param_dict_,
                                 full_name=None, parent_module=None, quant_param_dict=quant_param_dict, model=self)
    
    def set_init_done(self):
        apply_func_to_submodules(self, class_type=BaseQuantizer, function=set_init_done_)
    
    # ------ used for infer with CUDA kernel ------- 
    
    def quantize_and_save_weight(self, save_path):
        # set require_grad=False, since torch force the variable to be FP or complex (we assign them as torch.int8)
        for param in self.parameters():
            param.requires_grad_(False)

        # iter through all QuantLayers and get quantized INT, fill into the state_dict
        apply_func_to_submodules(self,
                class_type=QuantizedLinear,
                function=quantize_and_save_weight_,
                full_name=None,
                )
        # delete the quant_params and fp_weights in the state_dict
        sd = self.state_dict()
        keys_to_delte = ['fp_weight','fp_module']
        keys_to_rename = {
                'w_quantizer.delta': 'scale_weight',
                'w_quantizer.zero_point': 'zp_weight',
                }
        for k in list(sd.keys()):
            if any(substring in k for substring in keys_to_delte):
                del sd[k]
            if any(substring in k for substring in keys_to_rename):
                original_k = k
                for substring in keys_to_rename.keys():
                    if substring in k:
                        k = k.replace(substring, keys_to_rename[substring])
                sd[k] = sd.pop(original_k)

        # IMPORTANT: modify the zero_point since cuda kernel use 128 as the base zero_point instead of 0.
        # FIXED in PTQ
        # for k in list(sd.keys()):
            # if 'zp_weight' in k:
                # sd[k] = (sd[k] + 128).to(torch.int16)

        # INFO: we implement the "general" version of layernorm
        # with the affine transform (wx+b) fused after the layernorm operation
        # so the vanilla layernorm has w=1 and b=0
        n_block = len(self.blocks)
        for i_block in range(n_block):
            hidden_size = self.blocks[i_block].mlp.fc1.in_features
            sd['blocks.{}.norm1.weight'.format(i_block)] = torch.ones((hidden_size,), dtype=torch.float16)
            sd['blocks.{}.norm2.weight'.format(i_block)] = torch.ones((hidden_size,), dtype=torch.float16)

        logger.info('Remaining Keys')
        for k in list(sd.keys()):
            logger.info('key: {},  {}'.format(k, sd[k].dtype))
        torch.save(sd, save_path)

        logger.info("\nFinished Saving the Quantized Checkpoint...\n")
        
    # ------ used for infer with mixed precision ------- 
    def bitwidth_refactor(self):
        apply_func_to_submodules(self, class_type=QuantizedLinear, function=bitwidth_refactor_,
                                 name=None, parent_module=None, quant_config=self.quant_config, full_name=None)