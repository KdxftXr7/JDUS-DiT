import os
import sys
import logging
import argparse
import math

from scipy import stats

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf, ListConfig
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, sys.path[0] + '/../../')
from diffusion import create_diffusion
from models.models import DiT, DiT_models
from models.download import find_model
from models.quant_dit import QuantDiT
from quant_utils.base.base_quantizer import BaseQuantizer, StaticQuantizer, DynamicQuantizer
from quant_utils.base.quant_layer import QuantizedLinear
from quant_utils.utils import apply_func_to_submodules, seed_everything, setup_logging

def get_uts_calib(activation, norm_weights):
    T, N, C = activation.shape
    token_counts = (norm_weights * N).astype(int)
    remainder = N - token_counts.sum()
    token_counts[np.argmax(token_counts)] += remainder
    sampled_tokens = []
    for t in range(T):
        count = token_counts[t]
        if count > 0:
            indices = torch.randperm(N)[:count]
            sampled_tokens.append(activation[t, indices, :])
    X_cal_complete = torch.cat(sampled_tokens, dim=0)
    final_shuffle = torch.randperm(N)
    X_cal_complete = X_cal_complete[final_shuffle]
    
    return X_cal_complete

def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    if args.log is not None:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
        if not os.path.exists(f"{args.log}/generated_imgs"):
            os.makedirs(f"{args.log}/generated_imgs")
            
    log_file = os.path.join(args.log, 'run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    latent_size = args.image_size // 8
    ptq_config_file = args.ptq_config
    quant_config = OmegaConf.load(ptq_config_file)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    model=QuantDiT(quant_config, ckpt_path, depth=28, hidden_size=1152, patch_size=2,
                   num_heads=16, input_size=latent_size, num_classes=args.num_classes).to(device)

    model.half()   
    if_mixed_precision = isinstance(quant_config.weight.n_bits, ListConfig) or isinstance(quant_config.act.n_bits, ListConfig)
    if if_mixed_precision:
        model.bitwidth_refactor()

    model.eval()  
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    if quant_config.get("jdus",None) is not None:
        from quant_utils.jdus.jdus_quant_layer import JdusQuantizedLinear
        
        def init_jdus_matrix_(module, full_name, save_dir, logger):
            assert isinstance(module, JdusQuantizedLinear)
            
            file_name = full_name.replace('.', '_') + '.pth'
            file_path = os.path.join(save_dir, file_name)
            
            if not os.path.exists(file_path):
                file_name = full_name.replace('.', '_') + '.pt'
                file_path = os.path.join(save_dir, file_name)
            
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} not found, skip.")
                return

            logger.info(f"Loading data for layer {full_name} from {file_path}")
            ptq_calib_data = torch.load(file_path)
        
            batch_size = ptq_calib_data.shape[0]
            uniform_array = np.ones(batch_size) / batch_size
            activation_norm = get_uts_calib(ptq_calib_data, uniform_array)
            
            module.get_jdus(activation_norm)
            
            module.update_quantized_weight_rotated(args.gptq)
            
            del ptq_calib_data
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Layer {full_name} finished.")
        
        assert quant_config.calib_data.save_path is not None
        calib_data_dir = os.path.join(args.log, quant_config.calib_data.save_path)
        
        
        kwargs = {}
        apply_func_to_submodules(
            model,
            class_type=JdusQuantizedLinear,  
            function=init_jdus_matrix_,
            save_dir=calib_data_dir,
            logger=logger,
            full_name='',
            **kwargs
        )
    else:
        raise ValueError("Only support one quantization for now.")

    all_labels = np.arange(args.resume_num, 1000)
    warmup = False
    model.set_init_done()
    using_cfg = args.cfg_scale > 1.0
    for c in all_labels:
        class_labels = [c] * 10 
        
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device, dtype=torch.float16)
        y = torch.tensor(class_labels, device=device)  
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0).contiguous()
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            model_kwargs = dict(y=y)
        z = z.half()
        if not warmup:
            t = torch.tensor([1] * z.shape[0], device=device, dtype=torch.float16).contiguous()
            _ = model(z,t,y)
            warmup = True

        from torch.cuda.amp import autocast
        with autocast():
            samples = diffusion.ddim_sample_loop(
                model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  
        samples = vae.decode(samples / 0.18215).sample

        for i, samples in enumerate(samples):
            save_image(samples, os.path.join(args.log, 'generated_imgs', f'sample_{c}_{i}.png'), normalize=True, value_range=(-1, 1))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument('--ptq-config', default='./configs/config.yaml', type=str)
    parser.add_argument("--log", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quant_param_ckpt", type=str, default="./quant_params.pth")
    parser.add_argument("--hardware", action='store_true', help='whether to use_cuda_kernel')
    parser.add_argument("--profile", action='store_true', help='profile mode, measure the e2e latency')
    parser.add_argument("--quant_weight_ckpt", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--resume_num", type=int, default=0, help="Resume from a class-number")
    parser.add_argument("--gptq", action='store_true', default=False, help="Use gptq")
    args = parser.parse_args()
    main(args)