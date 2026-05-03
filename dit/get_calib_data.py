import os
import sys
import logging
import argparse
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, sys.path[0] + '/../../')
from diffusion import create_diffusion
from models.models import DiT, DiT_models, DiT_XL_2
from models.download import find_model
from quant_utils.utils import apply_func_to_submodules, seed_everything, setup_logging


class SaveActivationHook:
    def __init__(self):
        self.hook_handle = None
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        '''
        the input shape could be [BS, C] or [BS, N_token, C]
        only keep the channel_dim for reduced saved act size
        '''
        C = module_in[0].shape[-1]
        data = module_in[0].reshape([-1,C]).cpu()  # [BS, C]
        
        self.outputs.append(data)
    
    def clear(self):
        self.outputs = []


def add_hook_to_module_(module, hook_cls):
    hook = hook_cls()
    hook.hook_handle = module.register_forward_hook(hook)
    return hook


def collect_linear_layers(model):
    linear_layers = {}
    
    def _collect(name, module):
        if isinstance(module, nn.Linear):
            linear_layers[name] = module
    
    for name, module in model.named_modules():
        _collect(name, module)
        
    return linear_layers


def process_one_layer(latent_size, device, using_cfg, diffusion, model, vae, save_path, layer_name, layer_module, logger):
    layer_save_dir = os.path.join(save_path, f"{layer_name.replace('.', '_')}.pth")
    
    hook = SaveActivationHook()
    hook_handle = layer_module.register_forward_hook(hook)
    
    logger.info(f"start collecting layer {layer_name} activation values...")
    
    run_inference(latent_size, device, using_cfg, diffusion, model, vae)
    
    save_data = torch.stack(hook.outputs, dim=0)
    logger.info(f"layer_name: {layer_name}, hook_input_shape: {save_data.shape}")
    torch.save(save_data, layer_save_dir)
    
    hook_handle.remove()
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logger.info(f"layer {layer_name} activation values collection complete")
    
    
def run_inference(latent_size, device, using_cfg, diffusion, model, vae):
    class_labels = random.sample(range(1000), 50)
    class_labels = np.array(class_labels)
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    if using_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    else:
        model_kwargs = dict(y=y)
    z = z.half()

    from torch.cuda.amp import autocast
    with autocast():
        samples = diffusion.ddim_sample_loop(
            model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,
        )
    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample


def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"
    using_cfg = args.cfg_scale > 1.0

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    if args.log is not None:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
    log_file = os.path.join(args.log, 'run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    latent_size = args.image_size // 8
    ptq_config_file = args.ptq_config
    quant_config = OmegaConf.load(ptq_config_file)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    model = DiT_XL_2(input_size=latent_size, num_classes=args.num_classes).to(device)

    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    linear_layers = collect_linear_layers(model)

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    
    save_path = os.path.join(args.log, quant_config.calib_data.save_path)
    os.makedirs(save_path, exist_ok=True)


    for i in range(0, len(linear_layers), 1):
        layer_name = list(linear_layers.keys())[i]
        layer_module = list(linear_layers.values())[i]
        
        process_one_layer(
            latent_size, device, using_cfg, diffusion, model, vae,
            save_path, layer_name, layer_module, logger
        )
        
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logger.info("all layers activation values collection complete")
    

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
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)