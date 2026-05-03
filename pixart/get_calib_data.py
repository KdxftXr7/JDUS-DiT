import os
import sys
import time
import logging
import argparse
import shutil

import torch
import torch.nn as nn
import diffusers
from diffusers import PixArtSigmaPipeline
from omegaconf import OmegaConf

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, sys.path[0] + '/../../')
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
        data = module_in[0].reshape([-1, C]).cpu()  # [BS, C]

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


def process_one_layer(pipe, prompts, args, quant_config, save_path, layer_name, layer_module, logger):
    layer_save_file = os.path.join(save_path, f"{layer_name.replace('.', '_')}.pt")

    hook = SaveActivationHook()
    hook_handle = layer_module.register_forward_hook(hook)

    logger.info(f"Start collecting layer {layer_name} activation values...")

    model = pipe.transformer
    N_batch = len(prompts) // args.batch_size  # drop_last

    for i in range(N_batch):
        images = pipe(
            prompt=prompts[i * args.batch_size: (i + 1) * args.batch_size],
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images

    if len(hook.outputs) > 0:
        save_data = torch.stack(hook.outputs, dim=0)
        logger.info(f"Layer {layer_name}, saved activation shape: {save_data.shape}")
        torch.save(save_data, layer_save_file)
    else:
        logger.warning(f"Layer {layer_name} has no activations collected, skipping save.")

    hook_handle.remove()

    hook.clear()
    del hook
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info(f"Layer {layer_name} activation values collection complete, saved to {layer_save_file}")


def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.log is not None:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
    log_file = os.path.join(args.log, 'run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        torch_dtype=torch.float16
    ).to(device)


    model = pipe.transformer
    quant_config = OmegaConf.load(args.quant_config)

    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())

    save_path = os.path.join(args.log, quant_config.calib_data.save_path)
    os.makedirs(save_path, exist_ok=True)

    linear_layers = collect_linear_layers(model)
    logger.info(f"Total {len(linear_layers)} Linear layers collected.")

    for i, (layer_name, layer_module) in enumerate(linear_layers.items()):
        logger.info(f"[{i+1}/{len(linear_layers)}] Processing layer: {layer_name}")

        process_one_layer(
            pipe, prompts, args, quant_config,
            save_path, layer_name, layer_module, logger
        )

        torch.cuda.empty_cache()


    logger.info("All layers activation values collection complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument('--quant-config', required=True, type=str)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)
