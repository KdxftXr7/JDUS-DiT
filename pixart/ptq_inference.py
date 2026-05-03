import os
import sys
import time
import logging
import argparse
import shutil
import math

import numpy as np
from scipy import stats
import torch
import diffusers
from omegaconf import OmegaConf, ListConfig

from models.customize_pipeline_pixart_sigma import CustomizePixArtSigmaPipeline
from models.customize_pipeline_pixart_transformer_2d import CustomizePixArtTransformer2DModel
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, sys.path[0] + '/../../')
from quant_utils.utils import apply_func_to_submodules, seed_everything, setup_logging
diffusers.models.PixArtTransformer2DModel = CustomizePixArtTransformer2DModel
diffusers.PixArtSigmaPipeline = CustomizePixArtSigmaPipeline
from diffusers import PixArtSigmaPipeline

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

    if args.log is not None:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
        if not os.path.exists(f"{args.log}/generated_images"):
            os.makedirs(f"{args.log}/generated_images")
    log_file = os.path.join(args.log, 'run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        torch_dtype=torch.float16
    ).to(device)

    quant_config = OmegaConf.load(args.quant_config)
    pipe.convert_quant(quant_config)
    model = pipe.transformer
    
    def init_jdus_matrix_(module, full_name, save_dir, logger):
            assert isinstance(module, JdusQuantizedLinear)
            
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

            
    if quant_config.get("jdus", None) is not None:
        from quant_utils.jdus.jdus_quant_layer import JdusQuantizedLinear
        
        assert quant_config.calib_data.save_path is not None
        calib_data_dir = os.path.join(args.log, quant_config.calib_data.save_path)
        
        metadata_file = os.path.join(calib_data_dir, 'metadata.pt')
        if not os.path.exists(metadata_file):
            logger.error(f"Metadata file {metadata_file} not found.")
            raise FileNotFoundError(f"Metadata file {metadata_file} not found.")
        
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
        raise NotImplementedError("Only JDUS is supported for now")
    
    model.set_init_done()
    
    pipe.model = model

    logger.info(str(model))

    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())
                    
    N_batch = len(prompts) // args.batch_size # drop_last
    for i in range(N_batch):
        images = pipe(
            prompt=prompts[i*args.batch_size: (i+1)*args.batch_size],
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images
        print(f"Export image of batch {i}")

        save_path = os.path.join(args.log, "generated_images")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for i_image in range(args.batch_size):
            images[i_image].save(os.path.join(save_path, f"output_{i_image + args.batch_size*i}.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument('--quant-config', required=True, type=str)
    parser.add_argument("--quant_param_ckpt", type=str, default="./quant_params.pth")
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hardware", action='store_true', help='whether to use_cuda_kernel')
    parser.add_argument("--profile", action='store_true', help='profile mode, measure the e2e latency')
    parser.add_argument("--quant_weight_ckpt", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--smooth_quant", action='store_true', help="Use smooth quantization")
    parser.add_argument("--gptq", action='store_true', default=False, help="Use gptq quantization")
    args = parser.parse_args()
    main(args)