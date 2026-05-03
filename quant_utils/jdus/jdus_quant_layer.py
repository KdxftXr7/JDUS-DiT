import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from quant_utils.base.quant_layer import QuantizedLinear
from quant_utils.jdus.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda
import logging
import numpy as np
import scipy.linalg

class JdusQuantizedLinear(QuantizedLinear):
    
    def __init__(self, in_features: int, out_features: int, bias: bool,
                 device: None, quant_config: dict, fp_module: torch.nn.Linear) -> None:
        super().__init__(in_features, out_features, bias, device, quant_config, fp_module)

        self.jdus_matrix = None   # init so could be load in quant_params
        self.logger = logging.getLogger(self.__class__.__name__)
        self.alpha = quant_config.jdus.alpha
        self.svd_threshold = quant_config.jdus.svd_threshold
        self.epsilon_threshold = quant_config.jdus.epsilon_threshold
    
    def get_jdus(self, input_tensor):
        self.input_data = input_tensor
        self.hadamard_matrix = random_hadamard_matrix(self.in_features, "cuda")
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            weight_tensor   = self.weight.data.to(device, dtype=torch.float64)
            X = input_tensor.cpu().numpy().astype(np.float64)
            W = weight_tensor.cpu().numpy().astype(np.float64)
            C_X = np.cov(X, rowvar=False)
            C_W = np.cov(W, rowvar=False)

            eigvals_W, U_W = np.linalg.eigh(C_W)
            eigvals_W_clipped = np.clip(eigvals_W, a_min=np.max(eigvals_W) * self.svd_threshold, a_max=None)
            C_W_inv_clipped = U_W @ np.diag(1.0 / eigvals_W_clipped) @ U_W.T

            epsilon_x = max(self.epsilon_threshold, 10 * self.epsilon_threshold * np.max(np.diag(C_X)))
            C_X_stable = C_X + epsilon_x * np.eye(X.shape[1])

            eigenvalues, G_clipped = scipy.linalg.eigh(C_W_inv_clipped, C_X_stable)
            G_t = torch.from_numpy(G_clipped).cuda()
            X_tmp_t = torch.from_numpy(X).cuda() @ G_t
            W_tmp_t = torch.from_numpy(W).cuda() @ torch.inverse(G_t).T 

            max_X = np.clip(np.percentile(np.abs(X_tmp_t.cpu().numpy()), 99.5, axis=0), a_min=1e-5, a_max=None)
            max_W = np.clip(np.percentile(np.abs(W_tmp_t.cpu().numpy()), 99.5, axis=0), a_min=1e-5, a_max=None)
            S_diag = (max_W ** self.alpha) / (max_X ** self.alpha)
            G_balanced = torch.from_numpy(G_clipped * S_diag[np.newaxis, :]).to(device, dtype=torch.float64)
        self.jdus_matrix = torch.matmul(G_balanced.double(), self.hadamard_matrix)
        self.weight_rotation_matrix = torch.inverse(self.jdus_matrix).t()

    def update_quantized_weight_rotated(self, gptq=False, batch_size=-1):
        self.w_quantizer.init_done = False 
        
        if gptq:
            dtype_ = self.input_data.dtype
            if len(self.input_data.shape) == 3:
                B, N_token, C = self.input_data.shape
                assert N_token % self.q_cfg.calib_batch_size == 0, f"input shape {self.input_data.shape} is not supported"
                self.input_data = self.input_data.reshape([B, self.q_cfg.calib_batch_size, N_token // self.q_cfg.calib_batch_size, C])
                self.input_data = self.input_data.reshape([B * self.q_cfg.calib_batch_size, N_token // self.q_cfg.calib_batch_size, C])
                B, N_token, C = self.input_data.shape
            elif len(self.input_data.shape) == 2:
                if batch_size == -1:
                    B = self.q_cfg.calib_batch_size
                    N_token = self.input_data.shape[0] // B
                    C = self.input_data.shape[1]
                else:
                    B = batch_size
                    N_token = self.input_data.shape[0] // B
                    C = self.input_data.shape[1]
            else:
                raise ValueError(f"input shape {self.input_data.shape} is not supported")
            self.input_data = torch.matmul(self.input_data.double(), self.jdus_matrix.to(self.input_data.device)).to(dtype=dtype_)
            self.input_data = self.input_data.reshape([B * N_token, -1])
            self.input_data = self.input_data.reshape([B, N_token, C])
            self.w_quantizer.gptq_add_batch(self.input_data)
        
        self.weight.data = self.w_quantizer(torch.matmul(self.weight.data.double(), self.weight_rotation_matrix).float())
        self.w_quantizer.init_done = True
        self.input_data = None
        del self.input_data
        torch.cuda.empty_cache()
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        input shape: [B,N_token,C]
        """
        if not self.quant_mode:
            return self.fp_module(x, *args, **kwargs)
        else: 
            dtype_ = x.dtype
            B, N_token, C = x.shape
            x = torch.matmul(x.double(), self.jdus_matrix).to(dtype=dtype_)
            x = x.reshape([B*N_token,-1])
            x = self.a_quantizer(x)
            x = x.reshape([B, N_token, C])
            y = F.linear(x, self.weight.to(x.dtype), self.bias, *args, **kwargs)
            return y