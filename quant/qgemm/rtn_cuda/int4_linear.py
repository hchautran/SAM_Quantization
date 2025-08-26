import torch
import torch.nn as nn
import qgemm 
import time

@torch.inference_mode()
def sym_quant_rowwise(x: torch.Tensor):
    assert x.dim() == 2 and x.dtype == torch.float16 and x.is_cuda
    start_time = time.perf_counter()
    # scale = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / 7.0
    scale = torch.randn((x.size(0), 1), device=x.device, dtype=torch.float16)
    # end1= time.perf_counter()

    q = qgemm.sym_quant(x, scale)
    # end2 = time.perf_counter()
   
    return q, scale


class Int4Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('qweight', torch.empty(0))
        self.register_buffer('scale', torch.empty(0))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def quantize_weights(self, weight):
        # weight shape: [out_features, in_features] = [N, K]
        assert weight.dim() == 2 and weight.dtype == torch.float16 and weight.is_cuda
        qweight, scale = sym_quant_rowwise(weight)
        self.qweight = qweight  # shape [N, K//2] (packed)
        self.scale = scale.squeeze(-1)  # shape [N] (remove singleton dim)
        return self
        
    def forward(self, x):

        x_q, x_scale = sym_quant_rowwise(x)
        # matmul: x_q[M, K//2] @ qweight[N, K//2] -> output[M, N] (INT32)
        output = qgemm.matmul(x_q, self.qweight)

        # Dequantize using the C++ signature: (q, scale_row, scale_col, bits)
        # scale_row: [M], scale_col: [N]
     
        x_scale_flat = x_scale.squeeze(-1)  # [M]
        output = qgemm.sym_dequant(output, x_scale_flat, self.scale, 32)

        if self.bias is not None:
            output += self.bias

        return output
    """
    x --> int 4
    x_int 4 * w_int4 --> int32
    int32 --> float32
    
    """
   