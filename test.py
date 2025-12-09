# import sys
# sys.path.insert(0, 'freesam')
import torch
import math
import time
import torch.nn.functional as F

# Import the compiled .so file directly
import freesam.freesam as freesam

# a = torch.rand((256,512), device="cuda")
# b = torch.rand((512,1024), device="cuda")
# c = freesam.gemm(a.half(),b.half(), 2)
# c_true = a@b
# print(c)
# print(a@b)

# print((c - c_true))


Q = torch.randn(25, 16, 196, 64, device='cuda', dtype=torch.float)
K = torch.randn(25, 16, 196, 64, device='cuda', dtype=torch.float)
V = torch.randn(25, 16, 196, 64, device='cuda', dtype=torch.float)


rel_pos_w = torch.rand(25, 16, 196, 14, device='cuda').contiguous()
rel_pos_h = torch.rand(25, 16, 196, 14, device='cuda').contiguous()

def classic_attention(Q, K, V, softmax_scale=0.0):
    # softmax_scale = 1.0 / math.sqrt(head_dim)
    if softmax_scale == 0:
        softmax_scale = math.sqrt(Q.shape[-1])
    Q = Q * softmax_scale
    QK = torch.matmul(Q, K.transpose(-1, -2))
    QK = (QK.view(25, 16, 14, 14, 14, 14) +
          rel_pos_h.view(25, 16, 14, 14, 14)[:,:,:,:,:,None] + 
          rel_pos_w.view(25, 16, 14, 14, 14)[:,:,:,:,None,:]
    ).reshape(25, 16, 196, 196)
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, V)
    return output 

warmup = 5
softmax_scale = math.sqrt(Q.shape[-1])

for i in range(warmup): 
    _ = freesam.flash_attn_rel(Q, K, V, rel_pos_h, rel_pos_w, softmax_scale )

start = time.time()
output = freesam.flash_attn_rel(Q, K, V, rel_pos_h, rel_pos_w, softmax_scale)
flash_inference_time = time.time() - start
print('flash', flash_inference_time)

for i in range(warmup): 
    _ = classic_attention(Q, K, V, softmax_scale=softmax_scale)


start = time.time()
cls_output = classic_attention(Q, K, V, softmax_scale=softmax_scale)
classic_inference_time = time.time() - start
print('classic ', classic_inference_time)
print(cls_output.dtype)
print(output.dtype)
print(torch.isnan(output).any())
print('speedup:', f'x{classic_inference_time/flash_inference_time}' )
assert torch.allclose(output, cls_output, atol=1e-4)




