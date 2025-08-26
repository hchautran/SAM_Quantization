import torch
import torch.nn as nn
import sys
import os
import time
import random
import numpy as np
# Add qgemm directory to Python path
qgemm_dir = "/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/quant/qgemm"
sys.path.insert(0, qgemm_dir)

from int4_linear import Int4Linear
def set_seed(seed=42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    # torch.cuda.manual_seed_all(seed)  # PyTorch (all GPUs)
    # torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    # torch.backends.cudnn.benchmark = False 
set_seed(42)
# Create a simple dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 512)
        # self.layer2 = nn.Linear(256, 128) 
        self.layer3 = nn.Linear(512, 64)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        # x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
def replace_linear_with_int4(model, exclude_modules=None):
    """Replace nn.Linear with Int4Linear and quantize weights"""
    if exclude_modules is None:
        exclude_modules = []
    
    for name, module in model.named_children():
        if any(exclude in name for exclude in exclude_modules):
            continue
            
        if isinstance(module, nn.Linear):
            print(f"Replacing {name}: {module}")
            
            # Create Int4Linear replacement
            int4_layer = Int4Linear(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None
            )
            
            # Move Int4Linear to GPU FIRST
            int4_layer = int4_layer.cuda()
            
            # Quantize weights
            with torch.no_grad():
                weight_fp16 = module.weight.to(torch.float16).cuda()
                int4_layer.quantize_weights(weight_fp16)
                
                if module.bias is not None:
                    # Ensure bias is on GPU and correct dtype
                    int4_layer.bias.data = module.bias.to(torch.float16).cuda()
            
            # Replace the module
            setattr(model, name, int4_layer)
            print(f"  -> Replaced with Int4Linear (moved to GPU)")
        else:
            # Recursively process child modules
            replace_linear_with_int4(module, exclude_modules)

def benchmark_model(model, input_tensor, name, warmup=10, iters=50):
    """Benchmark model inference time"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Timing
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            output = model(input_tensor)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_time_ms = (end_time - start_time) * 1000.0 / iters
    print(f"{name} average inference time: {avg_time_ms:.3f} ms")
    
    return output, avg_time_ms

def test_int4_replacement():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy model
    original_model = DummyModel().to(device)
    print("Original model:")
    print(original_model)
    
    # Create random input - IMPORTANT: Use float16 for quantized model
    batch_size = 4
    input_tensor_fp32 = torch.randn(batch_size, 1024, dtype=torch.float32, device=device)  # FP32 input for original model
    input_tensor_fp16 = input_tensor_fp32.to(torch.float16)  # Convert to FP16
    
    print(f"\nInput tensor shapes:")
    print(f"  FP32 input: {input_tensor_fp32.shape}, dtype: {input_tensor_fp32.dtype}")
    print(f"  FP16 input: {input_tensor_fp16.shape}, dtype: {input_tensor_fp16.dtype}")
    
    # Test original model with FP32 input
    print("\n" + "="*60)
    print("TESTING ORIGINAL MODEL (FP32)")
    print("="*60)
    
    original_output, original_time = benchmark_model(
        original_model, input_tensor_fp32, "Original FP32 Model"
    )
    print(f"Original output shape: {original_output.shape}")
    print(f"Original output sample: {original_output[0, :5]}")
    
    # Create a copy for quantization
    quantized_model = DummyModel().to(device)
    quantized_model.load_state_dict(original_model.state_dict())
    
    # Replace with Int4Linear
    print("\n" + "="*60)
    print("REPLACING WITH INT4 LAYERS")
    print("="*60)
    replace_linear_with_int4(quantized_model)
    
    print("\nQuantized model:")
    print(quantized_model)
    
    # Test quantized model with FP16 input
    print("\n" + "="*60)
    print("TESTING QUANTIZED MODEL (FP16)")
    print("="*60)
    
    quantized_output, quantized_time = benchmark_model(
        quantized_model, input_tensor_fp16, "Quantized INT4 Model"
    )
    print(f"Quantized output shape: {quantized_output.shape}")
    print(f"Quantized output sample: {quantized_output[0, :5]}")
    
    # Compare outputs (convert to same dtype for comparison)
    print("\n" + "="*60)
    print("ACCURACY COMPARISON")
    print("="*60)
    original_output_fp16 = original_output.to(torch.float16)
    diff = (original_output_fp16 - quantized_output).abs()
    
    print(f"Output difference (FP16 comparison):")
    print(f"  Max absolute error: {diff.max().item():.6f}")
    print(f"  Mean absolute error: {diff.mean().item():.6f}")
    print(f"  Relative error: {(diff.mean() / original_output_fp16.abs().mean()).item():.6f}")
    
    # Performance comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    speedup = original_time / quantized_time
    print(f"Original model time:  {original_time:.3f} ms")
    print(f"Quantized model time: {quantized_time:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 1:
        print("🚀 Quantized model is FASTER!")
    else:
        print("⚠️  Original model is faster (may need larger models to see INT4 benefits)")
if __name__ == "__main__":
    test_int4_replacement()