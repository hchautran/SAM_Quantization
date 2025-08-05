# w8a8_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

try:
    import w8a8_matmul_ext
    EXTENSION_AVAILABLE = True
except ImportError:
    EXTENSION_AVAILABLE = False
    print("Warning: W8A8 extension not available. Please compile first.")

class W8A8Linear(nn.Module):
    """
    8-bit weight, 8-bit activation linear layer using custom CUDA kernel
    """
    def __init__(self, in_features, out_features, bias=True, use_tensor_cores=True):
        super(W8A8Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_tensor_cores = use_tensor_cores
        
        # Initialize weight and bias as float32, will be quantized during forward
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Quantization scales (learnable parameters)
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('output_scale', torch.tensor(1.0))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def quantize_tensor(self, x, scale=None):
        """Quantize tensor to int8"""
        if scale is None:
            scale = x.abs().max() / 127.0
        x_scaled = x / scale
        x_int8 = torch.clamp(torch.round(x_scaled), -128, 127).to(torch.int8)
        return x_int8, scale
    
    def forward(self, x):
        if not EXTENSION_AVAILABLE:
            # Fallback to regular PyTorch matmul
            return F.linear(x, self.weight, self.bias)
        
        # Ensure input is 2D
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))
        
        # Quantize input and weight
        x_int8, input_scale = self.quantize_tensor(x)
        weight_int8, weight_scale = self.quantize_tensor(self.weight)
        
        # Update scales
        self.input_scale.copy_(input_scale)
        self.weight_scale.copy_(weight_scale)
        
        # Perform W8A8 matrix multiplication
        # Note: PyTorch uses row-major, so we need x @ weight.T
        # Our kernel expects A @ B, so we pass x @ weight_int8.T
        result = w8a8_matmul_ext.w8a8_matmul(
            x_int8, 
            weight_int8.t().contiguous(),  # Transpose weight for correct multiplication
            float(input_scale),
            float(weight_scale), 
            float(self.output_scale),
            self.use_tensor_cores
        )
        
        # Convert back to float and add bias
        result = result.float()
        if self.bias is not None:
            result = result + self.bias.unsqueeze(0)
        
        # Restore original shape
        if len(original_shape) > 2:
            result = result.view(*original_shape[:-1], self.out_features)
        
        return result

class W8A8MLP(nn.Module):
    """Example MLP using W8A8 layers"""
    def __init__(self, input_size, hidden_sizes, output_size, use_tensor_cores=True):
        super(W8A8MLP, self).__init__()
        
        layers = []
        in_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(W8A8Linear(in_size, hidden_size, use_tensor_cores=use_tensor_cores))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        layers.append(W8A8Linear(in_size, output_size, use_tensor_cores=use_tensor_cores))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

def benchmark_w8a8():
    """Benchmark W8A8 vs regular matmul"""
    if not EXTENSION_AVAILABLE:
        print("Extension not available, cannot benchmark")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark on {device}")
    
    # Test different matrix sizes
    sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    
    for M, K, N in sizes:
        print(f"\nMatrix size: {M}x{K} @ {K}x{N}")
        
        # Create test data
        A_float = torch.randn(M, K, device=device)
        B_float = torch.randn(K, N, device=device)
        
        # Quantize to int8
        A_int8 = torch.clamp(torch.round(A_float * 127), -128, 127).to(torch.int8)
        B_int8 = torch.clamp(torch.round(B_float * 127), -128, 127).to(torch.int8)
        
        # Warm up
        for _ in range(10):
            _ = w8a8_matmul_ext.w8a8_matmul(A_int8, B_int8, 1/127.0, 1/127.0, 1.0, True)
            _ = torch.matmul(A_float, B_float)
        
        torch.cuda.synchronize()
        
        # Benchmark W8A8
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            result_w8a8 = w8a8_matmul_ext.w8a8_matmul(A_int8, B_int8, 1/127.0, 1/127.0, 1.0, True)
        torch.cuda.synchronize()
        w8a8_time = (time.time() - start_time) / num_runs
        
        # Benchmark regular matmul
        start_time = time.time()
        for _ in range(num_runs):
            result_float = torch.matmul(A_float, B_float)
        torch.cuda.synchronize()
        float_time = (time.time() - start_time) / num_runs
        
        # Calculate metrics
        gops_w8a8 = (2 * M * N * K) / (w8a8_time * 1e9)
        gops_float = (2 * M * N * K) / (float_time * 1e9)
        speedup = float_time / w8a8_time
        
        print(f"W8A8 time: {w8a8_time*1000:.3f} ms ({gops_w8a8:.1f} GOPS)")
        print(f"Float32 time: {float_time*1000:.3f} ms ({gops_float:.1f} GOPS)")
        print(f"Speedup: {speedup:.2f}x")

def test_w8a8_layer():
    """Test W8A8Linear layer"""
    print("Testing W8A8Linear layer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create layer
    layer = W8A8Linear(256, 128).to(device)
    
    # Test input
    x = torch.randn(32, 256, device=device)  # Batch size 32
    
    # Forward pass
    with torch.no_grad():
        output = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test MLP
    print("\nTesting W8A8MLP...")
    mlp = W8A8MLP(256, [512, 256], 10).to(device)
    
    with torch.no_grad():
        mlp_output = mlp(x)
    
    print(f"MLP output shape: {mlp_output.shape}")
    print(f"MLP output range: [{mlp_output.min():.3f}, {mlp_output.max():.3f}]")

if __name__ == "__main__":
    print("W8A8 Matrix Multiplication PyTorch Extension")
    print(f"Extension available: {EXTENSION_AVAILABLE}")
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA device: {device_name}")
        
        if EXTENSION_AVAILABLE:
            test_w8a8_layer()
            benchmark_w8a8()
    else:
        print("CUDA not available")