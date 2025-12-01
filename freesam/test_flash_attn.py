"""
Test script for Flash Attention 2 implementation
"""
import torch
import torch.nn.functional as F
import time

# After building: import freesam
# For now, this is a template showing how to use the module


def test_flash_attn_v2():
    """Test Flash Attention 2 forward pass"""
    print("=" * 80)
    print("Testing Flash Attention 2 Implementation")
    print("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration
    batch_size = 2
    num_heads = 8
    seq_len = 512
    head_dim = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping test.")
        return

    # Create random inputs
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    print(f"\nInput shapes:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")

    # Softmax scale
    softmax_scale = 1.0 / (head_dim ** 0.5)

    # Reference implementation using PyTorch
    print("\n" + "-" * 80)
    print("Running PyTorch reference implementation...")
    print("-" * 80)

    torch.cuda.synchronize()
    start = time.time()

    # Reference: standard attention computation
    # scores = (Q @ K^T) / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
    attn_weights = F.softmax(scores, dim=-1)
    output_ref = torch.matmul(attn_weights, V)

    torch.cuda.synchronize()
    ref_time = time.time() - start

    print(f"Reference output shape: {output_ref.shape}")
    print(f"Reference time: {ref_time * 1000:.2f} ms")

    # Flash Attention 2 implementation
    print("\n" + "-" * 80)
    print("Running Flash Attention 2 implementation...")
    print("-" * 80)

    try:
        import freesam

        torch.cuda.synchronize()
        start = time.time()

        output_flash = freesam.flash_attn_v2_forward(Q, K, V, softmax_scale, False)

        torch.cuda.synchronize()
        flash_time = time.time() - start

        print(f"Flash Attention output shape: {output_flash.shape}")
        print(f"Flash Attention time: {flash_time * 1000:.2f} ms")
        print(f"Speedup: {ref_time / flash_time:.2f}x")

        # Compare outputs
        print("\n" + "-" * 80)
        print("Comparing outputs...")
        print("-" * 80)

        max_diff = torch.max(torch.abs(output_ref - output_flash)).item()
        mean_diff = torch.mean(torch.abs(output_ref - output_flash)).item()

        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")

        # Check if outputs are close
        if torch.allclose(output_ref, output_flash, rtol=1e-3, atol=1e-3):
            print("\n✓ Test PASSED: Outputs match within tolerance")
        else:
            print("\n✗ Test FAILED: Outputs differ significantly")

    except ImportError:
        print("freesam module not found. Please build the extension first:")
        print("  cd /home/chauht2/SAM_Quantization/freesam")
        print("  python setup.py install")


def test_causal_attention():
    """Test Flash Attention 2 with causal masking"""
    print("\n" + "=" * 80)
    print("Testing Flash Attention 2 with Causal Masking")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping test.")
        return

    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    device = torch.device('cuda')

    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    softmax_scale = 1.0 / (head_dim ** 0.5)

    # Reference with causal mask
    scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output_ref = torch.matmul(attn_weights, V)

    try:
        import freesam

        output_flash = freesam.flash_attn_v2_forward(Q, K, V, softmax_scale, True)

        max_diff = torch.max(torch.abs(output_ref - output_flash)).item()
        mean_diff = torch.mean(torch.abs(output_ref - output_flash)).item()

        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")

        if torch.allclose(output_ref, output_flash, rtol=1e-3, atol=1e-3):
            print("\n✓ Causal attention test PASSED")
        else:
            print("\n✗ Causal attention test FAILED")

    except ImportError:
        print("freesam module not found. Please build first.")


def benchmark_flash_attn():
    """Benchmark Flash Attention 2 across different sequence lengths"""
    print("\n" + "=" * 80)
    print("Benchmarking Flash Attention 2")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return

    try:
        import freesam
    except ImportError:
        print("freesam module not found. Please build first.")
        return

    batch_size = 1
    num_heads = 8
    head_dim = 64
    seq_lengths = [128, 256, 512, 1024, 2048]

    device = torch.device('cuda')

    print(f"\nConfiguration: batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"\n{'Seq Len':<10} {'Ref (ms)':<12} {'Flash (ms)':<12} {'Speedup':<10} {'Memory (MB)':<12}")
    print("-" * 80)

    for seq_len in seq_lengths:
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

        softmax_scale = 1.0 / (head_dim ** 0.5)

        # Warmup
        for _ in range(5):
            scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
            attn_weights = F.softmax(scores, dim=-1)
            _ = torch.matmul(attn_weights, V)
            _ = freesam.flash_attn_v2_forward(Q, K, V, softmax_scale, False)

        torch.cuda.synchronize()

        # Benchmark reference
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(10):
            scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
            attn_weights = F.softmax(scores, dim=-1)
            _ = torch.matmul(attn_weights, V)
        torch.cuda.synchronize()
        ref_time = (time.time() - start) / 10
        ref_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Benchmark Flash Attention
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(10):
            _ = freesam.flash_attn_v2_forward(Q, K, V, softmax_scale, False)
        torch.cuda.synchronize()
        flash_time = (time.time() - start) / 10
        flash_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        speedup = ref_time / flash_time

        print(f"{seq_len:<10} {ref_time*1000:<12.2f} {flash_time*1000:<12.2f} {speedup:<10.2f}x {flash_mem:<12.2f}")

        # Clean up
        del Q, K, V
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Flash Attention 2 Test Suite")
    print("=" * 80)

    test_flash_attn_v2()
    test_causal_attention()
    benchmark_flash_attn()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
