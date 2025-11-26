"""
Example usage of SAM2 entropy processors.

This script demonstrates how to use the entropy-based processors for SAM2.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from processors.encoder.entropy_sam2 import (
    PositionalPruneSAM2Processor,
    HeadPruneSAM2Processor,
    PositionalQuantSAM2Processor,
)


def example_basic_usage():
    """
    Basic example showing how to initialize and use SAM2 processors.
    """
    print("=" * 60)
    print("Example 1: Basic Initialization")
    print("=" * 60)

    # Initialize processors
    positional_prune = PositionalPruneSAM2Processor()
    head_prune = HeadPruneSAM2Processor()
    positional_quant = PositionalQuantSAM2Processor()

    print(f"✓ PositionalPruneSAM2Processor initialized: {positional_prune.strategy_name}")
    print(f"✓ HeadPruneSAM2Processor initialized: {head_prune.strategy_name}")
    print(f"✓ PositionalQuantSAM2Processor initialized: {positional_quant.strategy_name}")
    print()


def example_with_sam2_predictor():
    """
    Example showing how to calibrate with SAM2 predictor.

    Note: This requires SAM2 to be installed and properly configured.
    """
    print("=" * 60)
    print("Example 2: Calibration with SAM2 Predictor")
    print("=" * 60)

    try:
        # Import SAM2 modules
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.modeling.backbones.hieradet import MultiScaleAttention

        print("✓ SAM2 modules imported successfully")

        # This is pseudocode - adjust paths to your actual model
        # predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

        # Initialize processor
        processor = PositionalPruneSAM2Processor()

        # Set parameters (you would get these from your config)
        class MockArgs:
            class Quantization:
                percent_entropy = 0.5
                percent_entropy_global = 0.3
                high_entropy = True
                prune_global = True
            quantization = Quantization()

        processor.set_params(MockArgs())
        print(f"✓ Parameters set: percent={processor.percent}, global_percent={processor.global_percent}")

        # Calibrate on sample data
        # processor.calibrate(predictor, MultiScaleAttention, num_samples=32)

        print("✓ Calibration would run here with actual predictor")
        print()

    except ImportError as e:
        print(f"⚠ SAM2 not available: {e}")
        print("  Install SAM2 to run full calibration")
        print()


def example_entropy_calculation():
    """
    Example showing entropy calculation on sample attention matrices.
    """
    print("=" * 60)
    print("Example 3: Entropy Calculation")
    print("=" * 60)

    processor = PositionalPruneSAM2Processor()

    # Create sample attention matrix (simulating attention weights)
    # Shape: (H*W, H*W) for one head
    seq_len = 64  # e.g., 8x8 spatial dimensions
    attn_matrix = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)

    # Calculate entropy
    entropy = processor.calculate_entropy(attn_matrix)

    print(f"Attention matrix shape: {attn_matrix.shape}")
    print(f"Calculated entropy: {entropy.item():.4f}")
    print(f"Min attention value: {attn_matrix.min().item():.6f}")
    print(f"Max attention value: {attn_matrix.max().item():.6f}")
    print()

    # Compare different attention patterns
    print("Comparing different attention patterns:")

    # Uniform attention (high entropy)
    uniform_attn = torch.ones(seq_len, seq_len) / seq_len
    uniform_entropy = processor.calculate_entropy(uniform_attn)
    print(f"  Uniform attention entropy: {uniform_entropy.item():.4f}")

    # Focused attention (low entropy)
    focused_attn = torch.zeros(seq_len, seq_len)
    focused_attn[:, 0] = 1.0  # All attention on first token
    focused_entropy = processor.calculate_entropy(focused_attn)
    print(f"  Focused attention entropy: {focused_entropy.item():.4f}")
    print()


def example_process_attention():
    """
    Example showing how the process() method works with SAM2 attention.
    """
    print("=" * 60)
    print("Example 4: Processing Attention with Pruning")
    print("=" * 60)

    # Create mock module similar to MultiScaleAttention
    class MockMultiScaleAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 8
            self.dim = 512
            self.qkv = torch.nn.Linear(self.dim, 3 * self.dim)
            self.proj = torch.nn.Linear(self.dim, self.dim)
            self.processor = None

        def forward(self, x):
            if self.processor:
                return self.processor.process(x, self, "test_layer")
            return x

    # Initialize
    module = MockMultiScaleAttention()
    processor = PositionalPruneSAM2Processor()
    module.processor = processor

    # Create sample input: (B, H, W, C)
    B, H, W, C = 1, 8, 8, 512
    x = torch.randn(B, H, W, C)

    print(f"Input shape: {x.shape} (B={B}, H={H}, W={W}, C={C})")
    print(f"Number of attention heads: {module.num_heads}")
    print(f"Head dimension: {C // module.num_heads}")

    # Mock calibration results (simulate having entropy stats)
    processor.final_entropy_stats = {
        "test_layer": torch.tensor([False, False, True, False, True, False, False, True])
    }

    num_pruned = processor.final_entropy_stats["test_layer"].sum().item()
    print(f"Pruning mask: {processor.final_entropy_stats['test_layer'].tolist()}")
    print(f"Number of heads to prune: {num_pruned}/{module.num_heads}")

    # Note: This would fail without proper mock setup, just showing the interface
    print("✓ Process method interface demonstrated")
    print()


def example_key_differences():
    """
    Show key differences between SAM and SAM2 processors.
    """
    print("=" * 60)
    print("Example 5: SAM vs SAM2 Key Differences")
    print("=" * 60)

    print("Architecture Differences:")
    print("  SAM:")
    print("    - Image Encoder: Vision Transformer (ViT)")
    print("    - Attention: Manual Q @ K.T / sqrt(d)")
    print("    - Tensor shape: (B*num_heads, H*W, dim)")
    print("    - Positional encoding: Additive sine/cosine")
    print()
    print("  SAM2:")
    print("    - Image Encoder: Hiera hierarchical backbone")
    print("    - Attention: F.scaled_dot_product_attention (optimized)")
    print("    - Tensor shape: (B, num_heads, H*W, dim)")
    print("    - Positional encoding: Rotary (RoPE)")
    print()

    print("Processor Implementation Differences:")
    print("  ✓ SAM2 computes attention manually in hooks (SDPA is black box)")
    print("  ✓ SAM2 handles different tensor shapes and reshaping")
    print("  ✓ SAM2 uses MultiScaleAttention instead of Attention module")
    print("  ✓ SAM2 processors work with Hiera blocks instead of ViT blocks")
    print()

    print("Module Classes to Use:")
    print("  SAM:  segment_anything.modeling.image_encoder.Attention")
    print("  SAM2: sam2.modeling.backbones.hieradet.MultiScaleAttention")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SAM2 Entropy Processor Examples")
    print("=" * 60 + "\n")

    example_basic_usage()
    example_with_sam2_predictor()
    example_entropy_calculation()
    example_process_attention()
    example_key_differences()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
