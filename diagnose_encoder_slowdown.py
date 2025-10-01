"""
Diagnostic script to identify encoder quantization slowdown issues
"""
import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from encoder_quant import image_encoder_monkey_patch
from quant_utils import ImageEncoderProcessor
from segment_anything.modeling.image_encoder import Block
from profiler import InferenceProfiler, get_profiler, profile
import train.utils.misc as misc
from data_utils import OnlineDataset
from train.utils.dataloader import get_im_gt_name_dict, Resize
from torch.utils.data import DataLoader
from torchvision import transforms


def get_sample_data():
    """Get a sample image for testing"""
    dataset_dis = {
        "name": "DIS5K-VD",
        "im_dir": "./data/DIS5K/DIS-VD/im",
        "gt_dir": "./data/DIS5K/DIS-VD/gt",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    valid_im_gt_list = get_im_gt_name_dict([dataset_dis], flag="valid")
    gos_dataset = OnlineDataset(
        [valid_im_gt_list[0]],
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=True
    )
    dataloader = DataLoader(gos_dataset, 1, drop_last=False)

    for data_val in dataloader:
        imgs = data_val['image'].permute(0, 2, 3, 1).cpu().numpy().squeeze()
        labels_boxes = misc.masks_to_boxes(data_val['label'][:, 0, :, :]).cpu().numpy()
        return imgs, labels_boxes

    return None, None


def test_baseline_speed(predictor, image, num_runs=20):
    """Test baseline (non-quantized) inference speed"""
    print("\n" + "="*80)
    print("BASELINE (No Quantization)")
    print("="*80)

    # Warmup
    for _ in range(5):
        predictor.set_image(image)

    # Time encoder only
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        predictor.set_image(image)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Encoder (set_image): {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {1000.0/mean_time:.2f} images/sec")

    return mean_time


def test_quantized_encoder_speed(predictor, processor, image, n_bits=8, num_runs=20):
    """Test quantized encoder inference speed"""
    print("\n" + "="*80)
    print(f"QUANTIZED ENCODER ({n_bits}-bit)")
    print("="*80)

    # Apply quantization
    image_encoder_monkey_patch(
        predictor.model,
        processor=processor,
        n_bits=n_bits,
        weight_quant='per_channel',
        k_preserve=0
    )

    # Warmup
    for _ in range(5):
        predictor.set_image(image)

    # Time encoder
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        predictor.set_image(image)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Encoder (set_image): {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {1000.0/mean_time:.2f} images/sec")

    return mean_time


def test_quantized_encoder_no_processor(predictor, image, n_bits=8, num_runs=20):
    """Test quantized encoder WITHOUT processor (just quantized linear layers)"""
    print("\n" + "="*80)
    print(f"QUANTIZED ENCODER WITHOUT PROCESSOR ({n_bits}-bit)")
    print("="*80)

    # Apply quantization without processor
    image_encoder_monkey_patch(
        predictor.model,
        processor=None,  # No processor
        n_bits=n_bits,
        weight_quant='per_channel',
        k_preserve=0
    )

    # Warmup
    for _ in range(5):
        predictor.set_image(image)

    # Time encoder
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        predictor.set_image(image)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"Encoder (set_image): {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {1000.0/mean_time:.2f} images/sec")

    return mean_time


def diagnose_encoder_attention(predictor, image):
    """Profile individual components of encoder attention"""
    print("\n" + "="*80)
    print("DETAILED ENCODER ATTENTION PROFILING")
    print("="*80)

    # Patch with instrumented forward pass
    from segment_anything.modeling.image_encoder import Attention

    original_forward = Attention.forward

    def profiled_forward(self, x):
        with profile("attn_qkv_proj"):
            B, H, W, _ = x.shape
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        with profile("attn_matmul"):
            attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            with profile("attn_rel_pos"):
                from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
                attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        with profile("attn_softmax"):
            attn = attn.softmax(dim=-1)

        with profile("attn_v_matmul"):
            x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

        with profile("attn_proj"):
            x = self.proj(x)

        return x

    # Monkey patch
    Attention.forward = profiled_forward

    # Run a few iterations
    profiler = get_profiler()
    profiler.clear()

    for _ in range(10):
        predictor.set_image(image)

    # Restore original
    Attention.forward = original_forward

    # Print results
    profiler.print_summary()


def main():
    """Main diagnostic function"""
    print("SAM Encoder Quantization Slowdown Diagnostic")
    print("=" * 80)

    # Load model
    model_type = 'vit_l'
    checkpoint_path = './pretrained_checkpoint/sam_hq_vit_l.pth'
    print(f"Loading model: {model_type}")

    # Get sample data
    image, box = get_sample_data()
    if image is None:
        print("ERROR: Could not load sample data")
        return

    print(f"Image shape: {image.shape}")

    # Test 1: Baseline
    sam_baseline = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor_baseline = SamPredictor(sam_baseline)
    baseline_time = test_baseline_speed(predictor_baseline, image)

    # Test 2: Quantized with processor
    sam_quant = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor_quant = SamPredictor(sam_quant)

    print("\nCalibrating processor...")
    processor = ImageEncoderProcessor('encoder_attn')
    # Note: Processor calibration is disabled in the current implementation
    # So this should not cause slowdown

    quant_time = test_quantized_encoder_speed(predictor_quant, processor, image)

    # Test 3: Quantized without processor
    sam_quant_no_proc = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor_quant_no_proc = SamPredictor(sam_quant_no_proc)
    quant_no_proc_time = test_quantized_encoder_no_processor(predictor_quant_no_proc, image)

    # Test 4: Detailed profiling
    sam_detailed = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor_detailed = SamPredictor(sam_detailed)
    diagnose_encoder_attention(predictor_detailed, image)

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"Baseline:                    {baseline_time:.2f} ms")
    print(f"Quantized (with processor):  {quant_time:.2f} ms  (speedup: {baseline_time/quant_time:.2f}x)")
    print(f"Quantized (no processor):    {quant_no_proc_time:.2f} ms  (speedup: {baseline_time/quant_no_proc_time:.2f}x)")
    print()

    if quant_time > baseline_time:
        print("❌ SLOWDOWN DETECTED!")
        print(f"   Quantized encoder is {quant_time/baseline_time:.2f}x SLOWER")
        print("\nPossible causes:")
        print("1. Extra reshape operations in AttentionObserver.forward()")
        print("2. Processor overhead (even when disabled)")
        print("3. Quantized linear layers slower than expected")
        print("4. CUDA kernel launch overhead")
    else:
        print("✅ No slowdown detected")

    print("="*80)


if __name__ == '__main__':
    main()