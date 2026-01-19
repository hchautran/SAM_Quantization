"""
Profiling utilities for SAM quantization analysis
"""
import time
import torch
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class Timer:
    """Simple timer context manager"""

    def __init__(self, name: str = "Operation", sync_cuda: bool = True):
        self.name = name
        self.sync_cuda = sync_cuda
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time


class ProfilerStats:
    """Collect and analyze profiling statistics"""

    def __init__(self):
        self.timings = defaultdict(list)
        self.enabled = True

    def record(self, name: str, elapsed: float):
        """Record a timing measurement"""
        if self.enabled:
            self.timings[name].append(elapsed)

    def get_stats(self, name: str) -> Dict:
        """Get statistics for a specific timer"""
        if name not in self.timings or len(self.timings[name]) == 0:
            return {}

        times = np.array(self.timings[name])
        return {
            'count': len(times),
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'total': np.sum(times),
        }

    def get_all_stats(self) -> pd.DataFrame:
        """Get statistics for all timers as DataFrame"""
        stats_list = []
        for name in sorted(self.timings.keys()):
            stats = self.get_stats(name)
            if stats:
                stats['name'] = name
                stats_list.append(stats)

        if stats_list:
            df = pd.DataFrame(stats_list)
            df = df[['name', 'count', 'mean', 'std', 'min', 'median', 'max', 'total']]
            return df
        return pd.DataFrame()

    def print_summary(self, sort_by: str = 'total'):
        """Print profiling summary"""
        df = self.get_all_stats()
        if df.empty:
            print("No profiling data collected")
            return

        # Sort by specified column
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=False)

        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        print("="*80 + "\n")

    def clear(self):
        """Clear all collected timings"""
        self.timings.clear()

    def disable(self):
        """Disable profiling"""
        self.enabled = False

    def enable(self):
        """Enable profiling"""
        self.enabled = True


# Global profiler instance
_global_profiler = ProfilerStats()


def get_profiler() -> ProfilerStats:
    """Get the global profiler instance"""
    return _global_profiler


@contextmanager
def profile(name: str, profiler: Optional[ProfilerStats] = None, sync_cuda: bool = True):
    """
    Context manager for profiling code blocks

    Usage:
        with profile("encoder_forward"):
            output = encoder(x)
    """
    if profiler is None:
        profiler = _global_profiler

    timer = Timer(name, sync_cuda=sync_cuda)
    with timer:
        yield timer

    profiler.record(name, timer.elapsed)


class InferenceProfiler:
    """Profile SAM inference with detailed breakdown"""

    def __init__(self, sync_cuda: bool = True):
        self.profiler = ProfilerStats()
        self.sync_cuda = sync_cuda

    def profile_sam_inference(self, predictor, image: np.ndarray,
                             input_box: Optional[np.ndarray] = None,
                             num_runs: int = 10, warmup: int = 3):
        """
        Profile SAM inference with detailed timing breakdown

        Args:
            predictor: SamPredictor instance
            image: Input image
            input_box: Optional bounding box
            num_runs: Number of profiling runs
            warmup: Number of warmup runs

        Returns:
            Dictionary with profiling results
        """
        print(f"Warming up for {warmup} iterations...")
        for _ in range(warmup):
            predictor.set_image(image)
            if input_box is not None:
                predictor.predict(box=input_box)

        print(f"Profiling for {num_runs} iterations...")
        self.profiler.clear()

        for i in range(num_runs):
            # Profile set_image (encoder)
            with profile("set_image", self.profiler, self.sync_cuda):
                predictor.set_image(image)

            if input_box is not None:
                # Profile predict (decoder)
                with profile("predict", self.profiler, self.sync_cuda):
                    masks, scores, logits = predictor.predict(box=input_box)

                # Profile end-to-end
                with profile("end_to_end", self.profiler, self.sync_cuda):
                    predictor.set_image(image)
                    predictor.predict(box=input_box)

        return self.get_results()

    def get_results(self) -> Dict:
        """Get profiling results as dictionary"""
        df = self.profiler.get_all_stats()
        if df.empty:
            return {}

        results = {}
        for _, row in df.iterrows():
            results[row['name']] = {
                'mean': row['mean'],
                'std': row['std'],
                'throughput': 1.0 / row['mean'] if row['mean'] > 0 else 0
            }

        return results

    def print_results(self):
        """Print profiling results"""
        self.profiler.print_summary(sort_by='mean')

        results = self.get_results()
        if results:
            print("\nThroughput:")
            for name, stats in results.items():
                print(f"  {name}: {stats['throughput']:.2f} ops/sec")


def compare_inference_speed(predictor_baseline, predictor_quantized,
                            image: np.ndarray, input_box: Optional[np.ndarray] = None,
                            num_runs: int = 20):
    """
    Compare inference speed between baseline and quantized models

    Args:
        predictor_baseline: Baseline SamPredictor
        predictor_quantized: Quantized SamPredictor
        image: Input image
        input_box: Optional bounding box
        num_runs: Number of runs

    Returns:
        Comparison dictionary
    """
    profiler_baseline = InferenceProfiler()
    profiler_quantized = InferenceProfiler()

    print("="*80)
    print("BASELINE MODEL")
    print("="*80)
    results_baseline = profiler_baseline.profile_sam_inference(
        predictor_baseline, image, input_box, num_runs=num_runs
    )
    profiler_baseline.print_results()

    print("\n" + "="*80)
    print("QUANTIZED MODEL")
    print("="*80)
    results_quantized = profiler_quantized.profile_sam_inference(
        predictor_quantized, image, input_box, num_runs=num_runs
    )
    profiler_quantized.print_results()

    # Comparison
    print("\n" + "="*80)
    print("SPEEDUP COMPARISON")
    print("="*80)

    comparison = {}
    for key in results_baseline.keys():
        if key in results_quantized:
            baseline_time = results_baseline[key]['mean']
            quantized_time = results_quantized[key]['mean']
            speedup = baseline_time / quantized_time if quantized_time > 0 else 0

            comparison[key] = {
                'baseline_ms': baseline_time * 1000,
                'quantized_ms': quantized_time * 1000,
                'speedup': speedup,
                'slower': speedup < 1.0
            }

            status = "SLOWER" if speedup < 1.0 else "FASTER"
            print(f"{key:20s}: {baseline_time*1000:7.2f}ms -> {quantized_time*1000:7.2f}ms "
                  f"(speedup: {speedup:.2f}x) [{status}]")

    print("="*80 + "\n")

    return comparison