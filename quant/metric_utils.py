import os
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
from functools import partial
import json
from fvcore.nn import FlopCountAnalysis
import torch.nn.functional as F


class MetricCalculator:
    """
    Comprehensive metric calculator for SAM models based on existing utility files.
    Measures GFLOPs, accuracy metrics (mIoU, AP, AR), latency, and peak memory usage.
    """
    
    def __init__(self):
        self.reset()
        self.hooks = []
        
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_flops = 0
        self.total_params = 0
        self.peak_memory = 0
        self.inference_times = []
        self.ious = []
        self.boundary_ious = []
        self.aps = []
        self.ars = []
        self.module_times = defaultdict(list)
        self.module_flops = defaultdict(float)
        
    def clear_hooks(self):
        """Clear all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []



class LatencyCalculator(MetricCalculator):
    """Calculate inference latency based on latency_utils.py"""
    
    def __init__(self, target_modules: Tuple = None):
        super().__init__()
        self.target_modules = target_modules or (nn.Linear, nn.Conv2d, nn.MultiheadAttention)
        self.start_times = {}
        
    def register_latency_hooks(self, model: nn.Module):
        """Register hooks to measure module-wise latency."""
        def pre_hook(module: nn.Module, input, name):
            if isinstance(module, self.target_modules):
                self.start_times[name] = time.time()
                
        def post_hook(module: nn.Module, input, output, name):
            if isinstance(module, self.target_modules) and name in self.start_times:
                inference_time = time.time() - self.start_times[name]
                self.module_times[module.__class__.__name__].append(inference_time)
        
        for name, module in model.named_modules():
            if isinstance(module, self.target_modules):
                self.hooks.append(
                    module.register_forward_pre_hook(partial(pre_hook, name=name))
                )
                self.hooks.append(
                    module.register_forward_hook(partial(post_hook, name=name))
                )
    
    def measure_inference_time(self, model: nn.Module, inference_func: Callable, 
                             warmup_runs: int = 5, measure_runs: int = 20) -> Dict[str, float]:
        """
        Measure inference time with warmup.
        
        Args:
            model: PyTorch model
            inference_func: Function that performs inference
            warmup_runs: Number of warmup runs
            measure_runs: Number of measurement runs
            
        Returns:
            Dictionary with timing statistics
        """
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                inference_func()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure
        times = []
        for _ in range(measure_runs):
            start_time = time.time()
            
            with torch.no_grad():
                result = inference_func()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        self.inference_times.extend(times)
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'median_time_ms': np.median(times)
        }
    
    def get_module_latency_results(self) -> Dict[str, str]:
        """Get module-wise latency results."""
        results = {}
        for module_name, times in self.module_times.items():
            if times:
                avg_time = sum(times) / len(times) * 1000  # Convert to ms
                results[module_name] = f'{avg_time:.3f} ms'
        return results


class MemoryCalculator(MetricCalculator):
    """Calculate peak memory usage based on memory_utils.py"""
    
    def measure_peak_memory(self, inference_func: Callable) -> Tuple[any, float]:
        """
        Measure peak GPU memory usage during inference.
        
        Args:
            inference_func: Function that performs inference
            
        Returns:
            Tuple of (inference_result, peak_memory_mb)
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            result = inference_func()
            
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
            
            self.peak_memory = max(self.peak_memory, peak_memory_mb)
            
            return result, peak_memory_mb
        else:
            # CPU memory measurement (simplified)
            import psutil
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss
            
            result = inference_func()
            
            mem_after = process.memory_info().rss
            peak_memory_mb = (mem_after - mem_before) / (1024 * 1024)
            
            self.peak_memory = max(self.peak_memory, peak_memory_mb)
            
            return result, peak_memory_mb


class AccuracyCalculator(MetricCalculator):
    """Calculate accuracy metrics: mIoU, AP, AR - matching train.py implementation exactly"""
    
    def calculate_iou(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
        """
        Calculate Intersection over Union (IoU) - matches train.py compute_iou function exactly.
        
        Args:
            pred_masks: Predicted masks [B, 1, H, W] or [B, H, W]
            gt_masks: Ground truth masks [B, 1, H, W] or [B, H, W]
            
        Returns:
            Mean IoU across batch
        """
        # Ensure masks have the same format as train.py expects
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)  # Add channel dimension
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)
            
        assert gt_masks.shape[1] == 1, 'only support one mask per image now'
        
        # Handle size mismatch (same as train.py)
        if pred_masks.shape[2] != gt_masks.shape[2] or pred_masks.shape[3] != gt_masks.shape[3]:
            postprocess_preds = F.interpolate(pred_masks, size=gt_masks.size()[2:], mode='bilinear', align_corners=False)
        else:
            postprocess_preds = pred_masks
        
        # Calculate IoU for each sample (same as train.py)
        iou = 0
        for i in range(len(pred_masks)):
            # Use the same mask_iou function as train.py
            iou = iou + self._mask_iou(postprocess_preds[i], gt_masks[i])
        
        mean_iou = iou / len(pred_masks)
        self.ious.append(mean_iou)
        return mean_iou
    
    def calculate_boundary_iou(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
        """
        Calculate boundary IoU - matches train.py compute_boundary_iou function exactly.
        
        Args:
            pred_masks: Predicted masks [B, 1, H, W] or [B, H, W]
            gt_masks: Ground truth masks [B, 1, H, W] or [B, H, W]
            
        Returns:
            Mean boundary IoU across batch
        """
        # Ensure masks have the same format as train.py expects
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)
            
        assert gt_masks.shape[1] == 1, 'only support one mask per image now'
        
        # Handle size mismatch (same as train.py)
        if pred_masks.shape[2] != gt_masks.shape[2] or pred_masks.shape[3] != gt_masks.shape[3]:
            postprocess_preds = F.interpolate(pred_masks, size=gt_masks.size()[2:], mode='bilinear', align_corners=False)
        else:
            postprocess_preds = pred_masks
        
        # Calculate boundary IoU for each sample (same as train.py)
        iou = 0
        for i in range(len(pred_masks)):
            # Use the same boundary_iou function as train.py (note: gt first, pred second)
            iou = iou + self._boundary_iou(gt_masks[i], postprocess_preds[i])
        
        mean_boundary_iou = iou / len(pred_masks)
        self.boundary_ious.append(mean_boundary_iou)
        return mean_boundary_iou
    
    def _mask_iou(self, pred_label: torch.Tensor, label: torch.Tensor) :
        """
        Ccalculate mask iou for pred_label and gt_label
        """
        pred_label = (pred_label>0)[0].int()
        label = (label>128)[0].int()

        intersection = ((label * pred_label) > 0).sum()
        union = ((label + pred_label) > 0).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (intersection / union).item()
    
    def _boundary_iou(self, gt, dt, dilation_ratio=0.02) :
        """
        Compute boundary iou between two binary masks.
        :param gt (numpy array, uint8): binary mask
        :param dt (numpy array, uint8): binary mask
        :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
        :return: boundary iou (float) 
        """
        
        
        
        device = gt.device
        dt = (dt>0)[0].cpu().byte().numpy()
        gt = (gt>128)[0].cpu().byte().numpy()

        gt_boundary = mask_to_boundary(gt, dilation_ratio)
        dt_boundary = mask_to_boundary(dt, dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        boundary_iou = intersection / union
        return torch.tensor(boundary_iou).float().to(device)
    
    def calculate_ap_ar(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, pred_scores: torch.Tensor = None,
                    iou_thresholds: List[float] = None) :
        """
        Calculate Average Precision (AP) and Average Recall (AR) for one object per image.
        
        Args:
            pred_masks: Predicted masks [B, 1, H, W] or [B, H, W]
            gt_masks: Ground truth masks [B, 1, H, W] or [B, H, W]
            pred_scores: Confidence scores for predictions [B]
            iou_thresholds: IoU thresholds for evaluation (default: COCO thresholds)
                
        Returns:
            Tuple of (mean AP, mean AR) across batch and IoU thresholds
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        if pred_masks.numel() == 0 or gt_masks.numel() == 0:
            return 0.0, 0.0
        

        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)  
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)  
        
        assert gt_masks.shape[1] == 1, 'Only support one mask per image'
        
        if pred_scores is None:
            pred_scores = torch.ones(pred_masks.shape[0], device=pred_masks.device)

        if pred_masks.shape[2] != gt_masks.shape[2] or pred_masks.shape[3] != gt_masks.shape[3]:
            pred_masks = F.interpolate(pred_masks, size=gt_masks.size()[2:], mode='bilinear', align_corners=False)
        
        # Compute IoU for each sample
        ious = []
        for i in range(pred_masks.shape[0]):
            iou = self._mask_iou(pred_masks[i], gt_masks[i])  # Compute IoU for single pair
            ious.append(iou)
        ious = torch.tensor(ious, device=pred_masks.device)
        
        # Initialize lists for AP and AR
        aps = []
        ars = []
        
        for iou_th in iou_thresholds:
            # For each IoU threshold, compute precision and recall
            matches = (ious >= iou_th).float()  # [B], 1 if IoU >= threshold, else 0

            sorted_indices = torch.argsort(pred_scores, descending=True)
            sorted_matches = matches[sorted_indices]

            true_positives = sorted_matches.cumsum(dim=0)  # Cumulative TPs
            precisions = true_positives / torch.arange(1, len(true_positives) + 1, device=true_positives.device)  # TP / (TP + FP)
            recalls = true_positives / pred_masks.shape[0]  # TP / GT (1 GT per image)
            
            # Interpolate precision at recall levels (COCO-style, simplified)
            recall_levels = torch.linspace(0, 1, 101, device=recalls.device)
            interpolated_precisions = []
            for r in recall_levels:
                valid = recalls >= r
                if valid.any():
                    interpolated_precisions.append(precisions[valid].max())
                else:
                    interpolated_precisions.append(0.0)
            ap = torch.tensor(interpolated_precisions).mean().item()
            
            # AR: Average recall at this IoU threshold (fraction of GTs matched)
            ar = matches.mean().item()  # Since 1 GT per image, mean of matches
            
            aps.append(ap)
            ars.append(ar)
        
        mean_ap = np.mean(aps) if aps else 0.0
        mean_ar = np.mean(ars) if ars else 0.0
        
        # Store metrics
        self.aps.append(mean_ap)
        self.ars.append(mean_ar)
        
        return mean_ap, mean_ar


class ComprehensiveMetricCalculator:
    """
    Main class that combines all metric calculators for comprehensive evaluation.
    """
    
    def __init__(self, target_modules: Tuple = None):
        self.target_modules = target_modules or (nn.Linear, nn.Conv2d, nn.MultiheadAttention)
     
        self.latency_calc = LatencyCalculator(target_modules)
        self.memory_calc = MemoryCalculator()
        self.accuracy_calc = AccuracyCalculator()
        
    def calculate_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Calculate model size and parameter count."""
        param_count = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        total_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        return {
            'total_params': param_count,
            'total_size_mb': total_size_mb,
            'param_size_mb': param_size / (1024 * 1024),
            'buffer_size_mb': buffer_size / (1024 * 1024)
        }
    
    def profile_model(self, model: nn.Module, sample_input, inference_func: Callable,
                     warmup_runs: int = 5, measure_runs: int = 20) -> Dict[str, any]:
        """
        Comprehensive model profiling.
        
        Args:
            model: PyTorch model to profile
            sample_input: Sample input for FLOP calculation
            inference_func: Function that performs inference
            warmup_runs: Number of warmup runs for timing
            measure_runs: Number of measurement runs for timing
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Model size
        model_size = self.calculate_model_size(model)
        results.update(model_size)
        
        # Latency
        self.latency_calc.register_latency_hooks(model)
        timing_stats = self.latency_calc.measure_inference_time(
            model, inference_func, warmup_runs, measure_runs
        )
        results.update(timing_stats)
        results['module_latency'] = self.latency_calc.get_module_latency_results()
        self.latency_calc.clear_hooks()
        
        # Peak memory
        _, peak_memory = self.memory_calc.measure_peak_memory(inference_func)
        results['peak_memory_mb'] = peak_memory
        
        return results
    
    def calculate_accuracy_metrics(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Dict[str, float]:
        """Calculate all accuracy metrics - matches train.py implementation."""
        iou = self.accuracy_calc.calculate_iou(pred_masks, gt_masks)
        boundary_iou = self.accuracy_calc.calculate_boundary_iou(pred_masks, gt_masks)
        ap, ar = self.accuracy_calc.calculate_ap_ar(pred_masks, gt_masks)
        
        return {
            'iou': iou,
            'boundary_iou': boundary_iou,
            'ap': ap,
            'ar': ar
        }
    
    def get_summary(self) -> Dict[str, any]:
        """Get comprehensive summary of all metrics."""
        summary = {}
        
        # Accuracy metrics
        if self.accuracy_calc.ious:
            summary.update({
                'mean_iou': np.mean(self.accuracy_calc.ious),
                'std_iou': np.std(self.accuracy_calc.ious),
                'mean_boundary_iou': np.mean(self.accuracy_calc.boundary_ious),
                'std_boundary_iou': np.std(self.accuracy_calc.boundary_ious),
                'mean_ap': np.mean(self.accuracy_calc.aps),
                'std_ap': np.std(self.accuracy_calc.aps),
                'mean_ar': np.mean(self.accuracy_calc.ars),
                'std_ar': np.std(self.accuracy_calc.ars),
            })
        
        # Performance metrics
        if self.latency_calc.inference_times:
            summary.update({
                'mean_inference_time_ms': np.mean(self.latency_calc.inference_times),
                'std_inference_time_ms': np.std(self.latency_calc.inference_times),
            })
        
        summary.update({
            'gflops': self.gflops_calc.total_flops,
            'peak_memory_mb': self.memory_calc.peak_memory,
        })
        
        return summary
    
    def save_metrics(self, filepath: str, additional_data: Dict = None):
        """Save all metrics to JSON file."""
        summary = self.get_summary()
        
        if additional_data:
            summary.update(additional_data)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Comprehensive metrics saved to: {filepath}")
    
    def print_summary(self):
        """Print formatted summary of all metrics."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("COMPREHENSIVE METRICS SUMMARY")
        print("="*60)
        
        print(f"Model Complexity:")
        print(f"  GFLOPs: {summary.get('gflops', 0):.3f}")
        print(f"  Peak Memory: {summary.get('peak_memory_mb', 0):.2f} MB")
        
        if 'mean_inference_time_ms' in summary:
            print(f"\nPerformance:")
            print(f"  Mean Inference Time: {summary['mean_inference_time_ms']:.2f} ± {summary['std_inference_time_ms']:.2f} ms")
        
        if 'mean_iou' in summary:
            print(f"\nAccuracy Metrics:")
            print(f"  mIoU: {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
            print(f"  Boundary IoU: {summary['mean_boundary_iou']:.4f} ± {summary['std_boundary_iou']:.4f}")
            print(f"  AP: {summary['mean_ap']:.4f} ± {summary['std_ap']:.4f}")
            print(f"  AR: {summary['mean_ar']:.4f} ± {summary['std_ar']:.4f}")
        
        print("="*60)