import os
import logging
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from functools import partial
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
from segment_anything.modeling.transformer import TwoWayTransformer
from train.train import compute_iou, compute_boundary_iou, MaskDecoderHQ
from segment_anything import SamPredictor, sam_model_registry
import train.utils.misc as misc
from utils import show_mask_image
from decoder_quant import mask_decoder_monkey_patch, TwoWayTransformerObserver
from encoder_quant import image_encoder_monkey_patch
from quant_utils import (
    quantize_activation_per_token_absmax,
)
from processors import (
    get_encoder_processor,
    EncoderRecenterAttentionProcessor,
    EncoderAttentionProcessor,
    DecoderDoNothingProcessor,
)
from profiler import InferenceProfiler, compare_inference_speed
from segment_anything.modeling.image_encoder import Attention as EncoderSamAttention
from segment_anything.modeling.transformer import  Attention as  DecoderAttention
from train.segment_anything_training.modeling.image_encoder import Attention as EncoderAttentionTraining
from seginw.segment_anything.modeling.image_encoder import Attention as EncoderAttention 





def create_calib_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1 ):
    gos_dataloaders = []
    gos_datasets = []
    for i in range(len(name_im_gt_list)):   
        gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
        dataloader = DataLoader(gos_dataset, batch_size, drop_last=False)
        gos_dataloaders.append(dataloader)
        gos_datasets.append(gos_dataset) 
    return gos_dataloaders, gos_datasets

def setup_logger(path_log, state):
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(path_log, f'{state}.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_default_datasets():
    """Get default dataset configurations"""
    return [
        {
            "name": "DIS5K-VD",
            "im_dir": "./data/DIS5K/DIS-VD/im",
            "gt_dir": "./data/DIS5K/DIS-VD/gt",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        {
            "name": "ThinObject5k-TR",
            "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
            "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        }
    ]


def plot_output(imgs, masks, labels_boxes, scores: np.ndarray, example_idx, output_path='./output'):
    """Plot prediction output with mask and bounding box"""
    plt.figure(figsize=(10, 10))
    plt.imshow(imgs.squeeze())
    os.makedirs(output_path, exist_ok=True)

    if len(masks) > 0:
        show_mask_image(masks[0], plt.gca(), random_color=False)

    box = labels_boxes[0]
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    plt.title(f'Example {example_idx} - Score: {scores.item():.3f}')
    plt.savefig(f'{output_path}/sample_{example_idx}.png')
    plt.axis('off')
    plt.show()


class Evaluator:
    """Handles model evaluation on HQ44k dataset"""

    def __init__(self, accelerator, dataloaders, datasets):
        self.accelerator = accelerator
        self.dataloaders = dataloaders
        self.datasets = datasets

    def eval_hq44k(self, predictor: SamPredictor, num_samples=None, plot_figures=False):
        """Evaluate model on HQ44k dataset"""
        test_stats = {}

        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print('valid_dataloader len:', len(dataloader))
            progress_bar = tqdm(total=len(dataloader) if not num_samples else num_samples, desc=f"Eval HQ44k")
            metric_logger = misc.MetricLogger(delimiter="  ")
            index = 0

            for data_val in metric_logger.log_every(dataloader, 2):
                if index == num_samples:
                    break

                _, inputs_val, labels_val, _, labels_ori, ori_image = (
                    data_val['imidx'], data_val['image'], data_val['label'],
                    data_val['shape'], data_val['ori_label'], data_val['ori_im']
                )

                imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
                predictor.set_image(imgs.squeeze())
                labels_boxes = misc.masks_to_boxes(labels_val[:, 0, :, :])

                masks, scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=labels_boxes,
                    hq_token_only=True
                )

                iou = compute_iou(masks, labels_ori)
                boundary_iou = compute_boundary_iou(masks, labels_ori)
                loss_dict = {"val_iou_" + str(k): iou, "val_boundary_iou_" + str(k): boundary_iou}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                progress_bar.update(1)

                if plot_figures:
                    masks = masks.squeeze(1).cpu().detach().numpy()
                    labels_boxes = labels_boxes.cpu().detach().numpy()
                    scores = scores.squeeze().cpu().detach().numpy()
                    plot_output(imgs, masks, labels_boxes, scores, index)

                index += 1

            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
            test_stats.update(resstat)

        return test_stats


class QKAnalyzer:
    """Analyzes Q/K quantization errors in attention layers"""

    def __init__(self, accelerator, dataloaders):
        self.accelerator = accelerator
        self.dataloaders = dataloaders
        self.qk_error_results = []

    def analyze_qk_quantization_errors(self, predictor, num_samples=5, n_bits_range=[4, 6, 8], save_results=True):
        """
        Analyze quantization errors in Q and K tensors for cross attention operators

        Args:
            predictor: SamPredictor instance
            num_samples: Number of samples to analyze
            n_bits_range: List of bit widths to test
            save_results: Whether to save results to CSV
        """
        print("Starting Q/K quantization error analysis...")

        # Clear previous observations
        TwoWayTransformerObserver.clear_dict()

        # Run inference to collect attention data
        for sample_idx in range(num_samples):
            print(f"Processing sample {sample_idx + 1}/{num_samples}")

            # Use existing evaluation data
            dataloader = self.accelerator.prepare(self.dataloaders[0])
            for i, data_val in enumerate(dataloader):
                if i == sample_idx:
                    _, inputs_val, labels_val, _, labels_ori, ori_image = (
                        data_val['imidx'], data_val['image'], data_val['label'],
                        data_val['shape'], data_val['ori_label'], data_val['ori_im']
                    )
                    imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
                    predictor.set_image(imgs.squeeze())
                    labels_boxes = misc.masks_to_boxes(labels_val[:, 0, :, :]).cpu().numpy()

                    # Run prediction to collect attention data
                    masks, scores, _ = predictor.predict(
                        box=labels_boxes,
                        hq_token_only=False
                    )
                    break

        # Analyze collected attention data
        results = self._analyze_attention_data(n_bits_range)

        if save_results:
            self._save_qk_error_results(results)

        return results

    def _analyze_attention_data(self, n_bits_range):
        """Analyze the collected attention data for Q/K quantization errors"""
        results = {}

        # Define attention layers to analyze
        attention_layers = ['p2i_attn', 'i2p_attn', 'final_attn']
        tensor_keys = {
            'p2i_attn': ('p2i_q', 'p2i_k', 'p2i_v'),
            'i2p_attn': ('i2p_q', 'i2p_k', 'i2p_v'),
            'final_attn': ('final_q', 'final_k', 'final_v')
        }

        for layer_name in attention_layers:
            if layer_name in TwoWayTransformerObserver.attention_score:
                print(f"Analyzing {layer_name}...")

                attn_data = TwoWayTransformerObserver.attention_score[layer_name]
                q_key, k_key, v_key = tensor_keys[layer_name]

                if layer_name == 'final_attn':
                    q = TwoWayTransformerObserver.attention_score[q_key]
                    k = TwoWayTransformerObserver.attention_score[k_key]
                    v = TwoWayTransformerObserver.attention_score[v_key]
                    attn_map = attn_data
                else:
                    # Use first layer for analysis
                    if len(attn_data) > 0:
                        q = TwoWayTransformerObserver.attention_score[q_key][0]
                        k = TwoWayTransformerObserver.attention_score[k_key][0]
                        v = TwoWayTransformerObserver.attention_score[v_key][0]
                        attn_map = attn_data[0]
                    else:
                        continue

                # Analyze this layer
                layer_results = {}

                # 1. Bit-width sensitivity analysis
                layer_results['bit_sensitivity'] = self._analyze_attention_sensitivity(q, k, v, n_bits_range)

                # 2. Q vs K error contribution
                layer_results['qk_contribution'] = self._analyze_qk_error_contribution(q, k, v)

                results[layer_name] = layer_results

        return results

    def _analyze_attention_sensitivity(self, q, k, v, n_bits_range):
        """Analyze how different bit widths affect attention patterns"""
        results = {}

        # Original attention computation
        original_attn = self._compute_attention(q, k, v)

        for n_bits in n_bits_range:
            # Test Q quantization only
            q_quant = self._quantize_tensor(q.clone(), n_bits)
            attn_q_quant = self._compute_attention(q_quant, k, v)

            # Test K quantization only
            k_quant = self._quantize_tensor(k.clone(), n_bits)
            attn_k_quant = self._compute_attention(q, k_quant, v)

            # Test both Q and K quantization
            attn_qk_quant = self._compute_attention(q_quant, k_quant, v)

            # Compute errors
            results[n_bits] = {
                'q_only_error': self._compute_attention_error(original_attn, attn_q_quant),
                'k_only_error': self._compute_attention_error(original_attn, attn_k_quant),
                'qk_joint_error': self._compute_attention_error(original_attn, attn_qk_quant),
                'q_statistics': self._compute_tensor_stats(q),
                'k_statistics': self._compute_tensor_stats(k)
            }

        return results

    def _analyze_qk_error_contribution(self, q, k, v, n_bits=8):
        """Analyze relative contribution of Q vs K to total quantization error"""
        # Original attention
        original_attn = self._compute_attention(q, k, v)

        # Individual quantization
        q_quant = self._quantize_tensor(q.clone(), n_bits)
        k_quant = self._quantize_tensor(k.clone(), n_bits)

        attn_q_only = self._compute_attention(q_quant, k, v)
        attn_k_only = self._compute_attention(q, k_quant, v)
        attn_both = self._compute_attention(q_quant, k_quant, v)

        # Compute error contributions
        q_error = self._compute_attention_error(original_attn, attn_q_only)
        k_error = self._compute_attention_error(original_attn, attn_k_only)
        joint_error = self._compute_attention_error(original_attn, attn_both)

        # Analyze interaction effects
        expected_independent = q_error['mse'] + k_error['mse']
        actual_joint = joint_error['mse']
        interaction_effect = actual_joint - expected_independent

        return {
            'q_error': q_error,
            'k_error': k_error,
            'joint_error': joint_error,
            'q_contribution_ratio': q_error['mse'] / (q_error['mse'] + k_error['mse'] + 1e-8),
            'interaction_effect': interaction_effect,
            'error_additivity': abs(interaction_effect) / (expected_independent + 1e-8)
        }

    def _compute_attention(self, q, k, v):
        """Compute attention weights from Q, K, V"""
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / (q.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = attn_weights @ v
        return attn_out

    def _quantize_tensor(self, tensor, n_bits):
        """Quantize tensor using per-token absmax quantization"""
        original_shape = tensor.shape
        tensor_flat = tensor.reshape(-1, original_shape[-1])
        quantized = quantize_activation_per_token_absmax(tensor_flat, n_bits=n_bits)
        return quantized.reshape(original_shape)

    def _compute_attention_error(self, original_attn, quantized_attn):
        """Compute error metrics between attention maps"""
        mse_error = F.mse_loss(original_attn, quantized_attn).item()
        mae_error = F.l1_loss(original_attn, quantized_attn).item()
        cosine_sim = F.cosine_similarity(
            original_attn.flatten(), quantized_attn.flatten(), dim=0
        ).item()

        return {
            'mse': mse_error,
            'mae': mae_error,
            'cosine_similarity': cosine_sim,
            'max_abs_error': (original_attn - quantized_attn).abs().max().item()
        }

    def _compute_tensor_stats(self, tensor, prefix=''):
        """Compute statistical properties of a tensor"""
        return {
            f'{prefix}range': tensor.abs().max().item(),
            f'{prefix}std': tensor.std().item(),
            f'{prefix}mean': tensor.mean().item(),
            f'{prefix}abs_mean': tensor.abs().mean().item(),
            f'{prefix}sparsity': (tensor.abs() < 1e-6).float().mean().item()
        }

    def _save_qk_error_results(self, results):
        """Save Q/K error analysis results"""
        timestamp = datetime.datetime.now().isoformat()

        # Flatten results for CSV storage
        flat_results = []
        for layer_name, layer_data in results.items():
            for analysis_type, analysis_data in layer_data.items():
                if analysis_type == 'bit_sensitivity':
                    for n_bits, bit_data in analysis_data.items():
                        flat_results.append({
                            'timestamp': timestamp,
                            'layer': layer_name,
                            'analysis_type': analysis_type,
                            'n_bits': n_bits,
                            **self._flatten_nested_dict(bit_data)
                        })
                elif analysis_type == 'qk_contribution':
                    flat_results.append({
                        'timestamp': timestamp,
                        'layer': layer_name,
                        'analysis_type': analysis_type,
                        **self._flatten_nested_dict(analysis_data)
                    })

        self.qk_error_results.extend(flat_results)

        # Save to CSV
        filename = f'qk_error_analysis_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df = pd.DataFrame(flat_results)
        df.to_csv(filename, index=False)
        print(f"Q/K error analysis results saved to {filename}")

    def _flatten_nested_dict(self, nested_dict, parent_key='', sep='_'):
        """Flatten nested dictionary for CSV storage"""
        items = []
        for k, v in nested_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_nested_dict(v, new_key, sep=sep).items())
            elif isinstance(v, torch.Tensor):
                # Convert tensor to scalar or list
                if v.numel() == 1:
                    items.append((new_key, v.item()))
                else:
                    items.append((new_key, v.cpu().numpy().tolist()))
            else:
                items.append((new_key, v))
        return dict(items)

    def clear_qk_error_results(self):
        """Clear stored Q/K error analysis results"""
        self.qk_error_results = []
        print("Cleared Q/K error analysis results")


class KPreserveExperimenter:
    """Handles k_preserve experiments"""

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.k_preserve_results = []

    def run_k_preserve_experiment(self, predictor, k_preserve_values, num_samples=None, experiment_config=None, target='decoder'):
        """
        Run a complete k_preserve experiment and save results.

        Args:
            predictor: SamPredictor instance
            k_preserve_values: List of k_preserve values to test
            num_samples: Number of samples for evaluation
            experiment_config: Configuration dict with n_bits, weight_quant, etc.
            target: 'decoder', 'encoder', or 'both'
        """
        from quant_utils import AttnBasedProcessor, ImageEncoderProcessor
        from segment_anything.modeling.image_encoder import Block

        print(f"Running k_preserve experiment on {target} with values: {k_preserve_values}")

        for k_preserve in k_preserve_values:
            print(f"\n=== Testing k_preserve = {k_preserve} ===")

            # Setup processors based on target
            encoder_processor = None
            decoder_processor = None

            if target in ['encoder', 'both']:
                encoder_processor = ImageEncoderProcessor('encoder_attn')
                encoder_processor.calibrate(
                    predictor=predictor,
                    modules=(Block,),
                    num_samples=32
                )

            if target in ['decoder', 'both']:
                decoder_processor = AttnBasedProcessor('decoder_attn')
                decoder_processor.calibrate(
                    predictor=predictor,
                    modules=(TwoWayTransformer),
                    num_samples=32
                )

            # Apply quantization with current k_preserve
            n_bits = experiment_config.get('n_bits', 4) if experiment_config else 4
            weight_quant = experiment_config.get('weight_quant', 'selective_channel') if experiment_config else 'selective_channel'

            if target in ['encoder', 'both'] and encoder_processor:
                image_encoder_monkey_patch(
                    predictor.model,
                    encoder_processor,
                    n_bits=n_bits,
                    weight_quant=weight_quant,
                    k_preserve=k_preserve
                )

            if target in ['decoder', 'both'] and decoder_processor:
                mask_decoder_monkey_patch(
                    predictor.model,
                    decoder_processor,
                    n_bits=n_bits,
                    weight_quant=weight_quant,
                    k_preserve=k_preserve
                )

            # Run evaluation
            test_stats = self.evaluator.eval_hq44k(predictor=predictor, num_samples=num_samples, plot_figures=False)

            # Store results
            config = experiment_config.copy() if experiment_config else {}
            config.update({
                'num_samples': num_samples,
                'experiment_type': 'k_preserve_sweep',
                'target': target
            })

            self.save_k_preserve_result(k_preserve, test_stats, config)

            # Clean up
            del encoder_processor, decoder_processor
            torch.cuda.empty_cache()

        # Save all results to CSV
        return self.save_k_preserve_results_to_csv()

    def save_k_preserve_result(self, k_preserve, test_stats, experiment_config=None):
        """Store results from a k_preserve experiment"""
        result_entry = {
            'k_preserve': k_preserve,
            'timestamp': datetime.datetime.now().isoformat(),
            **test_stats
        }

        # Add experiment configuration if provided
        if experiment_config:
            result_entry.update(experiment_config)

        self.k_preserve_results.append(result_entry)
        print(f"Stored result for k_preserve={k_preserve}")

    def save_k_preserve_results_to_csv(self, filename=None):
        """Save all k_preserve results to CSV file"""
        if not self.k_preserve_results:
            print("No k_preserve results to save")
            return None

        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'k_preserve_results_{timestamp}.csv'

        df = pd.DataFrame(self.k_preserve_results)
        df.to_csv(filename, index=False)
        print(f"K_preserve results saved to {filename}")
        return filename

    def clear_k_preserve_results(self):
        """Clear stored k_preserve results"""
        self.k_preserve_results = []
        print("Cleared k_preserve results")


class Visualizer:
    """Handles all visualization and plotting"""

    @staticmethod
    def visualize_qk_errors(results, save_path='./'):
        """Create visualizations of Q/K quantization errors"""
        # 1. Bit-width sensitivity comparison
        Visualizer._plot_bit_sensitivity(results, save_path)

        # 2. Q vs K error contribution
        Visualizer._plot_qk_contribution(results, save_path)

    @staticmethod
    def _plot_bit_sensitivity(results, save_path):
        """Plot bit-width sensitivity analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, (layer_name, layer_data) in enumerate(results.items()):
            if i >= len(axes) or 'bit_sensitivity' not in layer_data:
                continue

            ax = axes[i]
            bit_data = layer_data['bit_sensitivity']

            bits = list(bit_data.keys())
            q_errors = [bit_data[b]['q_only_error']['mae'] for b in bits]
            k_errors = [bit_data[b]['k_only_error']['mae'] for b in bits]
            qk_errors = [bit_data[b]['qk_joint_error']['mae'] for b in bits]

            ax.plot(bits, q_errors, 'o-', label='Q only', linewidth=2, markersize=6)
            ax.plot(bits, k_errors, 's-', label='K only', linewidth=2, markersize=6)
            ax.plot(bits, qk_errors, '^-', label='Q+K joint', linewidth=2, markersize=6)

            ax.set_xlabel('Quantization Bits')
            ax.set_ylabel('MAE Error')
            ax.set_title(f'{layer_name} Bit Sensitivity')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/qk_bit_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def _plot_head_errors(results, save_path):
        """Plot head-wise error analysis"""
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
        if len(results) == 1:
            axes = [axes]

        for i, (layer_name, layer_data) in enumerate(results.items()):
            if 'head_errors' not in layer_data:
                continue

            head_data = layer_data['head_errors']
            head_indices = [h['head_idx'] for h in head_data]
            attention_errors = [h['attention_error']['mse'] for h in head_data]

            axes[i].bar(head_indices, attention_errors, alpha=0.7, color='skyblue')
            axes[i].set_xlabel('Attention Head')
            axes[i].set_ylabel('MSE Error')
            axes[i].set_title(f'{layer_name} Head Errors')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/qk_head_errors.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def _plot_qk_contribution(results, save_path):
        """Plot Q vs K error contribution analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        layers = []
        q_contributions = []
        interaction_effects = []

        for layer_name, layer_data in results.items():
            if 'qk_contribution' not in layer_data:
                continue

            contrib_data = layer_data['qk_contribution']
            layers.append(layer_name)
            q_contributions.append(contrib_data['q_contribution_ratio'])
            interaction_effects.append(contrib_data['interaction_effect'])

        # Q vs K contribution ratio
        bars1 = ax1.bar(layers, q_contributions, alpha=0.7, color='lightcoral')
        ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Equal contribution')
        ax1.set_ylabel('Q Contribution Ratio')
        ax1.set_title('Q vs K Error Contribution')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Interaction effects
        colors = ['green' if x >= 0 else 'red' for x in interaction_effects]
        bars2 = ax2.bar(layers, interaction_effects, alpha=0.7, color=colors)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_ylabel('Interaction Effect')
        ax2.set_title('Q-K Quantization Interaction')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/qk_contribution.png', dpi=300, bbox_inches='tight')
        plt.show()


class Engine:
    """Main engine class for orchestrating quantization experiments"""

    def __init__(self, strategy_name: str, datasets=None, quantize_encoder=False, quantize_decoder=True) -> None:
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.stat = {}
        self.strategy_name = strategy_name
        self.quantize_encoder = quantize_encoder
        self.quantize_decoder = quantize_decoder

        # Setup datasets
        if datasets is None:
            datasets = get_default_datasets()

        valid_im_gt_list = get_im_gt_name_dict([datasets[0]], flag="valid")
        self.dataloaders, self.datasets = create_calib_dataloaders(
            valid_im_gt_list,
            my_transforms=[Resize([1024, 1024])],
            batch_size=1,
        )

        # Initialize components
        self.evaluator = Evaluator(self.accelerator, self.dataloaders, self.datasets)
        self.qk_analyzer = QKAnalyzer(self.accelerator, self.dataloaders)
        self.k_preserve_experimenter = KPreserveExperimenter(self.evaluator)
        self.visualizer = Visualizer()

        # Encoder-specific components (initialized on demand)
        self.encoder_qk_analyzer = None

    def apply_quantization(self, predictor, encoder_config=None, decoder_config=None,args_yaml= None):
        """
        Apply quantization to encoder and/or decoder based on configuration.

        Args:
            predictor: SamPredictor instance
            encoder_config: Dict with encoder quantization config {processor, n_bits, weight_quant, k_preserve}
            decoder_config: Dict with decoder quantization config {processor, n_bits, weight_quant, k_preserve}
        """

        if self.quantize_encoder and encoder_config:
            print("Applying encoder quantization...")
            image_encoder_monkey_patch(
                predictor.model,
                processor=encoder_config.get('processor'),
                n_bits=encoder_config.get('n_bits', 8),
                weight_quant=encoder_config.get('weight_quant', 'per_channel'),
                act_quant=encoder_config.get('act_quant', 'per_token'),
                args_yaml= args_yaml,
            )
            print(f"Encoder quantized: {encoder_config.get('n_bits', 8)}-bit, "
                  f"weight: {encoder_config.get('weight_quant', 'per_channel')}", f"activation: {encoder_config.get('act_quant', 'per_token')} ")

        if self.quantize_decoder and decoder_config:
            print("Applying decoder quantization...")
            mask_decoder_monkey_patch(
                predictor.model,
                processor=decoder_config.get('processor'),
                n_bits=decoder_config.get('n_bits', 8),
                weight_quant=decoder_config.get('weight_quant', 'per_channel'),
                k_preserve=decoder_config.get('k_preserve', 0),  
            )
            print(f"Decoder quantized: {decoder_config.get('n_bits', 8)}-bit, "
                  f"{decoder_config.get('weight_quant', 'per_channel')}, k_preserve={decoder_config.get('k_preserve', 0)}")

    def setup_and_calibrate_processors(self, predictor, num_calib_samples=32, encoder_processor:EncoderAttentionProcessor=None, decoder_processor:DecoderDoNothingProcessor=None,args_yaml=None):
        """
        Setup and calibrate processors for encoder and/or decoder.

        Args:
            predictor: SamPredictor instance
            num_calib_samples: Number of samples for calibration

        Returns:
            Tuple of (encoder_processor, decoder_processor)
        """

        if self.quantize_encoder:
            print("Setting up encoder processor...")
            # encoder_processor = EncoderAttentionProcessor()
            encoder_processor.set_params(args_yaml)
            encoder_processor.calibrate(
                predictor=predictor,
                modules=(DecoderAttention, EncoderAttentionTraining, EncoderAttention, EncoderSamAttention),
                num_samples=num_calib_samples
            )
            
            if args_yaml.quantization.quansmooth :
                encoder_processor.smooth_model(predictor,args_yaml.quantization.act_scales_file, args_yaml.quantization.centerQ)
            elif args_yaml.quantization.quanro:
                encoder_processor.quarot_model(predictor,args_yaml.quarot_inf, args_yaml.rtn_ro_config, centerQ=True)
                
            
            print(f"Encoder processor calibrated on {num_calib_samples} samples")
        

        if self.quantize_decoder:
            print("Setting up decoder processor...")
            # decoder_processor = DecoderDoNothingProcessor('decoder_attn')
            decoder_processor.calibrate(
                predictor=predictor,
                modules=(TwoWayTransformer,),
                num_samples=num_calib_samples
            )
            print(f"Decoder processor calibrated on {num_calib_samples} samples")

        return encoder_processor, decoder_processor

    # Delegate to Evaluator
    def eval_hq44k(self, predictor: SamPredictor, num_samples=None, plot_figures=False):
        """Delegate to evaluator component"""
        return self.evaluator.eval_hq44k(predictor, num_samples, plot_figures)

encoder_processor_registry = {
    'base': EncoderAttentionProcessor,
    'quarot': EncoderAttentionProcessor,
    'smooth': EncoderAttentionProcessor,
    'recenter': EncoderRecenterAttentionProcessor,
    # 'highlow':  EncoderHighLowAttentionProcessor,
}

decoder_processor_registry = {
    'base': EncoderAttentionProcessor,
    'quarot': EncoderAttentionProcessor,
    'smooth': EncoderAttentionProcessor,
    'recenter': EncoderRecenterAttentionProcessor,
    # 'highlow':  EncoderHighLowAttentionProcessor,
}



def override_args(args, args_yaml):
    """
    Override the parameters in args_yaml with the corresponding parameters from args.

    Args:
        args: Parsed command-line arguments.
        args_yaml: YAML configuration loaded as a dictionary or OmegaConf object.

    Returns:
        Updated args_yaml with overridden parameters.
    """
    # Map command-line arguments to YAML keys
    override_mapping = {
        'n_bits': 'quantization.n_bits',
        'n_bits_mlp': 'quantization.n_bits_mlp',
        'smooth': 'quantization.quansmooth',
        'quarot': 'quantization.quanro',
        'quantize_encoder': 'quantization.quanrtn',
        'en_weight_quant': 'quantization.weight_quant',
        'en_act_quant': 'quantization.act_quant',
        'centerQ': 'quantization.centerQ',
    }
    # Check if rtn_ro_config exists and extend override_mapping
    if hasattr(args_yaml, 'rtn_ro_config') and args_yaml.rtn_ro_config:
        override_mapping.update({
            'n_bits': 'rtn_ro_config.n_bits',
            'n_bits_mlp': 'rtn_ro_config.n_bits_mlp',
        })

    # Override parameters
    for arg_key, yaml_key in override_mapping.items():
        if hasattr(args, arg_key) and getattr(args, arg_key) is not None:
            value = getattr(args, arg_key)
            # Update nested keys in args_yaml
            keys = yaml_key.split('.')
            target = args_yaml
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value

    return args_yaml

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SAM Quantization Engine')
    parser.add_argument('--mode', type=str, default='eval',
                        choices=['eval', 'k_preserve', 'qk_analysis', 'benchmark'],
                        help='Execution mode')
    parser.add_argument('--quantize-encoder', action='store_true',
                        help='Enable encoder quantization')
    parser.add_argument('--encoder_processor', default='base',
                        help='Enable encoder quantization')
    parser.add_argument('--quantize-decoder', action='store_true', 
                        help='Enable decoder quantization')
    parser.add_argument('--n-bits', type=int, default=4,
                        help='Number of quantization bits')
    parser.add_argument('--n-bits-mlp', type=int, default=4,
                        help='Number of quantization bits')
    parser.add_argument('--en-weight-quant', type=str, default='per_channel',
                        choices=['per_channel', 'selective_channel'],
                        help='Weight quantization method')
    parser.add_argument('--de-weight-quant', type=str, default='per_channel',
                        choices=['per_channel', 'selective_channel'],
                        help='Weight quantization method')
    parser.add_argument('--en-act-quant', type=str, default='per_token',
                        choices=['per_token',  'low_high_density_activation'],
                        help='Weight quantization method')
    parser.add_argument('--de-act-quant', type=str, default='per_token',
                        choices=['per_channel', 'low_high_density_activation'],
                        help='Weight quantization method')
    parser.add_argument('--k-preserve', type=int, default=0,
                        help='Number of channels to preserve')
    parser.add_argument('--num-samples', type=int, default=400,
                        help='Number of evaluation samples')
    parser.add_argument('--num-calib-samples', type=int, default=16,
                        help='Number of calibration samples')
    parser.add_argument('--target', type=str, default='decoder',
                        choices=['decoder', 'encoder', 'both'],
                        help='Target for k_preserve experiments')
    parser.add_argument("--smooth", action='store_true', default=False, help="Smooth model")
    parser.add_argument('--quarot', action='store_true', default=False, help='Use quantization rotation')
    parser.add_argument("--config-file", type=str, default=None,)
    
    args = parser.parse_args()
    args_yaml = OmegaConf.load(args.config_file)
    # overide args_yaml with args
    args_yaml = override_args(args, args_yaml)

    model_type = 'vit_l'
    checkpoint_path = './pretrained_checkpoint/sam_hq_vit_l.pth'
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    # Initialize engine
    engine = Engine('hq44k', quantize_encoder=args.quantize_encoder, quantize_decoder=args.quantize_decoder)

    # encoder_processor = encoder_processor_registry[args.encoder_processor](args.encoder_processor, args.n_bits)
    # decoder_processor = decoder_processor_registry[args.encoder_processor](args.encoder_processor, args.n_bits)
    # print('='*50, f'using {encoder_processor.strategy_name}', 50*'=')

    # Setup and calibrate processors
    # enc_processor= get_encoder_processor("HEAD_PRUNE")
    # enc_processor= get_encoder_processor("POSITIONAL_PRUNE")
    enc_processor= get_encoder_processor("POSITIONAL_QUANT")
    encoder_processor, decoder_processor = engine.setup_and_calibrate_processors(
        predictor,
        num_calib_samples=args.num_calib_samples,
        encoder_processor=enc_processor,
        decoder_processor= DecoderDoNothingProcessor("DO_NOTHING"),
        args_yaml= args_yaml,
    )
    

    # Apply quantization
    encoder_config = {
        'processor': encoder_processor,
        'n_bits': args.n_bits,
        'weight_quant': args.en_weight_quant,
        'act_quant': args.en_act_quant,
    } if encoder_processor else None

    decoder_config = {
        'processor': decoder_processor,
        'n_bits': args.n_bits,
        'weight_quant': args.en_weight_quant,
        'act_quant': args.de_act_quant,
        'k_preserve': args.k_preserve
    } if decoder_processor else None

    engine.apply_quantization(predictor, encoder_config, decoder_config,args_yaml)
    # print_model_structure(predictor.model,"Final structure ")
    # Execute based on mode
    print("\n=== Running Evaluation ===")
    results = engine.eval_hq44k(predictor=predictor, num_samples=args.num_samples, plot_figures=False)
    print(f"\nResults: {results}")
    print("\n=== Execution Complete ===")
