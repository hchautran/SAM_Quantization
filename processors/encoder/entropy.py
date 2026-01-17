"""Attention head pruning processor for SAM quantization."""

import torch
from functools import partial
from collections import defaultdict
from tqdm.auto import tqdm

from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from ..base import AttentionProcessor, setup_logger
from torch.distributions import Exponential
from utils.quant_utils import quantize_activation_per_channel_absmax, quantize_activation_per_token_absmax

from omegaconf import OmegaConf

def exists(cfg, key):
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, key, default=None) is not None
    return hasattr(cfg, key)

class BaseEntropyProcessor(AttentionProcessor):
    """
    Base class for entropy-based attention head processing.

    Provides common functionality for:
    - Entropy calculation and statistics collection
    - Hook registration for attention pattern capture
    - Calibration loop execution
    - Head selection based on entropy thresholds or percentages
    """

    def __init__(self, strategy_name: str):
        super().__init__(strategy_name)
        self.stat = {}
        self.entropy_stats = defaultdict(list)
        self.threshold = 5.0
        self.percent = 0.5
        self.prunehighentropy = True
        self.prune_global = True
        self._mask_cache = {}

    def set_params(self, args):
        """Set parameters from args. Override in subclasses for custom params."""
        self.threshold = 5.0
        self.percent = (
            args.percent
            if exists(args, "percent")
            else args.quantization.percent_entropy
        )

        self.global_percent = (
            args.percent_global
            if exists(args, "percent_global")
            else args.quantization.percent_entropy_global
        )

        self.prunehighentropy = (
            args.high_entropy
            if exists(args, "high_entropy")
            else args.quantization.high_entropy
        )

        self.prune_global = (
            args.prune_global
            if exists(args, "prune_global")
            else args.quantization.prune_global
        )

        self.model_type = (
            args.model_type
            if exists(args, "model_type")
            else args.model.model_type
        )
    def calculate_entropy(self, attn_head):
        """
        Calculate entropy for attention head.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement calculate_entropy")

    def _create_attention_hook(self, name):
        """
        Create attention hook function.
        Must be implemented by subclasses to define hook behavior.
        """
        raise NotImplementedError("Subclasses must implement _create_attention_hook")

    def _register_hooks(self, predictor, modules):
        """Register hooks to capture attention patterns during forward pass."""
        hooks = []
        for name, component in predictor.model.image_encoder.named_modules():
            if isinstance(component, modules):
                print(f"Registering attention hook for {name}")
                hook_fn = self._create_attention_hook(name)
                hooks.append(component.register_forward_hook(hook_fn))
        return hooks, []

    def _run_forward_pass(self, predictor, data_val):
        """Run forward pass through image encoder only."""
        imgs = data_val['image'].permute(0, 2, 3, 1).cpu().numpy()
        predictor.set_image(imgs.squeeze())

    def _run_calibration_loop(self, predictor, num_samples):
        """Run the calibration data collection loop."""
        total_processed = 0
        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print(f'Dataloader {k} length:', len(dataloader))
            progress_bar = tqdm(
                total=min(num_samples, len(dataloader)),
                desc="Collecting entropy data"
            )

            for i, data_val in enumerate(dataloader):
                if total_processed >= num_samples:
                    break

                self._run_forward_pass(predictor, data_val)
                total_processed += 1
                progress_bar.update(1)

                # Log progress for first few samples
                if total_processed <= 3:
                    sample_head_key = list(self.entropy_stats.keys())[0] if self.entropy_stats else None
                    if sample_head_key:
                        current_length = len(self.entropy_stats[sample_head_key])
                        print(f'After sample {total_processed}: {sample_head_key} has {current_length} entropy values')

            if total_processed >= num_samples:
                break

    def _parse_head_key(self, head_key):
        """Parse layer name and head index from key format: 'blocks.22.attn.head_233'."""
        parts = head_key.split('.')
        layer_name = f"image_encoder.{parts[0]}.{parts[1]}.{parts[2]}"
        head_idx = int(parts[3].split('_')[1])
        return layer_name, head_idx

    def _compute_entropy_stats(self, entropy_values):
        """Convert entropy values to tensor and calculate mean."""
        entropy_tensor = torch.tensor(entropy_values)
        return torch.mean(entropy_tensor)

    def _calculate_num_heads_to_select(self, total_heads):
        """Calculate number of heads to select based on percentage."""
        return max(1, int(total_heads * self.percent))

    def _select_heads_by_threshold(self, predictor):
        """Select heads based on entropy threshold."""
        layer_heads = {}
        for head_key, entropy_values in self.entropy_stats.items():
            if len(entropy_values) > 0:
                entropy_mean = self._compute_entropy_stats(entropy_values)
                layer_name, head_idx = self._parse_head_key(head_key)

                if entropy_mean.item() > self.threshold:
                    if layer_name not in layer_heads:
                        layer_heads[layer_name] = []
                    layer_heads[layer_name].append(head_idx)

        return layer_heads

    def _group_entropy_by_layer(self):
        """Group entropy statistics by layer."""
        layer_heads = {}
        for head_key, entropy_values in self.entropy_stats.items():
            if len(entropy_values) > 0:
                entropy_mean = self._compute_entropy_stats(entropy_values)
                layer_name, head_idx = self._parse_head_key(head_key)

                if layer_name not in layer_heads:
                    layer_heads[layer_name] = []
                layer_heads[layer_name].append((head_idx, entropy_mean.item()))

        return layer_heads

    def _select_heads_by_percent(self, predictor, layer_heads, mask_size_fn):
        """
        Select heads based on percentage of highest/lowest entropy.

        Args:
            predictor: Model predictor
            layer_heads: Dictionary of layer_name -> [(head_idx, entropy_mean)]
            mask_size_fn: Function that takes layer_name and heads_with_entropy to compute mask size
        """
        final_stats = {}

        for layer_name, heads_with_entropy in layer_heads.items():
            if len(heads_with_entropy) > 0:
                # Sort heads by entropy
                heads_with_entropy.sort(
                    key=lambda x: x[1],
                    reverse=self.prunehighentropy
                )
                # import ipdb; ipdb.set_trace()
                # Calculate number of heads to select
                num_heads_to_select = self._calculate_num_heads_to_select(len(heads_with_entropy))
                selected_heads = heads_with_entropy[:num_heads_to_select]
                print("remain heads/ total heads:", len(heads_with_entropy)-num_heads_to_select, "/", len(heads_with_entropy))
                # Create mask
                mask_size = mask_size_fn(layer_name, heads_with_entropy)
                selected_indices = [head_idx for head_idx, _ in selected_heads]
                # mask = torch.isin(
                #     torch.arange(mask_size),
                #     torch.tensor(selected_indices)
                # ).to(predictor.device)
                # final_stats[layer_name] = mask
                
                elements = torch.arange(len(heads_with_entropy))
                test_elements = torch.tensor([head_idx for head_idx, _ in selected_heads])
                
                # Create mask using broadcasting
                mask = (elements.unsqueeze(1) == test_elements.unsqueeze(0)).any(dim=1).to(predictor.device)
                final_stats[layer_name] = mask

        return final_stats

    def _process_threshold_mode(self, predictor):
        """Process calibration in threshold mode. Override for custom behavior."""
        layer_heads = self._select_heads_by_threshold(predictor)
        self.final_entropy_stats = {}
        for layer_name, head_list in layer_heads.items():
            self.final_entropy_stats[layer_name] = head_list

    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_percent_mode")

    def calibrate(self, predictor, modules, num_samples=32):
        """Custom calibration that accumulates all entropy values, then calculates final statistics."""
        self.entropy_stats = defaultdict(list)
        attention_hooks, _ = self._register_hooks(predictor, modules)

        logger = setup_logger('./calib_logs', self.strategy_name)
        print('Collecting entropy values for all attention heads during calibration')

        self._run_calibration_loop(predictor, num_samples)

        # Remove hooks
        for hook in attention_hooks:
            hook.remove()

        # Calculate final statistics
        print('Calculating final entropy variance and mean from accumulated data...')

        if self.percent is None:
            self._process_threshold_mode(predictor)
        else:
            self._process_percent_mode(predictor)

        # self._log_final_stats()

    def _log_final_stats(self):
        """Log the final entropy statistics."""
        for layer_name, mask in list(self.final_entropy_stats.items()):
            if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
                num_selected = mask.sum().item()
                selected_indices = torch.where(mask)[0].tolist()
                print(f'Layer {layer_name}: {num_selected} heads marked: {selected_indices[:10]}{"..." if len(selected_indices) > 10 else ""}')
            else:
                print(f'Layer {layer_name}: {len(mask)} heads: {mask[:10] if len(mask) > 10 else mask}')

    def get_entropy_stats(self):
        """Return the collected entropy statistics."""
        return getattr(self, 'final_entropy_stats', {})

    def _compute_qkv(self, x, module):
        """Compute Q, K, V from input tensor."""
        B, H, W, _ = x.shape
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
        return q, k, v, B, H, W

    def _compute_attention(self, q, k, module, H, W):
        """Compute attention matrix from Q and K."""
        attn = (q * module.scale) @ k.transpose(-2, -1)
        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        return attn.softmax(dim=-1)

    def _reshape_output(self, x, B, num_heads, H, W):
        """Reshape attention output to original spatial dimensions."""
        return x.view(B, num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

class PruneRateProcessor(BaseEntropyProcessor):
    def __init__(self, strategy_name: str = 'PruneRate'):
        super().__init__(strategy_name)
    def set_params(self, args):
        """Set parameters from args. Override in subclasses for custom params."""
        self.threshold = 5.0
        self.percent = args.quantization.percent_entropy
        self.percent_global = args.quantization.percent_entropy_global
        self.prunehighentropy = args.quantization.high_entropy
        self.prune_global = args.quantization.prune_global
        self.model_type = args.model.model_type
    def calculate_entropy(self, attn_head):
        """Calculate mean entropy of the entire attention matrix."""
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        eps = 1e-12
        attn_head = torch.clamp(attn_head, min=eps).flatten()
        entropy = -torch.sum(attn_head * torch.log(attn_head))
        return entropy
    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode, sorting heads by entropy values."""
        layer_heads = self._group_entropy_by_layer()
        
        # Sort each layer's heads by entropy value in descending order
        sorted_layer_heads = {}
        for layer_name, heads_with_entropy in layer_heads.items():
            sorted_heads = sorted(heads_with_entropy, key=lambda x: x[1])
            sorted_layer_heads[layer_name] = [index for index, _ in sorted_heads]
        self.final_entropy_stats = sorted_layer_heads
        
        # Clean up the original entropy_stats to save memory
        del self.entropy_stats
    def _create_attention_hook(self, name):
        """Create attention hook for positional pruning."""
        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)
            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

            attn = attn.softmax(dim=-1)
            B_nhead, N, _ = attn.shape

            for head_idx in range(B_nhead):
                attn_head = attn[head_idx]
                mean_entropy = self.calculate_entropy(attn_head)
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(mean_entropy)

        return attention_hook
class EntropyValueCheck(BaseEntropyProcessor):
    def __init__(self, strategy_name: str = 'entropycheck'):
        super().__init__(strategy_name)
        self.global_percent = 0.5
    def set_params(self, args):
        pass
    def calculate_entropy(self, attn_head):
        """Calculate mean entropy of the entire attention matrix."""
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        eps = 1e-12
        attn_head = torch.clamp(attn_head, min=eps).flatten()
        entropy = -torch.sum(attn_head * torch.log(attn_head))
        return entropy
    def _create_attention_hook(self, name):
        """Create attention hook for positional pruning."""
        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)
            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

            attn = attn.softmax(dim=-1)
            B_nhead, N, _ = attn.shape

            for head_idx in range(B_nhead):
                attn_head = attn[head_idx]
                mean_entropy = self.calculate_entropy(attn_head)
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(mean_entropy)

        return attention_hook
    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode."""
        self.layer_heads = self._group_entropy_by_layer()
    
class PositionalPruneProcessor(BaseEntropyProcessor):
    """
    Processor that prunes attention heads based on mean entropy of their attention distribution.

    This processor:
    1. Computes a single mean entropy value for each head by treating the entire attention
       matrix as a flattened probability distribution
    2. Tracks entropy across calibration samples for each head position (B*num_heads index)
    3. Selects heads with highest/lowest entropy (based on percent_entropy) for pruning
    4. Replaces pruned head outputs with mean of value vectors

    The name "Positional" refers to indexing heads by their position in the batch*num_heads
    dimension (0-399 for 16 heads * 25 samples, for example), not positional encoding.
    """

    def __init__(self, strategy_name: str = 'PositionalHeadPruneProcessor'):
        super().__init__(strategy_name)
        self.global_percent = 0.5

    def set_params(self, args):
        super().set_params(args)

    def calculate_entropy(self, attn_head):
        """Calculate mean entropy of the entire attention matrix."""
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        eps = 1e-12
        attn_head = torch.clamp(attn_head, min=eps).flatten()
        entropy = -torch.sum(attn_head * torch.log(attn_head))
        return entropy

    def _calculate_num_heads_to_select(self, total_heads):
        """Override to support global vs local percentages."""
       
        if self.model_type == "vit_b" :
            if total_heads < 300:
                return  int(total_heads * self.global_percent)
        elif self.model_type == "vit_l" :
            if total_heads < 400:
                return int(total_heads * self.global_percent)
        elif self.model_type == "vit_h" :
            if total_heads < 400:
                return  int(total_heads * self.global_percent)
        
        return int(total_heads * self.percent)

    def _create_attention_hook(self, name):
        """Create attention hook for positional pruning."""
        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)
            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

            attn = attn.softmax(dim=-1)
            B_nhead, N, _ = attn.shape

            for head_idx in range(B_nhead):
                attn_head = attn[head_idx]
                mean_entropy = self.calculate_entropy(attn_head)
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(mean_entropy)

        return attention_hook

    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode."""
        layer_heads = self._group_entropy_by_layer()
        mask_size_fn = lambda layer_name, heads: len(heads)
        self.final_entropy_stats = self._select_heads_by_percent(predictor, layer_heads, mask_size_fn)

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Standard attention processing with optional head pruning."""
        # Determine if we should prune this layer
        if not self.prune_global:
            if self.model_type == "vit_b" :
                if not any(num in module_name for num in ["2", "5", "8", "11"]):
                    prune_mask = module.processor.final_entropy_stats.get(module_name, None)
                else:
                    prune_mask = None
            elif self.model_type =="vit_l":
                if not any(num in module_name for num in [".5", "11", "17", "23"]):
                    prune_mask = module.processor.final_entropy_stats.get(module_name, None)
                else:
                    prune_mask = None
            elif self.model_type == "vit_h":
                if not any(num in module_name for num in [".7", "15", "23", "31"]):
                    prune_mask = module.processor.final_entropy_stats.get(module_name, None)
                else:
                    prune_mask = None
        else:
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)

        q, k, v, B, H, W = self._compute_qkv(x, module)

        # Apply pruning mask if available
        if prune_mask is not None:
            prune_mask = prune_mask.repeat(q.shape[0] // prune_mask.shape[0])
            q_attn = q[~prune_mask, :, :]
            k_attn = k[~prune_mask, :, :]
            v_attn = v[~prune_mask, :, :]
            v_pruned = v[prune_mask, :, :]
        else:
            q_attn, k_attn, v_attn = q, k, v

        # Compute attention
        attn = self._compute_attention(q_attn, k_attn, module, H, W)
        x_attn = attn @ v_attn
        # Merge outputs
        if prune_mask is not None:
            x = torch.zeros_like(v).to(v.device)
            x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        x = self._reshape_output(x, B, module.num_heads, H, W)
        x = module.proj(x)
        return x


class HeadPruneProcessor(BaseEntropyProcessor):
    """Processor that identifies and prunes attention heads based on entropy."""

    def __init__(self, strategy_name: str = 'head_prune'):
        super().__init__(strategy_name)
        self.percent_global = 0.5

    def set_params(self, args):
        super().set_params(args)
        self.percent_global = args.percent_global

    def calculate_entropy(self, attn_head):
        """Calculate entropy for each position in a single attention head."""
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        eps = 1e-12
        attn_normalized = torch.clamp(attn_head, min=eps)
        entropy_per_position = -torch.mean(attn_normalized * torch.log(attn_normalized), dim=-1)
        return entropy_per_position

    def _create_attention_hook(self, name):
        """Create attention hook for head pruning."""
        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)
            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

            attn = attn.softmax(dim=-1).reshape(module.num_heads, B, attn.shape[-2], -1)
            n_heads, B, N, _ = attn.shape

            for head_idx in range(n_heads):
                attn_head = attn[head_idx]
                entropy_per_position = self.calculate_entropy(attn_head)
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(entropy_per_position.mean(-1).mean(-1).item())

        return attention_hook

    def _process_threshold_mode(self, predictor):
        """Process calibration in threshold mode."""
        layer_heads = {}
        for head_key, stats in self.entropy_stats.items():
            if len(stats) > 0:
                entropy_mean = self._compute_entropy_stats(stats)
                layer_name, head_idx = self._parse_head_key(head_key)

                if entropy_mean.item() > self.threshold:
                    if layer_name not in layer_heads:
                        layer_heads[layer_name] = []
                    layer_heads[layer_name].append(head_idx)

        # Convert to masks
        self.final_entropy_stats = {}
        for layer_name, head_indices in layer_heads.items():
            mask = torch.isin(torch.arange(16), torch.tensor(head_indices)).to(predictor.device)
            self.final_entropy_stats[layer_name] = mask

    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode."""
        layer_heads = {}
        for head_key, stats in self.entropy_stats.items():
            if len(stats) > 0:
                entropy_mean = self._compute_entropy_stats(stats)
                layer_name, head_idx = self._parse_head_key(head_key)

                if layer_name not in layer_heads:
                    layer_heads[layer_name] = []
                layer_heads[layer_name].append((head_idx, entropy_mean.item()))

        # Select heads by percentage
        self.final_entropy_stats = {}
        for layer_name, heads_with_entropy in layer_heads.items():
            if len(heads_with_entropy) > 0:
                heads_with_entropy.sort(
                    key=lambda x: x[1],
                    reverse=self.prunehighentropy
                )

                num_heads_to_select = max(1, int(len(heads_with_entropy) * self.percent))
                selected_heads = heads_with_entropy[:num_heads_to_select]

                # Create mask for batch * heads indexing
                selected_head_indices = torch.tensor([head_idx for head_idx, _ in selected_heads])
                mask_indices = (torch.arange(25)[:, None] * 16 + selected_head_indices).flatten()
                mask = torch.isin(torch.arange(16*25), mask_indices).to(predictor.device)
                self.final_entropy_stats[layer_name] = mask

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Standard attention processing with optional head pruning."""
        if not any(num in module_name for num in ["5", "11", "17", "23"]):
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)
        else:
            prune_mask = None

        q, k, v, B, H, W = self._compute_qkv(x, module)

        if prune_mask is not None:
            prune_mask = prune_mask.repeat(q.shape[0] // prune_mask.shape[0])
            q = q[~prune_mask, :, :]
            k = k[~prune_mask, :, :]
            v_pruned = v[prune_mask, :, :]
            v = v[~prune_mask, :, :]

        attn = self._compute_attention(q, k, module, H, W)
        x_attn = attn @ v

        if prune_mask is not None:
            x = torch.zeros(B * module.num_heads, attn.shape[-2], v.shape[-1]).to(v.device)
            x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        x = self._reshape_output(x, B, module.num_heads, H, W)
        x = module.proj(x)
        return x


class PositionalQuantProcessor(BaseEntropyProcessor):
    """
    Processor that quantizes attention heads based on global entropy of their attention distribution.

    Similar to PositionalPruneProcessor, but instead of pruning high-entropy heads,
    this processor:
    1. Computes a single entropy value for each head by treating the entire attention
       matrix as a flattened probability distribution (using sum instead of mean)
    2. Tracks entropy across calibration samples for each head position (B*num_heads index)
    3. Selects heads with highest/lowest entropy (based on percent_entropy)
    4. Applies aggressive quantization (2-bit) to selected heads instead of pruning

    The name "Positional" refers to indexing heads by their position in the batch*num_heads
    dimension (0-399 for 16 heads * 25 samples, for example), not positional encoding.
    """

    def __init__(self, strategy_name: str = 'PositionalHeadPruneProcessor'):
        super().__init__(strategy_name)
        self.global_percent = 0.5
        self.prune_global = False

    def set_params(self, args):
        super().set_params(args)

    def calculate_entropy(self, attn_head):
        """Calculate global entropy of the entire attention matrix."""
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        eps = 1e-12
        attn_head = torch.clamp(attn_head, min=eps).flatten()
        entropy = -torch.mean(attn_head * torch.log(attn_head))
        return entropy

    def _calculate_num_heads_to_select(self, total_heads):
        """Override to support global vs local percentages."""
        if self.model_type == "vit_b" :
            if total_heads < 300:
                return int(total_heads * self.global_percent)
        elif self.model_type == "vit_l" :
            if total_heads < 400:
                return int(total_heads * self.global_percent)
        elif self.model_type == "vit_h" :
            if total_heads < 400:
                return int(total_heads * self.global_percent)
        
        return int(total_heads * self.percent)

    def _create_attention_hook(self, name):
        """Create attention hook for positional quantization."""
        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)
            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

            attn = attn.softmax(dim=-1)
            B_nhead, N, _ = attn.shape

            for head_idx in range(B_nhead):
                attn_head = attn[head_idx]
                mean_entropy = self.calculate_entropy(attn_head)
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(mean_entropy)

        return attention_hook

    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode."""
        layer_heads = self._group_entropy_by_layer()

        self.final_entropy_stats = {}
        for layer_name, heads_with_entropy in layer_heads.items():
            if len(heads_with_entropy) > 0:
                heads_with_entropy.sort(
                    key=lambda x: x[1],
                    reverse=self.prunehighentropy
                )

                num_heads_to_select = self._calculate_num_heads_to_select(len(heads_with_entropy))
                selected_heads = heads_with_entropy[:num_heads_to_select]
                print("remain heads/ total heads:", len(heads_with_entropy)-num_heads_to_select, "/", len(heads_with_entropy))
                # mask = torch.isin(
                #     torch.arange(len(heads_with_entropy)),
                #     torch.tensor([head_idx for head_idx, _ in selected_heads])
                # ).to(predictor.device)
                # self.final_entropy_stats[layer_name] = mask
                elements = torch.arange(len(heads_with_entropy))
                test_elements = torch.tensor([head_idx for head_idx, _ in selected_heads])
                
                # Create mask using broadcasting
                mask = (elements.unsqueeze(1) == test_elements.unsqueeze(0)).any(dim=1).to(predictor.device)
                self.final_entropy_stats[layer_name] = mask
    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Standard attention processing with optional head quantization."""
        # Determine if we should quantize this layer
        if not self.prune_global:
            if self.model_type == "vit_b" :
                if not any(num in module_name for num in ["2", "5", "8", "11"]):
                    prune_mask = module.processor.final_entropy_stats.get(module_name, None)
                else:
                    prune_mask = None
            elif self.model_type =="vit_l":
                if not any(num in module_name for num in ["5", "11", "17", "23"]):
                    prune_mask = module.processor.final_entropy_stats.get(module_name, None)
                else:
                    prune_mask = None
            elif self.model_type == "vit_h":
                if not any(num in module_name for num in ["7", "15", "23", "31"]):
                    prune_mask = module.processor.final_entropy_stats.get(module_name, None)
                else:
                    prune_mask = None
        else:
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)

        B, H, W, _ = x.shape
        
        
        if self.model_type == "vit_b" : 
            n_bits = 4 if not prune_mask.shape[0] == 300 else 2
        elif  self.model_type == "vit_l" :
            n_bits = 4 if not prune_mask.shape[0] == 400 else 2
        elif  self.model_type == "vit_h" :
            n_bits = 4 if not prune_mask.shape[0] == 400 else 2

        q, k, v, B, H, W = self._compute_qkv(x, module)

        if prune_mask is not None:
            prune_mask = prune_mask.repeat(q.shape[0] // prune_mask.shape[0])
            q_attn = q[~prune_mask, :, :]
            k_attn = k[~prune_mask, :, :]
            v_attn = v[~prune_mask, :, :]
            q_prune = quantize_activation_per_token_absmax(q[prune_mask, :, :], n_bits)
            k_prune = quantize_activation_per_token_absmax(k[prune_mask, :, :], n_bits)
            v_prune = quantize_activation_per_channel_absmax(v[prune_mask, :, :], n_bits)
        else:
            q_attn, k_attn, v_attn = q, k, v

        attn = self._compute_attention(q_attn, k_attn, module, H, W)
        x_attn = attn @ v_attn

        if prune_mask is not None:
            x = torch.zeros_like(v).to(v.device)
            attn_prune = ((q_prune * module.scale) @ k_prune.transpose(-2, -1)).softmax(dim=-1)
            x_prune = quantize_activation_per_token_absmax(attn_prune, n_bits) @ v_prune
            x[prune_mask] = x_prune
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        x = self._reshape_output(x, B, module.num_heads, H, W)
        x = module.proj(x)
        return x
