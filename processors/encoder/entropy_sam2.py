"""Attention head pruning processor for SAM2 quantization."""

import torch
import torch.nn as nn
from collections import defaultdict
from tqdm.auto import tqdm
import torch.nn.functional  as F
import math


from ..base import AttentionProcessor, setup_logger
# from utils.quant_utils import quantize_activation_per_channel_absmax, quantize_activation_per_token_absmax


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class BaseEntropySAM2Processor(AttentionProcessor):
    """
    Base class for entropy-based attention head processing for SAM2.

    Key differences from SAM version:
    - Works with SAM2's MultiScaleAttention using F.scaled_dot_product_attention
    - Handles different tensor shapes: (B, num_heads, H*W, dim) vs (B*num_heads, H*W, dim)
    - Computes attention manually since SDPA is a black box
    - Supports Hiera hierarchical backbone instead of ViT

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
        self.percent_global = 0.5
        self.prunehighentropy = True
        self.prune_global = True
        self._mask_cache = {}

    def set_params(self, args):
        """Set parameters from args. Override in subclasses for custom params."""
        self.threshold = args.quantization.threshold
        self.percent = args.quantization.percent_entropy
        self.percent_global = args.quantization.percent_entropy_global
        self.prunehighentropy = args.quantization.high_entropy
        self.prune_global = args.quantization.prune_global

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
        """
        Parse layer name and head index from key format.

        SAM2 format: 'trunk.blocks.0.attn.head_0'
        Returns: ('image_encoder.trunk.blocks.0.attn', 0)
        """
        # Split by '.' to separate module path and head identifier
        parts = head_key.rsplit('.', 1)  # Split from right, max 1 split
        module_path = parts[0]  # e.g., 'trunk.blocks.0.attn'
        head_part = parts[1]     # e.g., 'head_0'

        # Extract head index from 'head_X' format
        head_idx = int(head_part.split('_')[1])

        # Prepend 'image_encoder.' to get full layer name
        layer_name = f"image_encoder.{module_path}"

        return layer_name, head_idx

    def _compute_entropy_stats(self, entropy_values):
        """Convert entropy values to tensor and calculate mean."""
        entropy_tensor = torch.tensor(entropy_values)
        return torch.mean(entropy_tensor)

    def _calculate_num_heads_to_select(self, total_heads ):
        """Calculate number of heads to select based on percentage."""
        if not total_heads < 20: 
            return max(1, int(total_heads * self.percent))
        else:  
            return max(1, int(total_heads * self.global_percent))

    def _select_heads_by_threshold(self, predictor):
        """Select heads based on entropy threshold and return masks."""
        # First, group heads by layer and collect those exceeding threshold
        layer_heads_dict = {}
        layer_total_heads = {}

        for head_key, entropy_values in self.entropy_stats.items():
            if len(entropy_values) > 0:
                entropy_mean = self._compute_entropy_stats(entropy_values)
                layer_name, head_idx = self._parse_head_key(head_key)

                # Track total number of heads per layer
                if layer_name not in layer_total_heads:
                    layer_total_heads[layer_name] = 0
                layer_total_heads[layer_name] = max(layer_total_heads[layer_name], head_idx + 1)

                # Collect heads exceeding threshold
                if entropy_mean.item() > self.threshold:
                    if layer_name not in layer_heads_dict:
                        layer_heads_dict[layer_name] = []
                    layer_heads_dict[layer_name].append(head_idx)

        # Convert to boolean masks like _select_heads_by_percent
        final_stats = {}
        for layer_name, selected_indices in layer_heads_dict.items():
            num_heads = layer_total_heads[layer_name]
            mask = torch.isin(
                torch.arange(num_heads),
                torch.tensor(selected_indices)
            ).to(predictor.device)
            final_stats[layer_name] = mask

        return final_stats

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

                # Calculate number of heads to select
                num_heads_to_select = self._calculate_num_heads_to_select(len(heads_with_entropy),layer_name)
                selected_heads = heads_with_entropy[:num_heads_to_select]

                # Create mask
                mask_size = mask_size_fn(layer_name, heads_with_entropy)
                selected_indices = [head_idx for head_idx, _ in selected_heads]
                mask = torch.isin(
                    torch.arange(mask_size),
                    torch.tensor(selected_indices)
                ).to(predictor.device)
                final_stats[layer_name] = mask

        return final_stats

    def _process_threshold_mode(self, predictor):
        """Process calibration in threshold mode. Override for custom behavior."""
        self.final_entropy_stats = self._select_heads_by_threshold(predictor)

    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_percent_mode")

    def calibrate(self, predictor, modules, num_samples=32):
        """Custom calibration that accumulates all entropy values, then calculates final statistics."""
        self.entropy_stats = defaultdict(list)
        attention_hooks, _ = self._register_hooks(predictor, modules)

        logger = setup_logger('./calib_logs', self.strategy_name)
        print(f'______Using: {self.strategy_name}_______')
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

        self._log_final_stats()

    def _log_final_stats(self):
        """Log the final entropy statistics."""
        for layer_name, mask in list(self.final_entropy_stats.items()):
            if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
                num_selected = mask.sum().item()
                selected_indices = torch.where(mask)[0].tolist()
                print(f'Layer {layer_name}: {num_selected}/{len(mask)} heads marked: {selected_indices[:10]}{"..." if len(selected_indices) > 10 else ""}')
            else:
                print(f'Layer {layer_name}: {len(mask)} heads: {mask[:10] if len(mask) > 10 else mask}')

    def get_entropy_stats(self):
        """Return the collected entropy statistics."""
        return getattr(self, 'final_entropy_stats', {})

    def _compute_qkv_sam2(self, x, module):
        """
        Compute Q, K, V from input tensor for SAM2 MultiScaleAttention.

        SAM2 uses shape (B, H, W, C) and produces (B, num_heads, H*W, dim).
        """
        B, H, W, C = x.shape

        # Project to QKV: (B, H, W, C) -> (B, H*W, 3, num_heads, head_dim)
        # Use -1 to automatically compute head_dim from dim_out
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1)
        # Permute to (3, B, num_heads, H*W, dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # Unbind to get separate Q, K, V: each (B, num_heads, H*W, dim)
        q, k, v = qkv.unbind(0)

        return q, k, v, B, H, W

    def _compute_attention_sam2(self, q, k, module):
        """
        Compute attention matrix from Q and K for SAM2.

        SAM2 attention: (B, num_heads, H*W, H*W)
        """
        # Scale Q
        scale = (q.size(-1)) ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        return attn

    def _reshape_output_sam2(self, x, B, num_heads, H, W):
        """Reshape attention output to original spatial dimensions for SAM2."""
        # x: (B, num_heads, H*W, dim) -> (B, H, W, C)
        dim = x.size(-1)
        return x.transpose(1, 2).reshape(B, H, W, num_heads * dim)
    def scaled_dot_product_attention_m(self,query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
       
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value
         

    def process(self, x: torch.Tensor, module, module_name: str = None) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if module.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), module.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, module.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = module.proj(x)

        return x


class PositionalTrainingPruneRateSAM2Processor(BaseEntropySAM2Processor):
    def __init__(self, strategy_name: str = 'PositionalTrainingPruneRateSAM2Processor'):
        super().__init__(strategy_name)
    def set_params(self, args):
        super().set_params(args)
        self.global_percent = args.quantization.percent_entropy_global
        self.train_state= args.quantization.train_state
        self.model_type ='hiera_b_plus'
    def calculate_entropy(self, attn ):
    
        # attn B, num_heads, H*W, H*W
        eps = 1e-12
        B , H, T, _ = attn.shape 
        attn = torch.clamp(attn, min=eps).view(B, H, -1)
        entropy = -torch.mean(attn * torch.log(attn), dim=-1)
        return entropy
    def _process_percent_mode(self, predictor):
        if self.train_state :
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
        else :
            layer_heads = self._group_entropy_by_layer()
            # Sort each layer's heads by entropy value in descending order
            sorted_layer_heads = {}
            for layer_name, heads_with_entropy in layer_heads.items():
                sorted_heads = sorted(heads_with_entropy, key=lambda x: x[1])
                sorted_layer_heads[layer_name] = [index for index, _ in sorted_heads]
            self.final_entropy_stats = sorted_layer_heads
            # Clean up the original entropy_stats to save memory
            del self.entropy_stats


            # """Process calibration in percent mode."""
            # self.layer_heads = self._group_entropy_by_layer()
            # self.final_entropy_stats = {}
            
    def _calculate_num_heads_to_select(self, module ):
        return module.prune_ddp.update_kept_head_number()

    def recompute_masks(self,predictor):
        mask_size_fn = lambda layer_name, heads: len(heads)

        ###TODO: implement recalculate mask follow the number of kept head take from predictor here
        self.final_entropy_stats = self._select_heads_by_ddp(predictor, self.layer_heads, mask_size_fn)

        self._log_final_stats()

        del self.entropy_stats
        del self.layer_heads
    def _select_heads_by_ddp(self, predictor, layer_heads, mask_size_fn):
        """
        Select heads based on the number of heads to keep from each layer's prune_ddp module.

        Args:
            predictor: Model predictor
            layer_heads: Dictionary of layer_name -> [(head_idx, entropy_mean)]
            mask_size_fn: Function that takes layer_name and heads_with_entropy to compute mask size
        """
        final_stats = {}
        
        # Map layer names to actual module paths to access prune_ddp
        for layer_name, heads_with_entropy in layer_heads.items():
            if len(heads_with_entropy) > 0:
                # Sort heads by entropy
                heads_with_entropy.sort(
                    key=lambda x: x[1],
                    reverse=self.prunehighentropy
                )
                if any(suffix in layer_name for suffix in [".2.", ".5.", ".21."]):# dont prune q_pool layer
                    number_of_head_to_keep = len( heads_with_entropy)
                else:
                    # Extract block index from layer_name (e.g., 'image_encoder.trunk.blocks.0.attn' -> 0)
                    block_index = int(layer_name.split('.')[3])
                    
                    
                    # Get the corresponding module from the predictor
                    attention_module = predictor.model.image_encoder.trunk.blocks[block_index].attn
                    
                    # Get number of heads to keep from the prune_ddp module
                    number_of_head_to_keep = int(attention_module.prune_ddp.update_kept_head_number())
                
                # If we need to keep all heads, no pruning is needed
                if number_of_head_to_keep >= len(heads_with_entropy):
                    # Keep all heads - create mask with all False values (no heads pruned)
                    mask_size = mask_size_fn(layer_name, heads_with_entropy)
                    mask = torch.zeros(mask_size, dtype=torch.bool).to(predictor.device)
                    final_stats[layer_name] = mask
                    continue
                    
                # Determine which heads to prune (those with highest entropy if prunehighentropy=True)
                # We keep the first number_of_head_to_keep heads after sorting
                number_heads_to_prune = len(heads_with_entropy) - number_of_head_to_keep
                selected_heads = heads_with_entropy[:number_heads_to_prune]
                
                # Create mask
                mask_size = mask_size_fn(layer_name, heads_with_entropy)
                selected_indices = [head_idx for head_idx, _ in selected_heads]
                mask = torch.isin(
                    torch.arange(mask_size),
                    torch.tensor(selected_indices)
                ).to(predictor.device)
                final_stats[layer_name] = mask
        return final_stats


    def _create_attention_hook(self, name):
        """Create attention hook for positional pruning in SAM2."""
        def attention_hook(module, input, output):
            
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, C = x.shape

            # Compute QKV using SAM2 format
            # Use -1 to automatically compute head_dim from dim_out
            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2,0,3,1,4)
            q, k, v = torch.unbind(qkv, 0)

            # Compute attention manually (SDPA is black box)
            if module.q_pool:
                q = do_pool(q.permute(0,2,1,3).reshape(B, H, W, -1), module.q_pool)
                H, W = q.shape[1:3]  # downsampled shape
                q = q.permute(0,2,1,3).reshape(B,  module.num_heads, H*W, -1)


            scale = (q.size(-1)) ** -0.5
            attn = (q * scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

            # attn shape: (B, num_heads, H*W, H*W)
            B_batch, n_heads, N, _ = attn.shape

            # Iterate over batch and heads to compute entropy
            attn_entropy = self.calculate_entropy(attn).reshape(-1,1)
            for b in range(B_batch):
                for head_idx in range(n_heads):
                    global_head_idx = b * n_heads + head_idx
                    
                    head_key = f"{name}.head_{global_head_idx}"
                    self.entropy_stats[head_key].append(attn_entropy[global_head_idx].item())

        return attention_hook
    def process_val(self, x: torch.Tensor, module, module_name: str = None):
        """
        Standard attention processing with optional head pruning for SAM2.

        Args:
            x: Input tensor (B, H, W, C)
            module: MultiScaleAttention module
            module_name: Name of the module for mask lookup
        """
        # Skip pruning for layers with Q pooling (stage transition layers)
        if module.q_pool:
            return super(PositionalPruneSAM2Processor, self).process(x, module, module_name)

        # Determine if we should prune this layer
        if not self.prune_global:
            if not any(num in module_name for num in ["12", "16", "20"]):
                prune_mask = module.processor.final_entropy_stats.get(module_name, None)
            else:
                prune_mask = None
        else:
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)

        B, H, W, _ = x.shape

        # Compute QKV using SAM2 format
        # qkv with shape (B, H * W, 3, nHead, head_dim)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1)
        # Permute to (3, B, nheads, H*W, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # Reshape to (3, B*nheads, H*W, head_dim) for easier masking
        qkv = qkv.reshape(3, B * module.num_heads, H * W, -1)
        # Unbind to get q, k, v: each (B*nheads, H*W, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply pruning mask if available (same as SAM1)

        if prune_mask is not None and isinstance(prune_mask, torch.Tensor):
            # Repeat mask for batch dimension
            prune_mask = prune_mask.repeat(q.shape[0] // prune_mask.shape[0])
            # Select heads based on mask
            q_attn = q[~prune_mask, :, :].unsqueeze(1)
            k_attn = k[~prune_mask, :, :].unsqueeze(1)
            v_attn = v[~prune_mask, :, :].unsqueeze(1)
            v_pruned = v[prune_mask, :, :]
        else:
            q_attn, k_attn, v_attn = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    
        # x_attn = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
        x_attn = self.scaled_dot_product_attention_m(q_attn, k_attn, v_attn)
        x_attn = x_attn.reshape(-1, H * W, x_attn.size(-1))

        # Merge outputs if pruning was applied
        if prune_mask is not None and isinstance(prune_mask, torch.Tensor):
            x = torch.zeros_like(v).to(v.device)
            # Fill pruned heads with mean of V
            x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
            # Fill kept heads with attention output
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        x = x.reshape(B, module.num_heads, H * W, -1)
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        # Apply output projection
        x = module.proj(x)
        flops = 0
        return x ,flops
    
class PositionalPruneSAM2Processor(BaseEntropySAM2Processor):
    """
    SAM2 version of PositionalPruneProcessor.

    Processor that prunes attention heads based on mean entropy of their attention distribution.

    This processor:
    1. Computes a single mean entropy value for each head by treating the entire attention
       matrix as a flattened probability distribution
    2. Tracks entropy across calibration samples for each head position
    3. Selects heads with highest/lowest entropy (based on percent_entropy) for pruning
    4. Replaces pruned head outputs with mean of value vectors

    Adapted for SAM2's MultiScaleAttention architecture.
    """

    def __init__(self, strategy_name: str = 'PositionalHeadPruneSAM2Processor'):
        super().__init__(strategy_name)
        self.global_percent = 0.5

    def set_params(self, args):
        super().set_params(args)
        self.global_percent = args.quantization.percent_entropy_global
        self.percent_8heads = args.quantization.percent_8heads
        self.percent_200heads = args.quantization.percent_200heads
        self.percent_400heads = args.quantization.percent_400heads
        self.percent_2048heads = args.quantization.percent_2048heads
        self.percent_4096heads = args.quantization.percent_4096heads
    def calculate_entropy(self, attn ):

        # attn B, num_heads, H*W, H*W
        eps = 1e-12
        B , H, T, _ = attn.shape 
        attn = torch.clamp(attn, min=eps).view(B, H, -1)
        entropy = -torch.mean(attn * torch.log(attn), dim=-1)
        return entropy

    
    def _calculate_num_heads_to_select(self, total_heads,layer_name ):
        
        """Calculate number of heads to select based on percentage."""
        if any(suffix in layer_name for suffix in [".2.", ".5.", ".21."]):# dont prune q_pool layer
            return 0
        if total_heads == 8:
            return int(total_heads * self.percent_8heads)
        elif total_heads == 200:
            return int(total_heads * self.percent_200heads)
        elif total_heads == 400:
            return int(total_heads * self.percent_400heads)
        elif total_heads == 2048:
            return int(total_heads * self.percent_2048heads)
        elif total_heads == 4096:
            return int(total_heads * self.percent_4096heads)

    def _create_attention_hook(self, name):
        """Create attention hook for positional pruning in SAM2."""
        def attention_hook(module, input, output):
            
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, C = x.shape

            # Compute QKV using SAM2 format
            # Use -1 to automatically compute head_dim from dim_out
            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2,0,3,1,4)
            q, k, v = torch.unbind(qkv, 0)

            # Compute attention manually (SDPA is black box)
            if module.q_pool:
                q = do_pool(q.permute(0,2,1,3).reshape(B, H, W, -1), module.q_pool)
                H, W = q.shape[1:3]  # downsampled shape
                q = q.permute(0,2,1,3).reshape(B,  module.num_heads, H*W, -1)


            scale = (q.size(-1)) ** -0.5
            attn = (q * scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

            # attn shape: (B, num_heads, H*W, H*W)
            B_batch, n_heads, N, _ = attn.shape

            # Iterate over batch and heads to compute entropy
            attn_entropy = self.calculate_entropy(attn).reshape(-1,1)
            for b in range(B_batch):
                for head_idx in range(n_heads):
                    global_head_idx = b * n_heads + head_idx
                    
                    head_key = f"{name}.head_{global_head_idx}"
                    self.entropy_stats[head_key].append(attn_entropy[global_head_idx].item())

        return attention_hook

    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode."""
        layer_heads = self._group_entropy_by_layer()
        mask_size_fn = lambda layer_name, heads: len(heads)
        self.final_entropy_stats = self._select_heads_by_percent(predictor, layer_heads, mask_size_fn)

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """
        Standard attention processing with optional head pruning for SAM2.

        Args:
            x: Input tensor (B, H, W, C)
            module: MultiScaleAttention module
            module_name: Name of the module for mask lookup
        """
        # Skip pruning for layers with Q pooling (stage transition layers)
        if module.q_pool:
            return super(PositionalPruneSAM2Processor, self).process(x, module, module_name)

        # Determine if we should prune this layer
        if not self.prune_global:
            if not any(num in module_name for num in ["12", "16", "20"]):
                prune_mask = module.processor.final_entropy_stats.get(module_name, None)
            else:
                prune_mask = None
        else:
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)

        B, H, W, _ = x.shape

        # Compute QKV using SAM2 format
        # qkv with shape (B, H * W, 3, nHead, head_dim)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1)
        # Permute to (3, B, nheads, H*W, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # Reshape to (3, B*nheads, H*W, head_dim) for easier masking
        qkv = qkv.reshape(3, B * module.num_heads, H * W, -1)
        # Unbind to get q, k, v: each (B*nheads, H*W, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply pruning mask if available (same as SAM1)

        if prune_mask is not None and isinstance(prune_mask, torch.Tensor):
            # Repeat mask for batch dimension
            prune_mask = prune_mask.repeat(q.shape[0] // prune_mask.shape[0])
            # Select heads based on mask
            q_attn = q[~prune_mask, :, :].unsqueeze(1)
            k_attn = k[~prune_mask, :, :].unsqueeze(1)
            v_attn = v[~prune_mask, :, :].unsqueeze(1)
            v_pruned = v[prune_mask, :, :]
        else:
            q_attn, k_attn, v_attn = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    
        x_attn = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
        # x_attn = self.scaled_dot_product_attention_m(q_attn, k_attn, v_attn)
        x_attn = x_attn.reshape(-1, H * W, x_attn.size(-1))

        # Merge outputs if pruning was applied
        if prune_mask is not None and isinstance(prune_mask, torch.Tensor):
            x = torch.zeros_like(v).to(v.device)
            # Fill pruned heads with mean of V
            x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
            # Fill kept heads with attention output
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        x = x.reshape(B, module.num_heads, H * W, -1)
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        # Apply output projection
        x = module.proj(x)

        return x


class HeadPruneSAM2Processor(BaseEntropySAM2Processor):
    """
    SAM2 version of HeadPruneProcessor.

    Processor that identifies and prunes attention heads based on entropy.
    Adapted for SAM2's MultiScaleAttention architecture.
    """

    def __init__(self, strategy_name: str = 'head_prune_sam2'):
        super().__init__(strategy_name)
        self.percent_global = 0.5

    def set_params(self, args):
        super().set_params(args)
        self.percent_global = args.quantization.percent_global

    def calculate_entropy(self, attn_head):
        """Calculate entropy for each position in a single attention head."""
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        eps = 1e-12
        attn_normalized = torch.clamp(attn_head, min=eps)
        entropy_per_position = -torch.mean(attn_normalized * torch.log(attn_normalized), dim=-1)
        return entropy_per_position

    def _create_attention_hook(self, name):
        """Create attention hook for head pruning in SAM2."""
        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, C = x.shape

            # Compute QKV using SAM2 format
            # Use -1 to automatically compute head_dim from dim_out
            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # Each: (B, num_heads, H*W, dim)

            # Compute attention manually
            scale = (q.size(-1)) ** -0.5
            attn = (q * scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

            # attn shape: (B, num_heads, H*W, H*W)
            B_batch, n_heads, N, _ = attn.shape

            for head_idx in range(n_heads):
                # Average entropy across batch
                attn_head = attn[:, head_idx]  # (B, H*W, H*W)
                entropy_per_position = self.calculate_entropy(attn_head)
                head_key = f"{name}.head_{head_idx}"
                # Average over batch and positions
                self.entropy_stats[head_key].append(entropy_per_position.mean().item())

        return attention_hook

    def _process_threshold_mode(self, predictor):
        """Process calibration in threshold mode."""
        # Use the base class implementation which now returns masks
        self.final_entropy_stats = self._select_heads_by_threshold(predictor)

    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode."""
        layer_heads = {}
        breakpoint()
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

                # Create mask for the heads
                num_heads = len(heads_with_entropy)
                selected_head_indices = torch.tensor([head_idx for head_idx, _ in selected_heads])
                mask = torch.isin(torch.arange(num_heads), selected_head_indices).to(predictor.device)
                self.final_entropy_stats[layer_name] = mask

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """
        Standard attention processing with optional head pruning for SAM2.

        Args:
            x: Input tensor (B, H, W, C)
            module: MultiScaleAttention module
            module_name: Name of the module for mask lookup
        """
        # Skip pruning for layers with Q pooling (stage transition layers)
        if module.q_pool:
            return super(HeadPruneSAM2Processor, self).process(x, module, module_name)

        # Check if we should prune this layer
        if not any(num in module_name for num in ["12", "16", "20"]):
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)
        else:
            prune_mask = None

        B, H, W, _ = x.shape

        # Compute QKV using SAM2 format
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        qkv = qkv.reshape(3, B * module.num_heads, H * W, -1)
        q, k, v = qkv.unbind(0)

        # Apply pruning mask if available (same as SAM1)
        if prune_mask is not None and isinstance(prune_mask, torch.Tensor):
            prune_mask = prune_mask.repeat(q.shape[0] // prune_mask.shape[0])
            q_attn = q[~prune_mask, :, :]
            k_attn = k[~prune_mask, :, :]
            v_attn = v[~prune_mask, :, :]
            v_pruned = v[prune_mask, :, :]
        else:
            q_attn, k_attn, v_attn = q, k, v

        # Compute attention using F.scaled_dot_product_attention
        # Need to reshape for SDPA: (B, num_heads, seq_len, head_dim)
        if prune_mask is not None and isinstance(prune_mask, torch.Tensor):
            num_heads_kept = q_attn.shape[0] // B
            q_sdpa = q_attn.reshape(B, num_heads_kept, H * W, -1)
            k_sdpa = k_attn.reshape(B, num_heads_kept, H * W, -1)
            v_sdpa = v_attn.reshape(B, num_heads_kept, H * W, -1)
        else:
            q_sdpa = q_attn.reshape(B, module.num_heads, H * W, -1)
            k_sdpa = k_attn.reshape(B, module.num_heads, H * W, -1)
            v_sdpa = v_attn.reshape(B, module.num_heads, H * W, -1)

        # Apply SDPA
        x_attn = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        # Reshape back to (B*num_heads_kept, H*W, head_dim)
        x_attn = x_attn.reshape(-1, H * W, x_attn.size(-1))

        # Merge outputs if pruning was applied
        if prune_mask is not None and isinstance(prune_mask, torch.Tensor):
            x = torch.zeros_like(v).to(v.device)
            x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        # Reshape back to spatial format
        x = x.reshape(B, module.num_heads, H * W, -1)
        x = x.transpose(1, 2).reshape(B, H, W, -1)

        # Apply output projection
        x = module.proj(x)
        return x


class PositionalQuantSAM2Processor(BaseEntropySAM2Processor):
    """
    SAM2 version of PositionalQuantProcessor.

    Processor that quantizes attention heads based on global entropy of their attention distribution.

    Similar to PositionalPruneSAM2Processor, but instead of pruning high-entropy heads,
    this processor:
    1. Computes a single entropy value for each head by treating the entire attention
       matrix as a flattened probability distribution
    2. Tracks entropy across calibration samples for each head position
    3. Selects heads with highest/lowest entropy (based on percent_entropy)
    4. Applies aggressive quantization (2-bit) to selected heads instead of pruning

    Adapted for SAM2's MultiScaleAttention architecture.
    """

    def __init__(self, strategy_name: str = 'PositionalHeadQuantSAM2Processor'):
        super().__init__(strategy_name)
        self.global_percent = 0.5
        self.prune_global = False

    def set_params(self, args):
        super().set_params(args)
        self.global_percent = args.quantization.percent_entropy_global

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
        if total_heads < 400:
            return max(1, int(total_heads * self.global_percent))
        else:
            return max(1, int(total_heads * self.percent))

    def _create_attention_hook(self, name):
        """Create attention hook for positional quantization in SAM2."""
        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, C = x.shape

            # Compute QKV using SAM2 format
            # Use -1 to automatically compute head_dim from dim_out
            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            # Compute attention manually
            scale = (q.size(-1)) ** -0.5
            attn = (q * scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

            B_batch, n_heads, N, _ = attn.shape

            for b in range(B_batch):
                for head_idx in range(n_heads):
                    attn_head = attn[b, head_idx]
                    mean_entropy = self.calculate_entropy(attn_head)
                    global_head_idx = b * n_heads + head_idx
                    head_key = f"{name}.head_{global_head_idx}"
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

                mask = torch.isin(
                    torch.arange(len(heads_with_entropy)),
                    torch.tensor([head_idx for head_idx, _ in selected_heads])
                ).to(predictor.device)
                self.final_entropy_stats[layer_name] = mask

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """
        Standard attention processing with optional head quantization for SAM2.

        Args:
            x: Input tensor (B, H, W, C)
            module: MultiScaleAttention module
            module_name: Name of the module for mask lookup
        """
        # Determine if we should quantize this layer
        if not self.prune_global:
            if not any(num in module_name for num in ["5", "11", "17", "23"]):
                quant_mask = module.processor.final_entropy_stats.get(module_name, None)
            else:
                quant_mask = None
        else:
            quant_mask = module.processor.final_entropy_stats.get(module_name, None)
        
        B, H_spatial, W_spatial, _ = x.shape
        n_bits = 4 if quant_mask is None or quant_mask.shape[0] != 400 else 2

        q, k, v, B, H, W = self._compute_qkv_sam2(x, module)

        if quant_mask is not None:
            # Reshape to flat head dimension
            q_flat = q.reshape(B * module.num_heads, H * W, -1)
            k_flat = k.reshape(B * module.num_heads, H * W, -1)
            v_flat = v.reshape(B * module.num_heads, H * W, -1)

            # Expand mask
            quant_mask_expanded = quant_mask.repeat(B)

            # Split into quantized and non-quantized heads
            q_attn = q_flat[~quant_mask_expanded, :, :]
            k_attn = k_flat[~quant_mask_expanded, :, :]
            v_attn = v_flat[~quant_mask_expanded, :, :]

            q_quant = quantize_activation_per_token_absmax(q_flat[quant_mask_expanded, :, :], n_bits)
            k_quant = quantize_activation_per_token_absmax(k_flat[quant_mask_expanded, :, :], n_bits)
            v_quant = quantize_activation_per_channel_absmax(v_flat[quant_mask_expanded, :, :], n_bits)

            # Compute attention for non-quantized heads
            scale = (q_attn.size(-1)) ** -0.5
            attn = (q_attn * scale) @ k_attn.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x_attn = attn @ v_attn

            # Compute attention for quantized heads
            attn_quant = ((q_quant * scale) @ k_quant.transpose(-2, -1)).softmax(dim=-1)
            attn_quant = quantize_activation_per_token_absmax(attn_quant, n_bits)
            x_quant = attn_quant @ v_quant

            # Merge outputs
            x_out = torch.zeros_like(v_flat).to(v.device)
            x_out[quant_mask_expanded] = x_quant
            x_out[~quant_mask_expanded] = x_attn

            # Reshape back
            x_out = x_out.reshape(B, module.num_heads, H * W, -1)
        else:
            # Standard attention
            attn = self._compute_attention_sam2(q, k, module)
            x_out = attn @ v

        x_out = self._reshape_output_sam2(x_out, B, module.num_heads, H, W)
        x_out = module.proj(x_out)
        return x_out
