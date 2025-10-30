"""Attention head pruning processor for SAM quantization."""

import torch
from functools import partial
from collections import defaultdict
from tqdm.auto import tqdm

from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from ..base import AttentionProcessor, setup_logger
from torch.distributions import Exponential
from quant_utils import quantize_activation_per_channel_absmax, quantize_activation_per_token_absmax 


class PositionalPruneProcessor(AttentionProcessor):
    """Processor that identifies and prunes attention heads based on entropy."""

    def __init__(self, strategy_name: str = 'PositionalHeadPruneProcessor'):
        super().__init__(strategy_name)
        self.stat = {}
        # Store all entropy values per position for each head
        self.entropy_stats = defaultdict(list)
        self.threshold = 5.0
        self.percent = 0.5 
        self.prunehighentropy = True 
        # Cache for masks to avoid recomputation
        self._mask_cache = {}

    def set_params(self, args):
        self.threshold = 5.0
        self.percent = args.quantization.percent_entropy
        self.prunehighentropy = args.quantization.high_entropy

    def calculate_entropy(self, attn_head):
        """Calculate entropy for each position in a single attention head"""
        # Convert to tensor if needed
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        # Ensure attention is normalized (should already be after softmax)
        eps = 1e-12
        attn_head = torch.clamp(attn_head, min=eps).flatten()
        entropy  =  -torch.sum(attn_head * torch.log(attn_head)) 

        return entropy

    def _register_hooks(self, predictor, modules):
        """Register hooks to capture attention patterns during forward pass"""
        from segment_anything.modeling.image_encoder import Attention

        def attention_hook(module, input, output, name):
            """Hook to capture attention patterns after softmax"""
            # Get input tensor
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            # Recompute attention to get the attention matrix
            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)

            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

            attn = attn.softmax(dim=-1)
            B_nhead, N, _ = attn.shape  # B * num_heads, N, N

            for head_idx in range(B_nhead):
                attn_head = attn[head_idx]  # Shape: (N, N)

                # Calculate entropy for each position
                entropy_per_position = self.calculate_entropy(attn_head)

                # Accumulate entropy values for this head
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(entropy_per_position)

        hooks = []

        # Register hooks for all encoder attention modules
        for name, component in predictor.model.image_encoder.named_modules():
            if isinstance(component, modules):
                print(f"Registering attention hook for {name}")
                hooks.append(
                    component.register_forward_hook(
                        partial(attention_hook, name=name)
                    )
                )
        return hooks, []  # Return (attention_hooks, linear_hooks)

    def _run_forward_pass(self, predictor, data_val):
        """Run forward pass through image encoder only"""
        imgs = data_val['image'].permute(0, 2, 3, 1).cpu().numpy()
        predictor.set_image(imgs.squeeze())

    def calibrate(self, predictor, modules, num_samples=32):
        """
        Custom calibration that accumulates all entropy values, then calculates final statistics
        """
        # Reset entropy statistics - store all entropy values
        self.entropy_stats = defaultdict(list)

        # Register hooks (only attention hooks needed)
        attention_hooks, _ = self._register_hooks(predictor, modules)

        logger = setup_logger('./calib_logs', self.strategy_name)
        print(f'______Using: {self.strategy_name}_______')
        print('Collecting entropy values for all attention heads during calibration')
        # Run calibration - accumulate all entropy values
        total_processed = 0
        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print(f'Dataloader {k} length:', len(dataloader))
            progress_bar = tqdm(total=min(num_samples, len(dataloader)),
                                desc=f"Collecting entropy data")

            for i, data_val in enumerate(dataloader):
                if total_processed >= num_samples:
                    break

                # Run forward pass to trigger hooks and accumulate entropy values
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

        # Remove hooks
        for hook in attention_hooks:
            hook.remove()

        # Calculate final statistics from all accumulated entropy values
        print('Calculating final entropy variance and mean from accumulated data...')
        self.final_entropy_stats = {}
        if self.percent is None:
            for head_key, entropy_values in self.entropy_stats.items():
                if len(entropy_values) > 0:
                    # Convert to tensor for calculation
                    entropy_tensor = torch.tensor(entropy_values)

                    # Calculate mean
                    entropy_mean = torch.mean(entropy_tensor)

                    # Parse layer and head index from key format: 'blocks.22.attn.head_233'
                    parts = head_key.split('.')
                    layer_name = f"image_encoder.{parts[0]}.{parts[1]}.{parts[2]}"
                    head_idx = int(parts[3].split('_')[1])  # Extract head number

                    # Only store heads whose entropy mean is greater than threshold
                    if entropy_mean.item() > self.threshold:
                        if layer_name not in self.final_entropy_stats:
                            self.final_entropy_stats[layer_name] = []
                        self.final_entropy_stats[layer_name].append(head_idx)

                mask = torch.isin(torch.arange(400), torch.tensor(
                    [head_idx for head_idx, _ in selected_heads]
                )).to(predictor.device)
                self.final_entropy_stats[layer_name] = mask
        else:
            # Group entropy stats by layer
            layer_heads = {}
            for head_key, entropy_values in self.entropy_stats.items():
                if len(entropy_values) > 0:
                    # Convert to tensor and calculate mean
                    entropy_tensor = torch.tensor(entropy_values)
                    entropy_mean = torch.mean(entropy_tensor)

                    # Parse layer and head index from key format: 'blocks.22.attn.head_233'
                    parts = head_key.split('.')
                    layer_name = f"image_encoder.{parts[0]}.{parts[1]}.{parts[2]}"
                    head_idx = int(parts[3].split('_')[1])

                    if layer_name not in layer_heads:
                        layer_heads[layer_name] = []
                    layer_heads[layer_name].append((head_idx, entropy_mean.item()))

            # For each layer, select top percent of heads with highest entropy
            for layer_name, heads_with_entropy in layer_heads.items():
                if len(heads_with_entropy) > 0:
                    # Sort heads by entropy mean in descending order
                    if self.prunehighentropy:
                        heads_with_entropy.sort(key=lambda x: x[1], reverse=True)
                    else:
                        heads_with_entropy.sort(key=lambda x: x[1])
                    # Calculate number of heads to select based on percentage
                    num_heads_to_select = max(1, int(len(heads_with_entropy) * self.percent))

                    # Select top percent of heads
                    selected_heads = heads_with_entropy[:num_heads_to_select]

                    # Store only the head indices
                    mask = torch.isin(torch.arange(400), torch.tensor(
                        [head_idx for head_idx, _ in selected_heads]
                    )).to(predictor.device)
                    self.final_entropy_stats[layer_name] = mask

                    

        # Log sample statistics
        for layer_name, head_list in list(self.final_entropy_stats.items()):
            print(f'Layer {layer_name}: {len(head_list)} heads with high entropy: {head_list[:10]}{"..." if len(head_list) > 10 else ""}')


    def get_entropy_stats(self):
        """Return the collected entropy statistics"""
        return getattr(self, 'final_entropy_stats', {})


    
        
        

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Standard attention processing with optional head pruning"""
        if not any(num in module_name for num in ["5", "11", "17", "23"]):  # modules whose shape is (16, 4096, 4096)
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)
        else:
            prune_mask  = None 

        B, H, W, _ = x.shape
        # Standard attention computation
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
        if prune_mask is not None:
            q_attn = q[~prune_mask, :, :]
            k_attn = k[~prune_mask, :, :]
            v_attn = v[~prune_mask, :, :]
            # q_prune= quantize_activation_per_token_absmax(q[prune_mask, :, :],2)
            # k_prune= quantize_activation_per_token_absmax(k[prune_mask, :, :],2)
            v_pruned = v[prune_mask, :, :]
        else:
            q_attn = q
            k_attn = k
            v_attn = v

        attn = (q_attn * module.scale) @ k_attn.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q_attn, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        B_nhead, N, _ = attn.shape  # B * num_heads, N, N

        x_attn = attn  @ v_attn
        if prune_mask is not None:
            x = torch.zeros_like(v).to(v.device)
            # attn_prune =  ((q_prune * module.scale) @ k_prune.transpose(-2, -1)).softmax(dim=-1)
            # x_prune = quantize_activation_per_token_absmax(attn_prune, 2) @ v_pruned
            # x[prune_mask] = x_prune 
            x[prune_mask] =  v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        x = x.view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)
        return x





class HeadPruneProcessor(AttentionProcessor):
    """Processor that identifies and prunes attention heads based on entropy."""

    def __init__(self, strategy_name: str = 'head_prune'):
        super().__init__(strategy_name)
        self.stat = {}
        # Store all entropy values per position for each head
        self.entropy_stats = defaultdict(list)
        self.threshold = 5.0
        self.percent = 0.5 
        self.prunehighentropy = True 
        # Cache for masks to avoid recomputation
        self._mask_cache = {}

    def set_params(self, args):
        self.threshold = 5.0
        self.percent = args.quantization.percent_entropy
        self.prunehighentropy = args.quantization.high_entropy

    def calculate_entropy(self, attn_head):
        """Calculate entropy for each position in a single attention head"""
        # Convert to tensor if needed
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        # Ensure attention is normalized (should already be after softmax)
        eps = 1e-12
        attn_normalized = torch.clamp(attn_head, min=eps)

        # Calculate entropy for each query position (each row)
        breakpoint()
        entropy_per_position = -torch.sum(attn_normalized * torch.log(attn_normalized), dim=-1)

        return entropy_per_position

    def _register_hooks(self, predictor, modules):
        """Register hooks to capture attention patterns during forward pass"""
        from segment_anything.modeling.image_encoder import Attention

        def attention_hook(module, input, output, name):
            """Hook to capture attention patterns after softmax"""
            # Get input tensor
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            # Recompute attention to get the attention matrix
            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)

            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

            attn = attn.softmax(dim=-1).reshape(module.num_heads, B, attn.shape[-2], -1)
            n_heads, B , N, _ = attn.shape  # B * num_heads, N, N
            

            for head_idx in range(n_heads):
                attn_head = attn[head_idx]  # Shape: (B, N, N)

                # Calculate entropy for each position
                entropy_per_position = self.calculate_entropy(attn_head)
                # Accumulate entropy values for this head
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(entropy_per_position.mean(-1).mean(-1).item())

        hooks = []

        # Register hooks for all encoder attention modules
        for name, component in predictor.model.image_encoder.named_modules():
            if isinstance(component, modules):
                print(f"Registering attention hook for {name}")
                hooks.append(
                    component.register_forward_hook(
                        partial(attention_hook, name=name)
                    )
                )
        return hooks, []  # Return (attention_hooks, linear_hooks)

    def _run_forward_pass(self, predictor, data_val):
        """Run forward pass through image encoder only"""
        imgs = data_val['image'].permute(0, 2, 3, 1).cpu().numpy()
        predictor.set_image(imgs.squeeze())

    def calibrate(self, predictor, modules, num_samples=32):
        """
        Custom calibration that accumulates all entropy values, then calculates final statistics
        """
        # Reset entropy statistics - store all entropy values
        self.entropy_stats = defaultdict(list)
        # Register hooks (only attention hooks needed)
        attention_hooks, _ = self._register_hooks(predictor, modules)

        logger = setup_logger('./calib_logs', self.strategy_name)
        print(f'______Using: {self.strategy_name}_______')
        print('Collecting entropy values for all attention heads during calibration')
        # Run calibration - accumulate all entropy values
        total_processed = 0
        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print(f'Dataloader {k} length:', len(dataloader))
            progress_bar = tqdm(total=min(num_samples, len(dataloader)), desc=f"Collecting entropy data")
            for i, data_val in enumerate(dataloader):
                if total_processed >= num_samples:
                    break
                # Run forward pass to trigger hooks and accumulate entropy values
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

        # Remove hooks
        for hook in attention_hooks:
            hook.remove()

        # Calculate final statistics from all accumulated entropy values
        print('Calculating final entropy variance and mean from accumulated data...')
        self.final_entropy_stats = {}
        if self.percent is None:
            # Threshold-based mode: collect high-entropy heads per layer
            layer_heads = {}
            for head_key, stats in self.entropy_stats.items():
                entropy_values = stats
                if len(entropy_values) > 0:
                    # Convert to tensor for calculation
                    entropy_tensor = torch.tensor(entropy_values)
                    # Calculate mean
                    entropy_mean = torch.mean(entropy_tensor)

                    # Parse layer and head index from key format: 'blocks.22.attn.head_233'
                    parts = head_key.split('.')
                    layer_name = f"image_encoder.{parts[0]}.{parts[1]}.{parts[2]}"
                    head_idx = int(parts[3].split('_')[1])  # Extract head number

                    # Only store heads whose entropy mean is greater than threshold
                    if entropy_mean.item() > self.threshold:
                        if layer_name not in layer_heads:
                            layer_heads[layer_name] = []
                        layer_heads[layer_name].append(head_idx)

            # Convert head lists to masks
            for layer_name, head_indices in layer_heads.items():
                mask = torch.isin(torch.arange(16), torch.tensor(head_indices)).to(predictor.device)
                self.final_entropy_stats[layer_name] = mask
        else:
            # Group entropy stats by layer
            layer_heads = {}
            for head_key, stats in self.entropy_stats.items():
                entropy_values = stats
                if len(entropy_values) > 0:
                    # Convert to tensor and calculate mean
                    entropy_mean = torch.tensor(entropy_values).mean()

                    # Parse layer and head index from key format: 'blocks.22.attn.head_233'
                    parts = head_key.split('.')
                    layer_name = f"image_encoder.{parts[0]}.{parts[1]}.{parts[2]}"
                    head_idx = int(parts[3].split('_')[1])

                    if layer_name not in layer_heads:
                        layer_heads[layer_name] = []
                    layer_heads[layer_name].append((head_idx, entropy_mean.item()))

            # For each layer, select top percent of heads with highest entropy
            for layer_name, heads_with_entropy in layer_heads.items():
                if len(heads_with_entropy) > 0:
                    # Sort heads by entropy mean in descending order
                    if self.prunehighentropy:
                        heads_with_entropy.sort(key=lambda x: x[1], reverse=True)
                    else:
                        heads_with_entropy.sort(key=lambda x: x[1], reverse=False)
                    # Calculate number of heads to select based on percentage
                    num_heads_to_select = max(1, int(len(heads_with_entropy) * self.percent))

                    # Select top percent of heads
                    selected_heads = heads_with_entropy[:num_heads_to_select]
                    

                    # Store only the head indices
                    # mask = torch.isin(torch.arange(16*25), torch.tensor(
                    #     [head_idx if head_idx % 25 in selected_heads else 0 for head_idx, _ in range(16*25)]
                    # )).to(predictor.device)
                    selected_head_indices = torch.tensor([head_idx for head_idx, _ in selected_heads])
                    mask_indices = (torch.arange(25)[:, None] * 16 + selected_head_indices).flatten()
                    mask = torch.isin(torch.arange(16*25), mask_indices).to(predictor.device)
                    self.final_entropy_stats[layer_name] = mask

                    

        # Log sample statistics
        for layer_name, mask in list(self.final_entropy_stats.items()):
            num_pruned = mask.sum().item()
            pruned_indices = torch.where(mask)[0].tolist()
            print(f'Layer {layer_name}: {num_pruned} heads marked for pruning: {pruned_indices[:10]}{"..." if len(pruned_indices) > 10 else ""}')


    def get_entropy_stats(self):
        """Return the collected entropy statistics"""
        return getattr(self, 'final_entropy_stats', {})
        

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Standard attention processing with optional head pruning"""
        if not any(num in module_name for num in ["5", "11", "17", "23"]):  # modules whose shape is (16, 4096, 4096)
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)
        else:
            prune_mask  = None 

        B, H, W, _ = x.shape
        # Standard attention computation
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
        if prune_mask is not None:
            q = quantize_activation_per_token_absmax(q[~prune_mask, :, :], n_bits=4)
            k = quantize_activation_per_token_absmax(k[~prune_mask, :, :], n_bits=4)
            v = quantize_activation_per_token_absmax(v[~prune_mask, :, :], n_bits=4)
            v_pruned = quantize_activation_per_token_absmax(v[prune_mask, :, :], n_bits=2)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        B_nhead, N, _ = attn.shape  # B * num_heads, N, N

        x_attn = attn @ v 
        if prune_mask is not None:
            x = torch.zeros_like(v).to(v.device)
            x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        x = x.view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)
        return x




class PositionalQuantProcessor(AttentionProcessor):
    """Processor that identifies and prunes attention heads based on entropy."""

    def __init__(self, strategy_name: str = 'PositionalHeadPruneProcessor'):
        super().__init__(strategy_name)
        self.stat = {}
        # Store all entropy values per position for each head
        self.entropy_stats = defaultdict(list)
        self.threshold = 5.0
        self.percent = 0.5 
        self.prunehighentropy = True 
        # Cache for masks to avoid recomputation
        self._mask_cache = {}

    def set_params(self, args):
        self.threshold = 5.0
        self.percent = args.quantization.percent_entropy
        self.prunehighentropy = args.quantization.high_entropy

    def calculate_entropy(self, attn_head):
        """Calculate entropy for each position in a single attention head"""
        # Convert to tensor if needed
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        # Ensure attention is normalized (should already be after softmax)
        eps = 1e-12
        attn_head = torch.clamp(attn_head, min=eps).flatten()
        entropy  =  -torch.sum(attn_head * torch.log(attn_head)) 

        return entropy

    def _register_hooks(self, predictor, modules):
        """Register hooks to capture attention patterns during forward pass"""
        from segment_anything.modeling.image_encoder import Attention

        def attention_hook(module, input, output, name):
            """Hook to capture attention patterns after softmax"""
            # Get input tensor
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            # Recompute attention to get the attention matrix
            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)

            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

            attn = attn.softmax(dim=-1)
            B_nhead, N, _ = attn.shape  # B * num_heads, N, N

            for head_idx in range(B_nhead):
                attn_head = attn[head_idx]  # Shape: (N, N)

                # Calculate entropy for each position
                entropy_per_position = self.calculate_entropy(attn_head)

                # Accumulate entropy values for this head
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(entropy_per_position)

        hooks = []

        # Register hooks for all encoder attention modules
        for name, component in predictor.model.image_encoder.named_modules():
            if isinstance(component, modules):
                print(f"Registering attention hook for {name}")
                hooks.append(
                    component.register_forward_hook(
                        partial(attention_hook, name=name)
                    )
                )
        return hooks, []  # Return (attention_hooks, linear_hooks)

    def _run_forward_pass(self, predictor, data_val):
        """Run forward pass through image encoder only"""
        imgs = data_val['image'].permute(0, 2, 3, 1).cpu().numpy()
        predictor.set_image(imgs.squeeze())

    def calibrate(self, predictor, modules, num_samples=32):
        """
        Custom calibration that accumulates all entropy values, then calculates final statistics
        """
        # Reset entropy statistics - store all entropy values
        self.entropy_stats = defaultdict(list)

        # Register hooks (only attention hooks needed)
        attention_hooks, _ = self._register_hooks(predictor, modules)

        logger = setup_logger('./calib_logs', self.strategy_name)
        print(f'______Using: {self.strategy_name}_______')
        print('Collecting entropy values for all attention heads during calibration')
        # Run calibration - accumulate all entropy values
        total_processed = 0
        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print(f'Dataloader {k} length:', len(dataloader))
            progress_bar = tqdm(total=min(num_samples, len(dataloader)),
                                desc=f"Collecting entropy data")

            for i, data_val in enumerate(dataloader):
                if total_processed >= num_samples:
                    break

                # Run forward pass to trigger hooks and accumulate entropy values
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

        # Remove hooks
        for hook in attention_hooks:
            hook.remove()

        # Calculate final statistics from all accumulated entropy values
        print('Calculating final entropy variance and mean from accumulated data...')
        self.final_entropy_stats = {}
        if self.percent is None:
            for head_key, entropy_values in self.entropy_stats.items():
                if len(entropy_values) > 0:
                    # Convert to tensor for calculation
                    entropy_tensor = torch.tensor(entropy_values)

                    # Calculate mean
                    entropy_mean = torch.mean(entropy_tensor)

                    # Parse layer and head index from key format: 'blocks.22.attn.head_233'
                    parts = head_key.split('.')
                    layer_name = f"image_encoder.{parts[0]}.{parts[1]}.{parts[2]}"
                    head_idx = int(parts[3].split('_')[1])  # Extract head number

                    # Only store heads whose entropy mean is greater than threshold
                    if entropy_mean.item() > self.threshold:
                        if layer_name not in self.final_entropy_stats:
                            self.final_entropy_stats[layer_name] = []
                        self.final_entropy_stats[layer_name].append(head_idx)

                mask = torch.isin(torch.arange(400), torch.tensor(
                    [head_idx for head_idx, _ in selected_heads]
                )).to(predictor.device)
                self.final_entropy_stats[layer_name] = mask
        else:
            # Group entropy stats by layer
            layer_heads = {}
            for head_key, entropy_values in self.entropy_stats.items():
                if len(entropy_values) > 0:
                    # Convert to tensor and calculate mean
                    entropy_tensor = torch.tensor(entropy_values)
                    entropy_mean = torch.mean(entropy_tensor)

                    # Parse layer and head index from key format: 'blocks.22.attn.head_233'
                    parts = head_key.split('.')
                    layer_name = f"image_encoder.{parts[0]}.{parts[1]}.{parts[2]}"
                    head_idx = int(parts[3].split('_')[1])

                    if layer_name not in layer_heads:
                        layer_heads[layer_name] = []
                    layer_heads[layer_name].append((head_idx, entropy_mean.item()))

            # For each layer, select top percent of heads with highest entropy
            for layer_name, heads_with_entropy in layer_heads.items():
                if len(heads_with_entropy) > 0:
                    # Sort heads by entropy mean in descending order
                    if self.prunehighentropy:
                        heads_with_entropy.sort(key=lambda x: x[1], reverse=True)
                    else:
                        heads_with_entropy.sort(key=lambda x: x[1])
                    # Calculate number of heads to select based on percentage
                    num_heads_to_select = max(1, int(len(heads_with_entropy) * self.percent))

                    # Select top percent of heads
                    selected_heads = heads_with_entropy[:num_heads_to_select]

                    # Store only the head indices
                    mask = torch.isin(torch.arange(400), torch.tensor(
                        [head_idx for head_idx, _ in selected_heads]
                    )).to(predictor.device)
                    self.final_entropy_stats[layer_name] = mask

                    

        # Log sample statistics
        for layer_name, head_list in list(self.final_entropy_stats.items()):
            print(f'Layer {layer_name}: {len(head_list)} heads with high entropy: {head_list[:10]}{"..." if len(head_list) > 10 else ""}')


    def get_entropy_stats(self):
        """Return the collected entropy statistics"""
        return getattr(self, 'final_entropy_stats', {})


    
        
        

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Standard attention processing with optional head pruning"""
        if not any(num in module_name for num in ["5", "11", "17", "23"]):  # modules whose shape is (16, 4096, 4096)
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)
        else:
            prune_mask  = None 

        B, H, W, _ = x.shape
        # Standard attention computation
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
        if prune_mask is not None:
            q_attn = q[~prune_mask, :, :]
            k_attn = k[~prune_mask, :, :]
            v_attn = v[~prune_mask, :, :]
            q_prune= quantize_activation_per_token_absmax(q[prune_mask, :, :],2)
            k_prune= quantize_activation_per_token_absmax(k[prune_mask, :, :],2)
            v_pruned = quantize_activation_per_channel_absmax(v[prune_mask, :, :],2)
        else:
            q_attn = q
            k_attn = k
            v_attn = v

        attn = (q_attn * module.scale) @ k_attn.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q_attn, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        B_nhead, N, _ = attn.shape  # B * num_heads, N, N

        x_attn = attn  @ v_attn
        if prune_mask is not None:
            x = torch.zeros_like(v).to(v.device)
            attn_prune =  ((q_prune * module.scale) @ k_prune.transpose(-2, -1)).softmax(dim=-1)
            x_prune = quantize_activation_per_token_absmax(attn_prune, 2) @ v_pruned
            x[prune_mask] = x_prune 
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        x = x.view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)
        return x


