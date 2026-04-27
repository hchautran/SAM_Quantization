from .entropy import BaseEntropyProcessor
from collections import defaultdict
import torch
import math
from block_sparse_attention import block_sparse_attn_simple
class PositionalSparseProcessorDiffDuo(BaseEntropyProcessor):
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

    def __init__(self, strategy_name: str = "PositionalSparseProcessorDiffDuo"):
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

    def create_mask(self, n, k=1, device="cpu"):
        idx = torch.arange(n, device=device)
        dist = (idx[:, None] - idx[None, :]).abs()
        return (dist <= k).int()

    def _create_attention_hook(self, name):
        """Create attention hook for positional pruning."""

        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            qkv = (
                module.qkv(x)
                .reshape(B, H * W, 3, module.num_heads, -1)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            if module.use_rel_pos:
                pos = self.get_decomposed_rel_pos(
                    q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W)
                )

            attn = (q * module.scale) @ k.transpose(-2, -1)
            attn = attn + pos
            attn = attn.softmax(dim=-1)
            B_nhead, N, _ = attn.shape

            for head_idx in range(B_nhead):
                attn_head = attn[head_idx]
                mean_entropy = self.calculate_entropy(attn_head)
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(mean_entropy)

            block_size = 128
            nrow = ncol = (H * W + block_size - 1) // block_size
            if nrow > 2:
                module.attn_mask = self.create_mask(
                    nrow, k=math.ceil(0.10 * nrow), device=q.device
                )[None, None, ...].expand((pos.shape[0], 1, -1, -1))
            else:
                module.attn_mask = torch.ones(
                    k.shape[0], 1, nrow, ncol, device=q.device, dtype=torch.bool
                )

        return attention_hook

    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode."""
        pass

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
                    key=lambda x: x[1], reverse=self.prunehighentropy
                )
                # import ipdb; ipdb.set_trace()
                # Calculate number of heads to select

                # num_heads_to_select = self._calculate_num_heads_to_select(len(heads_with_entropy))
                num_heads_to_select = self.prune_counts_per_layer.get(layer_name, None)

                selected_heads = heads_with_entropy[:num_heads_to_select]
                print(
                    "remain heads/ total heads:",
                    len(heads_with_entropy) - num_heads_to_select,
                    "/",
                    len(heads_with_entropy),
                )

                elements = torch.arange(len(heads_with_entropy))
                test_elements = torch.tensor(
                    [head_idx for head_idx, _ in selected_heads]
                )

                # Create mask using broadcasting
                mask = (
                    (elements.unsqueeze(1) == test_elements.unsqueeze(0))
                    .any(dim=1)
                    .to(predictor.device)
                )
                final_stats[layer_name] = mask

        return final_stats

    def get_prune_mask(self, module, module_name):
        if not self.prune_global:
            if self.model_type == "vit_b":
                if not any(num in module_name for num in ["2", "5", "8", "11"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
            elif self.model_type == "vit_l":
                if not any(num in module_name for num in [".5", "11", "17", "23"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
            elif self.model_type == "vit_h":
                if not any(num in module_name for num in [".7", "15", "23", "31"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
        else:
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)
        return prune_mask

    def calculate_pruned_heads_per_layer_percent_based(self, predictor):
        """
        Calculate the NUMBER of heads to prune per layer based on trained DiffPruneRate probability values.
        Separates Global and Local attention layers and determines cutoffs based on respective percentages.
        Returns a dictionary where keys are layer names and values are the count of heads to prune.
        """
        # 1. Define global layer indices based on model type
        global_indices = set()
        if self.model_type == "vit_b":
            global_indices = {2, 5, 8, 11}
        elif self.model_type == "vit_l":
            global_indices = {5, 11, 17, 23}
        elif self.model_type == "vit_h":
            global_indices = {7, 15, 23, 31}

        local_group_data = []  # Stores (value, head_index, layer_name)
        global_group_data = []

        # 2. Iterate through model blocks to collect probability values
        blocks = predictor.model.image_encoder.blocks

        for i, block in enumerate(blocks):
            layer_name = f"image_encoder.blocks.{i}.attn"

            # Ensure the attention module has the trained parameter wrapper
            if hasattr(block.attn, "prune_ddp"):
                # Get probability values (value per head)
                # prune_ddp.get_head_probability_diff_duo() returns the prob probabilities
                probs = (
                    block.attn.prune_ddp.get_head_probability_diff_duo().detach().cpu()
                )

                is_global = i in global_indices
                target_list = global_group_data if is_global else local_group_data

                # Collect tuples
                for head_idx, val in enumerate(probs):
                    target_list.append((val.item(), head_idx, layer_name))

        # 3. Sort groups in ascending order (smallest probabilities are pruned first)
        local_group_data.sort(key=lambda x: x[0])
        global_group_data.sort(key=lambda x: x[0])

        # 4. Determine cut-off counts based on percentages
        # e.g., if percent is 0.2, we prune the bottom 20% of heads across all local layers
        num_prune_local = int(len(local_group_data) * self.percent)
        num_prune_global = int(len(global_group_data) * self.global_percent)

        # 5. Select heads to prune
        heads_to_prune = (
            local_group_data[:num_prune_local] + global_group_data[:num_prune_global]
        )

        # 6. Count heads per layer to prune
        # We need to return {layer_name: number_of_heads_to_prune} based on the request
        prune_counts = defaultdict(int)

        # Initialize counts for all layers to 0 ensures layers with 0 pruned heads exist in dict
        for i in range(len(blocks)):
            layer_name = f"image_encoder.blocks.{i}.attn"
            prune_counts[layer_name] = 0

        for _, _, layer_name in heads_to_prune:
            prune_counts[layer_name] += 1

        print("\n=== Probability Pruning Statistics per Layer (Counts) ===")
        for layer_name, count in prune_counts.items():
            print(f"{layer_name}: Pruning {count} heads")
        print("=========================================================\n")

        # Store or return this based on how it's used.
        # The prompt asks to return a dictionary where value is number of heads to prune.
        # Often this is stored in a class attribute for later use, but here we return it or store it.
        # Since logic suggests this replaces the mask logic for counts, we might store it.
        # But specifically requested to return/implement logic.

        # Storing in a specific attribute for the 'diff' strategies might be needed,
        # but reusing final_entropy_stats for now or a new attribute is fine.
        # Given the previous context, 'final_entropy_stats' usually holds masks or sorted indices
        # but here the request is specifically for "number of heads".

        self.prune_counts_per_layer = dict(prune_counts)
        layer_heads = self._group_entropy_by_layer()
        mask_size_fn = lambda layer_name, heads: len(heads)
        self.final_entropy_stats = self._select_heads_by_percent(
            predictor, layer_heads, mask_size_fn
        )
        return self.final_entropy_stats

    def process(self, x: torch.Tensor, module, module_name: str = None):
        prune_mask = self.get_prune_mask(module, module_name)
        q, k, v, B, H, W = self._compute_qkv(x, module)
        dtype = q.dtype
        pos = self.get_decomposed_rel_pos(
            q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W)
        )
        prune_mask = prune_mask.repeat(q.shape[0] // prune_mask.shape[0])
        prune_mask = torch.where(prune_mask, -1, 1).to(torch.int32)

        if H * W <= 196:
            batch_size = B // 25
        else:
            batch_size = B
        # import ipdb; ipdb.set_trace()
        x = block_sparse_attn_simple(
            q.unsqueeze_(2).half(),
            k.unsqueeze_(2).half(),
            v.unsqueeze_(2).half(),
            module.attn_mask.repeat(batch_size, 1, 1, 1),
            head_mask_type=prune_mask,
            positional=pos.unsqueeze_(1).half(),
            softmax_scale=q.shape[-1] ** -0.5,
        ).to(dtype)

        x = self._reshape_output(x.squeeze(), B, module.num_heads, H, W)
        x = module.proj(x)
        return x


class PositionalSparseProcessorDuo(BaseEntropyProcessor):
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

    def __init__(self, strategy_name: str = "PositionalSparseProcessorDuo"):
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

    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode."""
        pass

    def create_mask(self, n, k=1, device="cpu"):
        idx = torch.arange(n, device=device)
        dist = (idx[:, None] - idx[None, :]).abs()
        return (dist <= k).int()

    def _create_attention_hook(self, name):
        """Create attention hook for positional pruning."""

        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            qkv = (
                module.qkv(x)
                .reshape(B, H * W, 3, module.num_heads, -1)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            if module.use_rel_pos:
                pos = self.get_decomposed_rel_pos(
                    q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W)
                )

            attn = (q * module.scale) @ k.transpose(-2, -1)
            attn = attn + pos
            attn = attn.softmax(dim=-1)
            B_nhead, N, _ = attn.shape

            for head_idx in range(B_nhead):
                attn_head = attn[head_idx]
                mean_entropy = self.calculate_entropy(attn_head)
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(mean_entropy)

            block_size = 128
            nrow = ncol = (H * W + block_size - 1) // block_size
            if nrow > 2:
                module.attn_mask = self.create_mask(
                    nrow, k=math.ceil(0.10 * nrow), device=q.device
                )[None, None, ...].expand((pos.shape[0], 1, -1, -1))
            else:
                module.attn_mask = torch.ones(
                    k.shape[0], 1, nrow, ncol, device=q.device, dtype=torch.bool
                )

        return attention_hook

    def get_prune_mask(self, module, module_name):
        if not self.prune_global:
            if self.model_type == "vit_b":
                if not any(num in module_name for num in ["2", "5", "8", "11"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
            elif self.model_type == "vit_l":
                if not any(num in module_name for num in [".5", "11", "17", "23"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
            elif self.model_type == "vit_h":
                if not any(num in module_name for num in [".7", "15", "23", "31"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
        else:
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)
        return prune_mask

    def calculate_pruned_heads_per_layer_percent_based(self, predictor):
        """
        Calculate heads to prune based on trained Full Attention Heads gate values.
        Separates Global and Local attention layers and prunes them based on respective percentages.
        Returns a boolean mask: True indicates the head should be pruned.
        """
        # 1. Define global layer indices based on model type
        global_indices = set()
        if self.model_type == "vit_b":
            global_indices = {2, 5, 8, 11}
        elif self.model_type == "vit_l":
            global_indices = {5, 11, 17, 23}
        elif self.model_type == "vit_h":
            global_indices = {7, 15, 23, 31}

        local_group_data = []  # Stores (value, head_index, layer_name)
        global_group_data = []
        layer_head_counts = {}  # Stores total heads per layer to construct masks later

        # 2. Iterate through model blocks to collect gate values
        # Accessing blocks via predictor.model.image_encoder.blocks
        blocks = predictor.model.image_encoder.blocks

        for i, block in enumerate(blocks):
            layer_name = f"image_encoder.blocks.{i}.attn"

            # Ensure the attention module has the trained parameter
            if hasattr(block.attn, "full_attention_heads"):
                # Get gate values (value per head)
                gate_values = block.attn.full_attention_heads.detach()

                # Store head count for this layer
                layer_head_counts[layer_name] = gate_values.shape[0]

                gate_values = gate_values.cpu()

                is_global = i in global_indices
                target_list = global_group_data if is_global else local_group_data

                # Collect tuples
                for head_idx, val in enumerate(gate_values):
                    target_list.append((val.item(), head_idx, layer_name))

        # 3. Sort groups in ascending order (smallest gate values are pruned first)
        local_group_data.sort(key=lambda x: x[0])
        global_group_data.sort(key=lambda x: x[0])

        # 4. Determine cut-off counts based on percentages
        num_prune_local = int(len(local_group_data) * self.percent)
        num_prune_global = int(len(global_group_data) * self.global_percent)

        # 5. Select heads to prune
        heads_to_prune = (
            local_group_data[:num_prune_local] + global_group_data[:num_prune_global]
        )

        # 6. Build result dictionary {layer_name: boolean_mask}
        # Organize pruned indices by layer for efficient mask creation
        pruned_indices_map = defaultdict(list)
        for _, head_idx, layer_name in heads_to_prune:
            pruned_indices_map[layer_name].append(head_idx)

        final_masks = {}
        for layer_name, total_heads in layer_head_counts.items():
            # Initialize mask as False (Keep)
            mask = torch.zeros(total_heads, dtype=torch.bool, device=predictor.device)

            if layer_name in pruned_indices_map:
                indices_to_prune = torch.tensor(
                    pruned_indices_map[layer_name], device=predictor.device
                )
                # Set pruned indices to True
                mask[indices_to_prune] = True

            final_masks[layer_name] = mask
        print("\n=== Pruning Statistics per Layer ===")
        for layer_name, mask in final_masks.items():
            num_pruned = mask.sum().item()
            total = mask.shape[0]
            print(
                f"{layer_name}: Pruned {num_pruned}/{total} heads ({(num_pruned / total) * 100:.1f}%)"
            )
        print("====================================\n")

        self.final_entropy_stats = final_masks

    def process(self, x: torch.Tensor, module, module_name: str = None):
        prune_mask = self.get_prune_mask(module, module_name)
        q, k, v, B, H, W = self._compute_qkv(x, module)
        dtype = q.dtype
        pos = self.get_decomposed_rel_pos(
            q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W)
        )

        prune_mask = prune_mask.repeat(q.shape[0] // prune_mask.shape[0])
        prune_mask = torch.where(prune_mask, -1, 1).to(torch.int32)

        if H * W <= 196:
            batch_size = B // 25
        else:
            batch_size = B

        x = block_sparse_attn_simple(
            q.unsqueeze_(2).half(),
            k.unsqueeze_(2).half(),
            v.unsqueeze_(2).half(),
            module.attn_mask.repeat(batch_size, 1, 1, 1),
            head_mask_type=prune_mask,
            positional=pos.unsqueeze_(1).half(),
            softmax_scale=q.shape[-1] ** -0.5,
        ).to(dtype)

        x = self._reshape_output(x.squeeze(), B, module.num_heads, H, W)
        x = module.proj(x)
        return x


class HeadPruneProcessor(BaseEntropyProcessor):
    """Processor that identifies and prunes attention heads based on entropy."""

    def __init__(self, strategy_name: str = "head_prune"):
        super().__init__(strategy_name)

    def set_params(self, args):
        super().set_params(args)

    def calculate_entropy(self, attn_head):
        """Calculate entropy for each position in a single attention head."""
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        eps = 1e-12
        attn_normalized = torch.clamp(attn_head, min=eps)
        entropy_per_position = -torch.mean(
            attn_normalized * torch.log(attn_normalized), dim=-1
        )
        return entropy_per_position

    def _create_attention_hook(self, name):
        """Create attention hook for head pruning."""

        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            qkv = (
                module.qkv(x)
                .reshape(B, H * W, 3, module.num_heads, -1)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)
            if module.use_rel_pos:
                pos = self.get_decomposed_rel_pos(
                    q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W)
                )
            attn = attn + pos
            attn = attn.softmax(dim=-1).reshape(module.num_heads, B, attn.shape[-2], -1)
            n_heads, B, N, _ = attn.shape

            for head_idx in range(n_heads):
                attn_head = attn[head_idx]
                entropy_per_position = self.calculate_entropy(attn_head)
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key].append(
                    entropy_per_position.mean(-1).mean(-1).item()
                )

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
            mask = torch.isin(torch.arange(16), torch.tensor(head_indices)).to(
                predictor.device
            )
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
        nu_heads = predictor.model.image_encoder.blocks[0].attn.num_heads

        # Select heads by percentage
        self.final_entropy_stats = {}
        for layer_name, heads_with_entropy in layer_heads.items():
            nu_subimages = 25
            percent = self.percent
            if self.model_type == "vit_b":
                if any(num in layer_name for num in ["2", "5", "8", "11"]):
                    nu_subimages = 1
                    percent = self.global_percent
            elif self.model_type == "vit_l":
                if any(num in layer_name for num in [".5", "11", "17", "23"]):
                    nu_subimages = 1
                    percent = self.global_percent
            elif self.model_type == "vit_h":
                if any(num in layer_name for num in [".7", "15", "23", "31"]):
                    nu_subimages = 1
                    percent = self.global_percent
            if len(heads_with_entropy) > 0:
                heads_with_entropy.sort(
                    key=lambda x: x[1], reverse=self.prunehighentropy
                )

                num_heads_to_select = int(len(heads_with_entropy) * percent)
                selected_heads = heads_with_entropy[:num_heads_to_select]

                # Create mask for batch * heads indexing
                selected_head_indices = torch.tensor(
                    [head_idx for head_idx, _ in selected_heads]
                )
                mask_indices = (
                    torch.arange(nu_subimages)[:, None] * nu_heads
                    + selected_head_indices
                ).flatten()
                mask = torch.isin(
                    torch.arange(nu_heads * nu_subimages), mask_indices
                ).to(predictor.device)
                self.final_entropy_stats[layer_name] = mask

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Standard attention processing with optional head pruning."""
        if not self.prune_global:
            if self.model_type == "vit_b":
                if not any(num in module_name for num in ["2", "5", "8", "11"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
            elif self.model_type == "vit_l":
                if not any(num in module_name for num in ["5", "11", "17", "23"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
            elif self.model_type == "vit_h":
                if not any(num in module_name for num in ["7", "15", "23", "31"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
        else:
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)
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
            x = torch.zeros(B * module.num_heads, attn.shape[-2], v.shape[-1]).to(
                v.device
            )
            x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(
                -1, x_attn.shape[-2], x_attn.shape[-1]
            )
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        x = self._reshape_output(x, B, module.num_heads, H, W)
        x = module.proj(x)
        return x


class WholeSubImageProcessor(BaseEntropyProcessor):
    def __init__(self, strategy_name: str = "head_prune"):
        super().__init__(strategy_name)

    def set_params(self, args):
        super().set_params(args)

    def calculate_entropy(self, attn_head):
        """Calculate entropy for each position in a single attention head."""
        if isinstance(attn_head, torch.Tensor) is False:
            attn_head = torch.from_numpy(attn_head)

        eps = 1e-12
        attn_normalized = torch.clamp(attn_head, min=eps)
        entropy_per_position = -torch.mean(
            attn_normalized * torch.log(attn_normalized), dim=-1
        )
        return entropy_per_position

    def _create_attention_hook(self, name):
        """Create attention hook for head pruning."""

        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            qkv = (
                module.qkv(x)
                .reshape(B, H * W, 3, module.num_heads, -1)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)
            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(
                    attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W)
                )

            attn = attn.softmax(dim=-1).reshape(B, module.num_heads, attn.shape[-2], -1)
            B, n_heads, N, _ = attn.shape

            for subimage_idx in range(B):
                attn_subimage = attn[subimage_idx]
                entropy_per_position = self.calculate_entropy(attn_subimage)
                head_key = f"{name}.subimage_{subimage_idx}"
                self.entropy_stats[head_key].append(
                    entropy_per_position.mean(-1).mean(-1).item()
                )

        return attention_hook

    def _process_percent_mode(self, predictor):
        """Process calibration in percent mode."""
        layer_heads = {}
        for head_key, stats in self.entropy_stats.items():
            if len(stats) > 0:
                entropy_mean = self._compute_entropy_stats(stats)
                layer_name, subimage_idx = self._parse_head_key(head_key)

                if layer_name not in layer_heads:
                    layer_heads[layer_name] = []
                layer_heads[layer_name].append((subimage_idx, entropy_mean.item()))

        nu_heads = predictor.model.image_encoder.blocks[0].attn.num_heads

        # Select heads by percentage
        self.final_entropy_stats = {}
        for layer_name, subimages_with_entropy in layer_heads.items():
            nu_subimages = 25
            percent = self.percent

            # Identify Global vs Local layers
            is_global = False
            if self.model_type == "vit_b":
                if any(num in layer_name for num in ["2", "5", "8", "11"]):
                    is_global = True
            elif self.model_type == "vit_l":
                if any(num in layer_name for num in [".5", "11", "17", "23"]):
                    is_global = True
            elif self.model_type == "vit_h":
                # Note: fixed module_name -> layer_name typo from context
                if any(num in layer_name for num in [".7", "15", "23", "31"]):
                    is_global = True

            if is_global:
                nu_subimages = 1
                percent = self.global_percent

            if len(subimages_with_entropy) > 0:
                subimages_with_entropy.sort(
                    key=lambda x: x[1], reverse=self.prunehighentropy
                )

                num_heads_to_select = int(len(subimages_with_entropy) * percent)
                selected_subimage = subimages_with_entropy[:num_heads_to_select]

                # Create mask for batch * heads indexing
                selected_subimage_indices = torch.tensor(
                    [subimage_idx for subimage_idx, _ in selected_subimage],
                    device=predictor.device,
                )

                if nu_subimages > 1:
                    # Expand indices: {2, 5} -> {2*H...2*H+H-1, 5*H...5*H+H-1}
                    # This maps selected sub-images to all their constituent heads
                    mask_indices = (
                        selected_subimage_indices.unsqueeze(1) * nu_heads
                        + torch.arange(nu_heads, device=predictor.device).unsqueeze(0)
                    ).flatten()
                else:
                    # Fallback for global or non-expanded case
                    mask_indices = selected_subimage_indices

                mask = torch.isin(
                    torch.arange(nu_heads * nu_subimages, device=predictor.device),
                    mask_indices,
                )
                self.final_entropy_stats[layer_name] = mask

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Standard attention processing with optional head pruning."""
        if not self.prune_global:
            if self.model_type == "vit_b":
                if not any(num in module_name for num in ["2", "5", "8", "11"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
            elif self.model_type == "vit_l":
                if not any(num in module_name for num in ["5", "11", "17", "23"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
            elif self.model_type == "vit_h":
                if not any(num in module_name for num in ["7", "15", "23", "31"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
        else:
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)
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
            x = torch.zeros(B * module.num_heads, attn.shape[-2], v.shape[-1]).to(
                v.device
            )
            x[prune_mask] = v_pruned
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

    def __init__(self, strategy_name: str = "PositionalHeadPruneProcessor"):
        super().__init__(strategy_name)

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
        if self.model_type == "vit_b":
            if total_heads < 300:
                return int(total_heads * self.global_percent)
        elif self.model_type == "vit_l":
            if total_heads < 400:
                return int(total_heads * self.global_percent)
        elif self.model_type == "vit_h":
            if total_heads < 400:
                return int(total_heads * self.global_percent)

        return int(total_heads * self.percent)

    def _create_attention_hook(self, name):
        """Create attention hook for positional quantization."""

        def attention_hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape

            qkv = (
                module.qkv(x)
                .reshape(B, H * W, 3, module.num_heads, -1)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

            attn = (q * module.scale) @ k.transpose(-2, -1)
            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(
                    attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W)
                )

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
                    key=lambda x: x[1], reverse=self.prunehighentropy
                )

                num_heads_to_select = self._calculate_num_heads_to_select(
                    len(heads_with_entropy)
                )
                selected_heads = heads_with_entropy[:num_heads_to_select]
                print(
                    "remain heads/ total heads:",
                    len(heads_with_entropy) - num_heads_to_select,
                    "/",
                    len(heads_with_entropy),
                )
                # mask = torch.isin(
                #     torch.arange(len(heads_with_entropy)),
                #     torch.tensor([head_idx for head_idx, _ in selected_heads])
                # ).to(predictor.device)
                # self.final_entropy_stats[layer_name] = mask
                elements = torch.arange(len(heads_with_entropy))
                test_elements = torch.tensor(
                    [head_idx for head_idx, _ in selected_heads]
                )

                # Create mask using broadcasting
                mask = (
                    (elements.unsqueeze(1) == test_elements.unsqueeze(0))
                    .any(dim=1)
                    .to(predictor.device)
                )
                self.final_entropy_stats[layer_name] = mask

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Standard attention processing with optional head quantization."""
        # Determine if we should quantize this layer
        if not self.prune_global:
            if self.model_type == "vit_b":
                if not any(num in module_name for num in ["2", "5", "8", "11"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
            elif self.model_type == "vit_l":
                if not any(num in module_name for num in ["5", "11", "17", "23"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
            elif self.model_type == "vit_h":
                if not any(num in module_name for num in ["7", "15", "23", "31"]):
                    prune_mask = module.processor.final_entropy_stats.get(
                        module_name, None
                    )
                else:
                    prune_mask = None
        else:
            prune_mask = module.processor.final_entropy_stats.get(module_name, None)

        B, H, W, _ = x.shape

        if self.model_type == "vit_b":
            n_bits = 4 if not prune_mask.shape[0] == 300 else 2
        elif self.model_type == "vit_l":
            n_bits = 4 if not prune_mask.shape[0] == 400 else 2
        elif self.model_type == "vit_h":
            n_bits = 4 if not prune_mask.shape[0] == 400 else 2

        q, k, v, B, H, W = self._compute_qkv(x, module)

        if prune_mask is not None:
            prune_mask = prune_mask.repeat(q.shape[0] // prune_mask.shape[0])
            q_attn = q[~prune_mask, :, :]
            k_attn = k[~prune_mask, :, :]
            v_attn = v[~prune_mask, :, :]
            q_prune = quantize_activation_per_token_absmax(q[prune_mask, :, :], n_bits)
            k_prune = quantize_activation_per_token_absmax(k[prune_mask, :, :], n_bits)
            v_prune = quantize_activation_per_channel_absmax(
                v[prune_mask, :, :], n_bits
            )
        else:
            q_attn, k_attn, v_attn = q, k, v

        attn = self._compute_attention(q_attn, k_attn, module, H, W)
        x_attn = attn @ v_attn

        if prune_mask is not None:
            x = torch.zeros_like(v).to(v.device)
            attn_prune = ((q_prune * module.scale) @ k_prune.transpose(-2, -1)).softmax(
                dim=-1
            )
            x_prune = quantize_activation_per_token_absmax(attn_prune, n_bits) @ v_prune
            x[prune_mask] = x_prune
            x[~prune_mask] = x_attn
        else:
            x = x_attn

        x = self._reshape_output(x, B, module.num_heads, H, W)
        x = module.proj(x)
        return x
