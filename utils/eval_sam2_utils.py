
from prunning_rate.sam2pruneduo import monkey_patch_train_sam2_duo, DuoPruneRateMultiScaleAttention
from prunning_rate.sam2prune import DiffPruneRateMultiScaleAttention

def analyze_model_head_pruning_and_flops(predictor_model, manual_heads_overrides=None):
    """
    Analyze head pruning ratios and calculate FLOPs, grouping layers by their original head count.
    
    Args:
        predictor_model: The SAM model with DiffPruneRateMultiScaleAttention modules
        manual_heads_overrides: Optional dict mapping original_heads (int) -> manual_keep (int). 
                              If a group is not in this dict, the average kept heads (rounded up) 
                              will be used as the manual target.
        
    Returns:
        dict: Dictionary containing head statistics and FLOPs information per group
    """
    if manual_heads_overrides is None:
        manual_heads_overrides = {}

    # Initialize counters and stats containers
    # Key: original_heads (int), Value: dict of accumulators
    group_stats = {}
    
    total_flops = 0
    total_baseline_flops = 0
    
    layer_info = []
    
    def get_layer_specs(n_heads):
        """Determine H, W, and batch_factor based on original head count."""
        if n_heads == 8:
            return 64, 64, 1
        elif n_heads == 200:
            return 14, 14, 25
        elif n_heads == 400:
            return 7, 7, 25
        elif n_heads == 2048:
            return 8, 8, 1024
        elif n_heads == 4096:
            return 4, 4, 1024
        
        # Fallback logic for unknown head counts
        if n_heads % 100 == 0:
            return 14, 14, 25
        return 14, 14, 1024

    # First Pass: Collect current stats and baseline FLOPs
    for name, module in predictor_model.named_modules():
        if isinstance(module, DiffPruneRateMultiScaleAttention):
            # Get head information
            kept_heads = int(module.prune_ddp.update_kept_head_number())
            original_heads = module.prune_ddp.head_number
            
            # Initialize group if encountered for the first time
            if original_heads not in group_stats:
                group_stats[original_heads] = {
                    'layer_count': 0,
                    'total_kept': 0,
                    'total_original': 0,
                    'current_flops': 0,
                    'baseline_flops': 0,
                    'manual_flops': 0
                }
            
            # Determine dimensions
            H, W, batch_factor = get_layer_specs(original_heads)
            
            # Calculate FLOPs for this attention module
            qkv_flops = module._calculate_qkv_flops(batch_factor, H, W)
            proj_flops = module._calculate_projection_flops(batch_factor, H, W)
            attention_flops = module._calculate_attention_flops(H, W, kept_heads)
            baseline_attention_flops = module._calculate_attention_flops(H, W, original_heads)
            
            module_flops = qkv_flops + proj_flops + attention_flops
            module_baseline_flops = qkv_flops + proj_flops + baseline_attention_flops
            
            # Update totals
            total_flops += module_flops
            total_baseline_flops += module_baseline_flops
            
            # Update group stats
            stats = group_stats[original_heads]
            stats['layer_count'] += 1
            stats['total_kept'] += kept_heads
            stats['total_original'] += original_heads
            stats['current_flops'] += module_flops
            stats['baseline_flops'] += module_baseline_flops
            
            # Store layer information
            layer_info.append({
                'name': name,
                'original_heads': original_heads,
                'kept_heads': kept_heads,
                'flops': module_flops,
                'baseline_flops': module_baseline_flops,
                'H': H, 'W': W, 'batch_factor': batch_factor
            })
    
    # Determine manual targets for each group
    group_manual_targets = {}
    for heads, stats in group_stats.items():
        if heads in manual_heads_overrides:
            target = manual_heads_overrides[heads]
        else:
            # Default: Average kept heads, rounded up
            avg = stats['total_kept'] / stats['layer_count'] if stats['layer_count'] > 0 else 0
            target = int(avg + 0.999)
        group_manual_targets[heads] = target

    # Second Pass: Calculate Manual FLOPs based on targets
    total_manual_flops = 0
    
    for name, module in predictor_model.named_modules():
        if isinstance(module, DiffPruneRateMultiScaleAttention):
            original_heads = module.prune_ddp.head_number
            target_heads = group_manual_targets.get(original_heads, original_heads)
            
            H, W, batch_factor = get_layer_specs(original_heads)
            
            qkv_flops = module._calculate_qkv_flops(batch_factor, H, W)
            proj_flops = module._calculate_projection_flops(batch_factor, H, W)
            manual_attention_flops = module._calculate_attention_flops(H, W, target_heads)
            
            manual_module_flops = qkv_flops + proj_flops + manual_attention_flops
            
            total_manual_flops += manual_module_flops
            group_stats[original_heads]['manual_flops'] += manual_module_flops
    
    # Finalize statistics
    formatted_group_stats = {}
    for heads, stats in group_stats.items():
        ratio = stats['total_kept'] / stats['total_original'] if stats['total_original'] > 0 else 0
        formatted_group_stats[heads] = {
            'layer_count': stats['layer_count'],
            'kept_heads': stats['total_kept'],
            'original_heads': stats['total_original'],
            'ratio': ratio,
            'flops': stats['current_flops'],
            'baseline_flops': stats['baseline_flops'],
            'manual_flops': stats['manual_flops'],
            'manual_target_heads': group_manual_targets[heads]
        }

    flops_reduction = (1 - total_flops / total_baseline_flops) * 100 if total_baseline_flops > 0 else 0
    manual_flops_reduction = (1 - total_manual_flops / total_baseline_flops) * 100 if total_baseline_flops > 0 else 0
    
    return {
        'layer_info': layer_info,
        'group_stats': formatted_group_stats,
        'flops_stats': {
            'total_flops': total_flops,
            'baseline_flops': total_baseline_flops,
            'reduction_percent': flops_reduction,
            'manual_flops': total_manual_flops,
            'manual_reduction_percent': manual_flops_reduction
        }
    }
def print_head_pruning_and_flops_info(predictor_model, logger):
    """
    Print and log detailed head pruning ratios and FLOPs information.
    Assumes `logger` is created by setup_logger().
    """

    def pl(msg):
        print(msg)
        logger.info(msg)

    pl("=" * 100)
    pl("Head Pruning Ratios")
    pl("=" * 60)

    analysis = analyze_model_head_pruning_and_flops(predictor_model)

    # Per-layer information
    pl("--- Per Layer Details - prune head number ---")
    for layer in analysis["layer_info"]:
        pruned_head = layer['original_heads'] - layer['kept_heads']
        pl(
            f"Layer {layer['name']}: "
            f"{pruned_head}/{layer['original_heads']} heads"
        )

    # Group statistics
    pl("\n--- Group Statistics ---")
    group_stats = analysis["group_stats"]

    for heads in sorted(group_stats.keys()):
        stats = group_stats[heads]
        if stats["original_heads"] == 0:
            continue

        pl(f"\nGroup (Original Heads: {heads})")
        pl(f"  Layers: {stats['layer_count']}")
        pl(
            f"  Kept: {stats['kept_heads']}/{stats['original_heads']} "
            f"({stats['ratio']:.2%}) | "
            f"Pruning rate: {1 - stats['ratio']:.2%}"
        )
        pl(f"  Manual Target per Layer: {stats['manual_target_heads']}")

    # FLOPs information
    flops = analysis["flops_stats"]

    pl("\n" + "=" * 60)
    pl("FLOPs Information (Attention Only)")
    pl("=" * 60)

    pl(
        f"Total Attention FLOPs (parameter pruning): "
        f"{flops['total_flops'] / 1e9:.2f} GFLOPs"
    )
    pl(
        f"Total Attention FLOPs (baseline): "
        f"{flops['baseline_flops'] / 1e9:.2f} GFLOPs"
    )

    if flops["reduction_percent"] >= 0:
        pl(
            f"Attention FLOPs Reduction (parameter pruning): "
            f"{flops['reduction_percent']:.2f}%"
        )

    # Manual pruning
    pl(
        f"\nTotal Attention FLOPs (manual pruning): "
        f"{flops['manual_flops'] / 1e9:.2f} GFLOPs"
    )

    if flops["manual_reduction_percent"] >= 0:
        pl(
            f"Attention FLOPs Reduction (manual pruning): "
            f"{flops['manual_reduction_percent']:.2f}%"
        )

    # Comparison
    diff = flops["manual_flops"] - flops["total_flops"]
    if diff > 0:
        pl(
            f"Manual pruning uses {diff / 1e9:.2f} GFLOPs "
            f"MORE than parameter pruning"
        )
    else:
        pl(
            f"Manual pruning uses {abs(diff) / 1e9:.2f} GFLOPs "
            f"LESS than parameter pruning"
        )

    pl("=" * 60 + "\n")



def print_duo_head_pruning_info(predictor_model):
    """
    Print detailed head pruning information for DuoPruneRateMultiScaleAttention modules.
    Shows which heads will be pruned based on alpha values and thresholds.

    Args:
        predictor_model: The SAM model with DuoPruneRateMultiScaleAttention modules
    """
    print("\n=== Duo Training Head Pruning Analysis ===")

    layer_info = []
    group_stats = {}

    # Iterate through all modules to find DuoPruneRateMultiScaleAttention
    for name, module in predictor_model.named_modules():
        if isinstance(module, DuoPruneRateMultiScaleAttention):
            # Check if module has full_attention_heads (alpha values)
            if not hasattr(module, 'full_attention_heads') or module.full_attention_heads is None:
                print(f"Warning: Layer {name} has no full_attention_heads parameter")
                continue

            # Get alpha values and clamp them to [0, 1]
            alpha_values = module.full_attention_heads.clamp(0, 1)
            total_heads = len(alpha_values)

            # Determine threshold based on layer type (same logic as in forward method)
            if hasattr(module, 'model_type') and module.model_type == "hiera_b_plus":
                # Special layers use global threshold
                if any(num in name for num in [".12.", ".20.", ".16."]):
                    threshold = getattr(module, 'global_threshold', 0.5)
                    threshold_type = "global"
                else:
                    threshold = getattr(module, 'threshold', 0.5)
                    threshold_type = "local"
            else:
                threshold = getattr(module, 'threshold', 0.5)
                threshold_type = "local"

            # Create pruning mask (True = prune, False = keep)
            prune_mask = alpha_values < threshold
            heads_to_prune = prune_mask.sum().item()
            heads_to_keep = total_heads - heads_to_prune


            # Store layer information
            layer_info.append({
                'name': name,
                'total_heads': total_heads,
                'heads_to_prune': heads_to_prune,
                'heads_to_keep': heads_to_keep,
                'threshold': threshold,
                'threshold_type': threshold_type,
                'alpha_values': alpha_values.detach().cpu().numpy(),
            })

            # Update group statistics
            if total_heads not in group_stats:
                group_stats[total_heads] = {
                    'layer_count': 0,
                    'total_heads_to_prune': 0,
                    'total_heads_to_keep': 0,
                    'total_heads': 0
                }

            stats = group_stats[total_heads]
            stats['layer_count'] += 1
            stats['total_heads_to_prune'] += heads_to_prune
            stats['total_heads_to_keep'] += heads_to_keep
            stats['total_heads'] += total_heads

    if not layer_info:
        print("No DuoPruneRateMultiScaleAttention modules found with full_attention_heads parameter")
        return

    # Print per-layer information
    print("--- Per Layer Details ---")
    for layer in layer_info:
        print(f"\nLayer {layer['name']}:")
        print(f"  Total heads: {layer['total_heads']}")
        print(f"  Heads to prune: {layer['heads_to_prune']} ({layer['heads_to_prune']/layer['total_heads']:.2%})")
        print(f"  Heads to keep: {layer['heads_to_keep']} ({layer['heads_to_keep']/layer['total_heads']:.2%})")
        print(f"  Threshold: {layer['threshold']:.4f} ({layer['threshold_type']})")

        # Show alpha values for first few heads as example
        alpha_sample = layer['alpha_values'][:min(10, len(layer['alpha_values']))]
        print(f"  Alpha values (first {len(alpha_sample)}): {alpha_sample}")


    # Print group statistics divided by head count (8, 200, 400, 2048, 4096)
    print("\n--- Group Statistics by Head Count ---")

    # Define the standard head groups
    standard_heads = [8, 200, 400, 2048, 4096]

    for head_count in standard_heads:
        if head_count in group_stats:
            stats = group_stats[head_count]
            avg_pruned_per_layer = stats['total_heads_to_prune'] / stats['layer_count'] if stats['layer_count'] > 0 else 0
            avg_kept_per_layer = stats['total_heads_to_keep'] / stats['layer_count'] if stats['layer_count'] > 0 else 0
            prune_rate = stats['total_heads_to_prune'] / stats['total_heads'] if stats['total_heads'] > 0 else 0

            print(f"\n=== Group: {head_count} heads ===")
            print(f"  Number of layers: {stats['layer_count']}")
            print(f"  Total heads: {stats['total_heads']}")
            print(f"  Total heads to prune: {stats['total_heads_to_prune']} ({prune_rate:.2%})")
            print(f"  Total heads to keep: {stats['total_heads_to_keep']} ({1-prune_rate:.2%})")
            print(f"  Average per layer - Prune: {avg_pruned_per_layer:.1f}, Keep: {avg_kept_per_layer:.1f}")

    # Check for any non-standard head counts
    non_standard = [h for h in group_stats.keys() if h not in standard_heads]
    if non_standard:
        print("\n=== Non-standard Head Counts ===")
        for head_count in sorted(non_standard):
            stats = group_stats[head_count]
            avg_pruned_per_layer = stats['total_heads_to_prune'] / stats['layer_count'] if stats['layer_count'] > 0 else 0
            avg_kept_per_layer = stats['total_heads_to_keep'] / stats['layer_count'] if stats['layer_count'] > 0 else 0
            prune_rate = stats['total_heads_to_prune'] / stats['total_heads'] if stats['total_heads'] > 0 else 0

            print(f"\nGroup: {head_count} heads")
            print(f"  Number of layers: {stats['layer_count']}")
            print(f"  Total heads to prune: {stats['total_heads_to_prune']}/{stats['total_heads']} ({prune_rate:.2%})")
            print(f"  Average per layer - Prune: {avg_pruned_per_layer:.1f}, Keep: {avg_kept_per_layer:.1f}")

    # Print overall statistics
    total_pruned_across_all = sum(stats['total_heads_to_prune'] for stats in group_stats.values())
    total_heads_across_all = sum(stats['total_heads'] for stats in group_stats.values())
    overall_prune_rate = total_pruned_across_all / total_heads_across_all if total_heads_across_all > 0 else 0

    print(f"\n--- Overall Statistics ---")
    print(f"Total heads across all layers: {total_heads_across_all}")
    print(f"Total heads to prune: {total_pruned_across_all} ({overall_prune_rate:.2%})")
    print(f"Total heads to keep: {total_heads_across_all - total_pruned_across_all} ({1-overall_prune_rate:.2%})")

    print("===========================\n")

def print_diff_duo_head_prunning_info(model, logger):
    """
    Print detailed head pruning information for DuoDiffPruneRateMultiScaleAttention modules.
    Shows which heads will be pruned based on probability values and thresholds.

    Args:
        model: The SAM model with DuoDiffPruneRateMultiScaleAttention modules
        logger: Logger instance for logging information
    """
    def pl(msg):
        print(msg)
        logger.info(msg)

    pl("\n=== Diff Duo Training Head Pruning Analysis ===")

    layer_info = []
    group_stats = {}

    # Iterate through all modules to find DuoDiffPruneRateMultiScaleAttention
    for name, module in model.named_modules():
        if hasattr(module, 'prune_ddp') and hasattr(module.prune_ddp, 'get_head_probability_diff_duo'):
            # Get probability values from DiffPruneRate
            try:
                probability_values = module.prune_ddp.get_head_probability_diff_duo()
                total_heads = len(probability_values)
            except Exception as e:
                pl(f"Warning: Layer {name} could not get probability values: {e}")
                continue

            # Determine threshold based on layer type and global/local logic
            is_global_layer = any(num in name for num in [".12.", ".20.", ".16."])
            
            if hasattr(module, 'model_type') and module.model_type == "hiera_b_plus":
                if is_global_layer:
                    threshold = getattr(module, 'global_threshold', 0.5)
                    threshold_type = "global"
                else:
                    threshold = getattr(module, 'threshold', 0.5)
                    threshold_type = "local"
            else:
                threshold = getattr(module, 'threshold', 0.5)
                threshold_type = "local"

            # Create pruning mask based on threshold
            # In diff duo logic, heads with probability > threshold are kept
            keep_mask = probability_values > threshold
            heads_to_keep = keep_mask.sum().item()
            heads_to_prune = total_heads - heads_to_keep

            # Store layer information
            layer_info.append({
                'name': name,
                'total_heads': total_heads,
                'heads_to_prune': heads_to_prune,
                'heads_to_keep': heads_to_keep,
                'threshold': threshold,
                'threshold_type': threshold_type,
                'is_global_layer': is_global_layer,
                'probability_values': probability_values.detach().cpu().numpy(),
            })

            # Update group statistics
            if total_heads not in group_stats:
                group_stats[total_heads] = {
                    'layer_count': 0,
                    'total_heads_to_prune': 0,
                    'total_heads_to_keep': 0,
                    'total_heads': 0,
                    'global_layers': 0,
                    'local_layers': 0
                }

            stats = group_stats[total_heads]
            stats['layer_count'] += 1
            stats['total_heads_to_prune'] += heads_to_prune
            stats['total_heads_to_keep'] += heads_to_keep
            stats['total_heads'] += total_heads
            
            if is_global_layer:
                stats['global_layers'] += 1
            else:
                stats['local_layers'] += 1

    if not layer_info:
        pl("No DuoDiffPruneRateMultiScaleAttention modules found with probability values")
        return

    # Print per-layer information
    pl("--- Per Layer Details ---")
    for layer in layer_info:
        pl(f"\nLayer {layer['name']}:")
        pl(f"  Total heads: {layer['total_heads']}")
        pl(f"  Heads to prune: {layer['heads_to_prune']} ({layer['heads_to_prune']/layer['total_heads']:.2%})")
        pl(f"  Heads to keep: {layer['heads_to_keep']} ({layer['heads_to_keep']/layer['total_heads']:.2%})")
        pl(f"  Threshold: {layer['threshold']:.4f} ({layer['threshold_type']})")
    

        # Show probability values for first few heads as example
        prob_sample = layer['probability_values'][:min(10, len(layer['probability_values']))]
        pl(f"  Probability values (first {len(prob_sample)}): {prob_sample}")

    # Print group statistics divided by head count (8, 200, 400, 2048, 4096)
    pl("\n--- Group Statistics by Head Count ---")

    # Define the standard head groups
    standard_heads = [8, 200, 400, 2048, 4096]

    for head_count in standard_heads:
        if head_count in group_stats:
            stats = group_stats[head_count]
            avg_pruned_per_layer = stats['total_heads_to_prune'] / stats['layer_count'] if stats['layer_count'] > 0 else 0
            avg_kept_per_layer = stats['total_heads_to_keep'] / stats['layer_count'] if stats['layer_count'] > 0 else 0
            prune_rate = stats['total_heads_to_prune'] / stats['total_heads'] if stats['total_heads'] > 0 else 0

            pl(f"\n=== Group: {head_count} heads ===")
            pl(f"  Number of layers: {stats['layer_count']}")
        
            pl(f"  Total heads: {stats['total_heads']}")
            pl(f"  Total heads to prune: {stats['total_heads_to_prune']} ({prune_rate:.2%})")
            pl(f"  Total heads to keep: {stats['total_heads_to_keep']} ({1-prune_rate:.2%})")
            pl(f"  Average per layer - Prune: {avg_pruned_per_layer:.1f}, Keep: {avg_kept_per_layer:.1f}")

    

    # Print overall statistics
    total_pruned_across_all = sum(stats['total_heads_to_prune'] for stats in group_stats.values())
    total_heads_across_all = sum(stats['total_heads'] for stats in group_stats.values())
    total_global_layers = sum(stats['global_layers'] for stats in group_stats.values())
    total_local_layers = sum(stats['local_layers'] for stats in group_stats.values())
    overall_prune_rate = total_pruned_across_all / total_heads_across_all if total_heads_across_all > 0 else 0

    pl(f"\n--- Overall Statistics ---")
    pl(f"Total layers: {total_global_layers + total_local_layers}")
    pl(f"  - Global threshold layers (.12., .20., .16.): {total_global_layers}")
    pl(f"  - Local threshold layers: {total_local_layers}")
    pl(f"Total heads across all layers: {total_heads_across_all}")
    pl(f"Total heads to prune: {total_pruned_across_all} ({overall_prune_rate:.2%})")
    pl(f"Total heads to keep: {total_heads_across_all - total_pruned_across_all} ({1-overall_prune_rate:.2%})")

    pl("===========================\n")