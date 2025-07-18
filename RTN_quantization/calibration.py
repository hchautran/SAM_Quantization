import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm


def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


# This code is implemented based on SAM input format at https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/sam.py
def get_act_scales_sam(model, batched_input, num_samples=512):
    """
    Get activation scales for SAM model focusing on image encoder and prompt encoder.
    
    Args:
        model: SAM model instance
        batched_input: List of input dictionaries for SAM
        num_samples: Number of samples to use for calibration
    
    Returns:
        dict: Dictionary mapping layer names to their activation scales (max absolute values per channel)
    """
    model.eval()
    device = next(model.parameters()).device 
    
    print(device)
    # exit()
    act_scales = {}

    def stat_tensor(name, tensor):
        """Calculate per-channel maximum absolute values"""
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        if tensor.dim() < 2:
            return
        
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        current_max = torch.max(tensor, dim=0)[0].float().cpu()
        
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], current_max)
        else:
            act_scales[name] = current_max

    def stat_input_hook(m, x, y, name):
        """Hook function to capture inputs to linear layers"""
        if x is None:
            return
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    
    # Hook image encoder linear layers (focus on attention QKV and after QKV)
    for name, module in model.image_encoder.named_modules():
        if isinstance(module, nn.Linear):
            full_name = f"image_encoder.{name}"
            if any(pattern in name for pattern in ['attn.qkv','attn.proj','mlp.lin1', 'mlp.lin2' ]):
                hooks.append(
                    module.register_forward_hook(functools.partial(stat_input_hook, name=full_name))
                )
    
    # Hook prompt encoder linear layers  
    for name, module in model.prompt_encoder.named_modules():
        if isinstance(module, nn.Linear):
            full_name = f"prompt_encoder.{name}"
            hooks.append(
                module.register_forward_hook(
                    functools.partial(stat_input_hook, name=full_name)
                )
            )

    print("Collecting activation scales for SAM...")
    num_samples = min(num_samples, len(batched_input))
    
    for i in tqdm(range(num_samples)):
        try:
            current_input = [batched_input[i % len(batched_input)]]
            for input_dict in current_input:
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.to(device)
            # Process image through image encoder
            input_images = torch.stack(
                [model.preprocess(x["image"]) for x in current_input], dim=0
            ).to(device)
            
            # Run image encoder (this will trigger hooks for image encoder)
            image_embeddings = model.image_encoder(input_images)
            
            # Process prompts through prompt encoder
            for image_record in current_input:
                if "point_coords" in image_record:
                    points = (
                        image_record["point_coords"].to(device) if image_record["point_coords"] is not None else None,
                        image_record["point_labels"].to(device) if image_record["point_labels"] is not None else None
                    )
                else:
                    points = None
                
                boxes = image_record.get("boxes", None)
                if boxes is not None:
                    boxes = boxes.to(device)
                    
                masks = image_record.get("mask_inputs", None)
                if masks is not None:
                    masks = masks.to(device)
                
                # Run prompt encoder (this will trigger hooks for prompt encoder)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=points,
                    boxes=boxes,
                    masks=masks,
                )
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Remove all hooks
    for hook in hooks:
        hook.remove()
    return act_scales
