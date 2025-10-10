import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple
import torch.nn.functional as F
import os
import pickle
from sklearn.metrics import mean_squared_error
from segment_anything.utils.transforms import ResizeLongestSide
def show_mask(mask, ax, random_color=False):

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score.item():.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.show()
    plt.close()

##############################################################
def generate_random_bboxes(n, image_shape):

    height, width = image_shape[:2]
    bboxes = []
    
    for _ in range(n):
        # Generate two random points
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(0, width)
        y2 = np.random.randint(0, height)
        
        # Ensure x1 < x2 and y1 < y2 (top-left, bottom-right format)
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Ensure minimum box size of 1x1
        if x_max == x_min:
            x_max = min(x_min + 1, width - 1)
        if y_max == y_min:
            y_max = min(y_min + 1, height - 1)
            
        bboxes.append([x_min, y_min, x_max, y_max])
    
    return np.array(bboxes)
def postprocess_masks(
        masks: torch.Tensor,
        image_encoder_image_size,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (image_encoder_image_size, image_encoder_image_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks.cpu()
def format_input_for_sam(image, input_point=None, input_label=None, input_box=None, device='cuda'):
    input_image = torch.as_tensor(image.astype(dtype=np.uint8), device=device).permute(2, 0, 1).contiguous()

    dict_input = {
        'image': input_image,
        'original_size': image.shape[:2]  # (H, W)
    }
    
    # Format points with consistent dimensions
    if input_point is not None and input_label is not None:
        # Ensure points are in correct format: (batch_size, num_points, 2)
        if isinstance(input_point, np.ndarray):
            input_point = torch.as_tensor(input_point, device=device, dtype=torch.float)
        if isinstance(input_label, np.ndarray):
            input_label = torch.as_tensor(input_label, device=device, dtype=torch.int)
            
        # Ensure 3D tensor format for points: (1, N, 2)
        if input_point.dim() == 2:
            input_point = input_point.unsqueeze(0)  # Add batch dimension
        if input_label.dim() == 1:
            input_label = input_label.unsqueeze(0)  # Add batch dimension
            
        dict_input['point_coords'] = input_point
        dict_input['point_labels'] = input_label
    
    # Format boxes with consistent dimensions
    if input_box is not None:
        if isinstance(input_box, np.ndarray):
            input_box = torch.as_tensor(input_box, device=device, dtype=torch.float)
        
        # Ensure 2D tensor format for boxes: (N, 4)
        if input_box.dim() == 1:
            input_box = input_box.unsqueeze(0)  # Add batch dimension
            
        dict_input['boxes'] = input_box

    return [dict_input]
def compare_and_visualize(original_file, state, current_masks, current_scores, input_point, input_label, input_box, image, save_path):
    """
    Compare original model outputs with current model outputs and create visualization
    """
    # Create comparison directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load original results
    with open(original_file, 'rb') as f:
        original_data = pickle.load(f)
        original_masks = original_data['masks']
        original_scores = original_data['scores']
    
    # Create 2x3 subplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison Analysis', fontsize=25)
    
    # 1. Original image (top-left)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=20)
    axes[0, 0].axis('off')
    
    # 2. Original image with original mask (top-middle)
    axes[0, 1].imshow(image)
    if len(original_masks) > 0:
        show_mask(original_masks[0], axes[0, 1])
        if input_box is not None:
            box_to_show = input_box[0] if len(input_box.shape) > 1 else input_box
            show_box(box_to_show, axes[0, 1])
        if input_point is not None and input_label is not None:
            show_points(input_point, input_label, axes[0, 1])
    axes[0, 1].set_title('Original Model Output', fontsize=20)
    axes[0, 1].axis('off')
    
    # 3. Original image with modified model mask (top-right)
    axes[0, 2].imshow(image)
    if len(current_masks) > 0:
        show_mask(current_masks[0], axes[0, 2])
        if input_box is not None:
            box_to_show = input_box[0] if len(input_box.shape) > 1 else input_box
            show_box(box_to_show, axes[0, 2])
        if input_point is not None and input_label is not None:
            show_points(input_point, input_label, axes[0, 2])
    axes[0, 2].set_title('Modified Model Output-'+state, fontsize=20)
    axes[0, 2].axis('off')
    
    # 4. Mask difference (bottom-left)
    if len(original_masks) > 0 and len(current_masks) > 0:
        original_mask_np = original_masks[0].cpu().numpy() if isinstance(original_masks[0], torch.Tensor) else original_masks[0]
        current_mask_np = current_masks[0].cpu().numpy() if isinstance(current_masks[0], torch.Tensor) else current_masks[0]
        
        # Squeeze to remove single dimensions and ensure 2D
        original_mask_np = np.squeeze(original_mask_np)
        current_mask_np = np.squeeze(current_mask_np)
        
        # Create binary difference mask (black and white)
        mask_diff = (original_mask_np != current_mask_np).astype(np.uint8)
        
        # Display as black (0) and white (1) image
        im = axes[1, 0].imshow(mask_diff, cmap='gray', vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[1, 0], shrink=0.6)
    axes[1, 0].set_title('Mask Difference\n(Black: Same, White: Different)',fontsize=20)
    axes[1, 0].axis('off')
    
    # 5. Score comparison bar chart (bottom-middle)
    if len(original_scores) > 0 and len(current_scores) > 0:
        categories = ['Original Model', 'Modified Model']
        original_score = original_scores[0].item() if isinstance(original_scores[0], torch.Tensor) else original_scores[0]
        current_score = current_scores[0].item() if isinstance(current_scores[0], torch.Tensor) else current_scores[0]
        
        scores = [original_score, current_score]
        colors = ['blue', 'orange']
        bars = axes[1, 1].bar(categories, scores, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Score', fontsize=15)
        axes[1, 1].set_title('Score Comparison', fontsize=20)
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
    
    # 6. Metrics information (bottom-right)
    axes[1, 2].axis('off')
    if len(original_masks) > 0 and len(current_masks) > 0 and len(original_scores) > 0 and len(current_scores) > 0:
        # Extract scalar values from tensors
        original_score = original_scores[0].item() if isinstance(original_scores[0], torch.Tensor) else original_scores[0]
        current_score = current_scores[0].item() if isinstance(current_scores[0], torch.Tensor) else current_scores[0]
        
        # Calculate metrics
        score_diff = abs(original_score - current_score)
        degradation_pct = (score_diff / original_score) * 100 if original_score != 0 else 0

        # Convert tensors to numpy arrays and flatten
        orig_flat = original_masks[0].cpu().numpy().flatten().astype(float) if isinstance(original_masks[0], torch.Tensor) else original_masks[0].flatten().astype(float)
        curr_flat = current_masks[0].cpu().numpy().flatten().astype(float) if isinstance(current_masks[0], torch.Tensor) else current_masks[0].flatten().astype(float)
        mask_mse = mean_squared_error(orig_flat, curr_flat)
        
        # Create text summary
        metrics_text = f"""Model Comparison Metrics:

                        Score Difference: {score_diff:.4f}

                        Degradation: {degradation_pct:.2f}%

                        Mask MSE: {mask_mse:.6f}

                        Original Score: {original_score:.4f}
                        Modified Score: {current_score:.4f}"""
        
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='top')
    
    axes[1, 2].set_title('Metrics Summary',fontsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_baseline_results(masks, scores, save_path):
    """Save original model results as baseline"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    baseline_data = {
        'masks': masks,
        'scores': scores
    }
    with open(save_path, 'wb') as f:
        pickle.dump(baseline_data, f)
def reset_everything():
    """Complete reset between runs"""
    # 1. Clear all loggers
    logging.shutdown()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.Logger.manager.loggerDict.clear()

    # 2. Destroy distributed process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # 3. Clear CUDA cache
    torch.cuda.empty_cache()

    # 4. Force garbage collection
    import gc
    gc.collect()