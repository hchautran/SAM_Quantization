import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("../sam-hq")
from segment_anything import sam_model_registry, SamPredictor
from calibration import get_act_scales_sam
from utils import smooth_sam


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
        
        print(f"Score: {score:.3f}")
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
    plt.close()


def prepare_batched_input_for_calibration():
    """
    Prepare batched input in the format expected by get_act_scales_sam
    """
    batched_input = []
    
    for i in range(8):
        print(f"Preparing image: {i}")
        
        # Load and prepare image
        image = cv2.imread(f'demo/input_imgs/example{i}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to torch tensor and normalize (0-255 -> 0-1)
        image_tensor = torch.from_numpy(image).float() / 255.0
        # Transpose from HWC to CHW format
        image_tensor = image_tensor.permute(2, 0, 1)
        
        # Prepare input dictionary based on the example logic
        input_dict = {
            "image": image_tensor,
            "original_size": image.shape[:2],  # (H, W)
        }
        
        # Add prompts based on your existing logic
        if i == 0:
            input_box = np.array([[4, 13, 1007, 1023]])
            input_dict["boxes"] = torch.from_numpy(input_box).float()
            
        elif i == 1:
            input_box = np.array([[306, 132, 925, 893]])
            input_dict["boxes"] = torch.from_numpy(input_box).float()
            
        elif i == 2:
            input_point = np.array([[495, 518], [217, 140]])
            input_label = np.ones(input_point.shape[0], dtype=np.int32)
            input_dict["point_coords"] = torch.from_numpy(input_point).float().unsqueeze(0)  # Add batch dim
            input_dict["point_labels"] = torch.from_numpy(input_label).long().unsqueeze(0)   # Add batch dim
            
        elif i == 3:
            input_point = np.array([[221, 482], [498, 633], [750, 379]])
            input_label = np.ones(input_point.shape[0], dtype=np.int32)
            input_dict["point_coords"] = torch.from_numpy(input_point).float().unsqueeze(0)
            input_dict["point_labels"] = torch.from_numpy(input_label).long().unsqueeze(0)
            
        elif i == 4:
            input_box = np.array([[64, 76, 940, 919]])
            input_dict["boxes"] = torch.from_numpy(input_box).float()
            
        elif i == 5:
            input_point = np.array([[373, 363], [452, 575]])
            input_label = np.ones(input_point.shape[0], dtype=np.int32)
            input_dict["point_coords"] = torch.from_numpy(input_point).float().unsqueeze(0)
            input_dict["point_labels"] = torch.from_numpy(input_label).long().unsqueeze(0)
            
        elif i == 6:
            input_box = np.array([[181, 196, 757, 495]])
            input_dict["boxes"] = torch.from_numpy(input_box).float()
            
        elif i == 7:
            # Multi box input
            input_box = np.array([[45, 260, 515, 470], [310, 228, 424, 296]])
            input_dict["boxes"] = torch.from_numpy(input_box).float()
        
        batched_input.append(input_dict)
    
    return batched_input


if __name__ == "__main__":
    sam_checkpoint = "./checkpoint_sam/sam_hq_vit_b_1.pth"  # Updated path
    model_type = "vit_l"
    device = "cuda"
    
    # Load SAM model (use sam-hq for HQ checkpoint)
    
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    
    print("Preparing calibration data...")
    batched_input = prepare_batched_input_for_calibration()
    
    print("Calculating activation scales...")
    act_scales = get_act_scales_sam(sam, batched_input, num_samples=len(batched_input))
    
    smoothed_sam= smooth_sam(sam, act_scales, alpha=0.5)
    # Save the smoothed SAM model
    torch.save(smoothed_sam.state_dict(), 'smoothed_sam.pth')
    
    print(f"\nCollected activation scales for {len(act_scales)} layers:")
    for name, scales in act_scales.items():
        print(f"{name}: shape {scales.shape}, max {scales.max():.4f}, min {scales.min():.4f}")
    
    # Save activation scales for later use
    torch.save(act_scales, "sam_activation_scales.pt")
    print("Activation scales saved to 'sam_activation_scales.pt'")


    



