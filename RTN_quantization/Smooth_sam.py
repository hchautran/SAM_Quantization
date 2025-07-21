import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys

# Add the sam-hq directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up to SAM_Quantization
sam_hq_path = os.path.join(project_root, "sam-hq")

sys.path.insert(0, sam_hq_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_anything import sam_model_registry, SamPredictor
from calibration import get_act_scales_sam
from utils import smooth_sam



def prepare_batched_input_for_calibration():
    """
    Prepare batched input in the format expected by get_act_scales_sam
    Returns CPU tensors that can be moved to device later
    """
    batched_input = []
    
    for i in range(8):
        print(f"Preparing image: {i}")
        
        # Load and prepare image
        image = cv2.imread(f'/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/sam-hq/demo/input_imgs/example{i}.png')
        if image is None:
            print(f"Error: Could not load image {i}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to torch tensor and normalize (0-255 -> 0-1)
        image_tensor = torch.from_numpy(image).float() / 255.0
        # Transpose from HWC to CHW format
        image_tensor = image_tensor.permute(2, 0, 1)
        
        # Prepare input dictionary - keep all tensors on CPU
        input_dict = {
            "image": image_tensor,  # CPU tensor
            "original_size": image.shape[:2],  # (H, W)
        }
        
        # Add prompts based on your existing logic - all CPU tensors
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
    sam_checkpoint = "./checkpoint_sam/sam_hq_vit_l_1.pth"  # Updated path
    model_type = "vit_l"
    device = "cuda"
    act_scales_file = None
    
    # Load SAM model (use sam-hq for HQ checkpoint)
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
  
    # Load or calculate activation scales
    act_scales_file = f"./checkpoint_act/sam_{model_type}activation_scales.pt"

    if os.path.exists(act_scales_file):
        print("Loading activation scales...")
        act_scales = torch.load(act_scales_file)
    else:
        batched_input = prepare_batched_input_for_calibration()
        act_scales = get_act_scales_sam(sam, batched_input, num_samples=len(batched_input))
        torch.save(act_scales, act_scales_file)
        print("Activation scales saved!")
   
    # Apply smoothing
    smoothed_sam = smooth_sam(sam, act_scales, alpha=0.5)

    # Save smoothed model
    torch.save(smoothed_sam.state_dict(), f'./checkpoint_sam/smoothed_{model_type}_sam.pth')
    print("Smoothed SAM model saved!")
    
    
    


    



