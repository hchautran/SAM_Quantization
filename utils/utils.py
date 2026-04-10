import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from segment_anything import SamPredictor, sam_model_registry
import pandas  as pd 
from typing import Optional, Tuple

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
def show_mask_image(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


# Utility Functions
def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def plot_mask_comparison(original_result, quantized_result, image_path, image_idx,save_path=None):
    """Plot side-by-side comparison of original and quantized masks"""
    import cv2
    
    # Load the original image
    image = cv2.imread(f'{image_path}/example{image_idx}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract masks and scores
    orig_mask = original_result['masks'][0]
    quant_mask = quantized_result['masks'][0]
    orig_score = original_result['scores'][0]
    quant_score = quantized_result['scores'][0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Original mask
    axes[0, 1].imshow(image)
    show_mask_image(orig_mask, axes[0, 1], random_color=False, borders=True)
    axes[0, 1].set_title(f'Original Model Mask\nScore: {orig_score:.4f}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Quantized mask
    axes[0, 2].imshow(image)
    show_mask_image(quant_mask, axes[0, 2], random_color=False, borders=True)
    axes[0, 2].set_title(f'Quantized Model Mask\nScore: {quant_score:.4f}', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Mask difference
    mask_diff = np.abs(orig_mask.astype(float) - quant_mask.astype(float))
    axes[1, 0].imshow(mask_diff, cmap='hot')
    axes[1, 0].set_title('Mask Difference\n(Red = High Difference)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Score comparison bar chart
    scores = [orig_score, quant_score]
    labels = ['Original', 'Quantized']
    colors = ['skyblue', 'lightcoral']
    bars = axes[1, 1].bar(labels, scores, color=colors, alpha=0.7)
    axes[1, 1].set_title('Score Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # Add score values on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Score difference
    score_diff = abs(orig_score - quant_score)
    score_degradation = (score_diff / orig_score) * 100
    
    axes[1, 2].text(0.5, 0.7, f'Score Difference: {score_diff:.6f}', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.5, f'Degradation: {score_degradation:.2f}%', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.3, f'Mask MSE: {np.mean(mask_diff):.6f}', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Quantization Impact', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mask comparison plot saved to {save_path}")
    
    plt.show()

def plot_individual_masks(original_result, quantized_result, image_path, save_path=None):
    """Plot individual masks separately for detailed analysis"""
    import cv2
    
    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract masks and scores
    orig_mask = original_result['masks'][0]
    quant_mask = quantized_result['masks'][0]
    orig_score = original_result['scores'][0]
    quant_score = quantized_result['scores'][0]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original mask
    axes[0, 0].imshow(image)
    show_mask_image(orig_mask, axes[0, 0], random_color=False, borders=True)
    axes[0, 0].set_title(f'Original Model Mask\nScore: {orig_score:.4f}', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Quantized mask
    axes[0, 1].imshow(image)
    show_mask_image(quant_mask, axes[0, 1], random_color=False, borders=True)
    axes[0, 1].set_title(f'Quantized Model Mask\nScore: {quant_score:.4f}', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Mask difference
    mask_diff = np.abs(orig_mask.astype(float) - quant_mask.astype(float))
    im = axes[1, 0].imshow(mask_diff, cmap='hot')
    axes[1, 0].set_title('Mask Difference\n(Red = High Difference)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Score comparison
    scores = [orig_score, quant_score]
    labels = ['Original', 'Quantized']
    colors = ['skyblue', 'lightcoral']
    bars = axes[1, 1].bar(labels, scores, color=colors, alpha=0.7)
    axes[1, 1].set_title('Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # Add score values on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual mask plot saved to {save_path}")
    
    plt.show()




def get_activation_boxplot(
    high_activations: torch.Tensor,
    low_activations: torch.Tensor,
    ax,
    token_wise=False,
    max_channels=64,
    offset=0,
    show_plot=True,
    diff: torch.Tensor = None,
):
    """
    Create violin plot comparing high and low activation distributions.

    Args:
        high_activations: High bit-width activations tensor
        low_activations: Low bit-width activations tensor
        ax: Matplotlib axis for plotting
        token_wise: Whether to plot token-wise or channel-wise
        max_channels: Maximum number of channels to visualize
        offset: Channel offset for visualization
        show_plot: Whether to display plot
        diff: Optional difference tensor to overlay
    """
    if not token_wise:
        high_data = to_numpy(
            high_activations.reshape(-1, high_activations.shape[-1])[
                :, offset : offset + max_channels
            ]
        )
        low_data = to_numpy(
            low_activations.reshape(-1, low_activations.shape[-1])[
                :, offset : offset + max_channels
            ]
        )
        high_channel_names = np.repeat(
            np.array([f"{i + 1}" for i in range(max_channels)]), high_data.shape[0]
        )
        low_channel_names = np.repeat(
            np.array([f"{i + 1}" for i in range(max_channels)]), low_data.shape[0]
        )
        high_data = high_data.flatten(order="F")
        low_data = low_data.flatten(order="F")

        types = ["high"] * (high_data.shape[0]) + ["low"] * (low_data.shape[0])

        df = pd.DataFrame(
            {
                "values": np.concatenate([high_data, low_data]),
                "channel": np.concatenate([high_channel_names, low_channel_names]),
                "types": types,
            }
        )

        # Create Plotly violin plot
        sns.violinplot(
            df, ax=ax, x="channel", y="values", hue="types", split=True, inner="quart"
        )
        if diff is not None:
            diff = to_numpy(diff.squeeze())[offset : offset + max_channels]
            print(diff.shape)
            print(len(high_channel_names))

            df = pd.DataFrame(
                {
                    "value": diff,
                    "channel": np.array([f"{i + 1}" for i in range(max_channels)]),
                }
            )
            sns.barplot(df, ax=ax, x="channel", y="value", alpha=0.5)

    else:
        if len(high_activations.shape) == 3:
            Bh, Th, Ch = high_activations.shape
            Bl, Tl, Cl = low_activations.shape
            high_data = to_numpy(
                high_activations.permute(1, 0, 2).reshape(Th, Bh, Ch)[
                    :, :, offset : offset + max_channels
                ]
            )
            low_data = to_numpy(
                low_activations.permute(1, 0, 2).reshape(Tl, Bl, Cl)[
                    :, :, offset : offset + max_channels
                ]
            )
        else:
            Bh, Hh, Th, Ch = high_activations.shape
            Bl, Hl, Tl, Cl = low_activations.shape
            high_data = to_numpy(
                high_activations.permute(2, 0, 1, 3).reshape(Th, Bh, Ch * Hh)[
                    :, :, offset : offset + max_channels
                ]
            )
            low_data = to_numpy(
                low_activations.permute(2, 0, 1, 3).reshape(Tl, Bl, Cl * Hl)[
                    :, :, offset : offset + max_channels
                ]
            )

        high_data = high_data.reshape(Th, -1)
        low_data = low_data.reshape(Tl, -1)

        high_token_names = np.repeat(
            np.array([f"{i + 1}" for i in range(Th)]), max_channels * Bh
        )
        low_token_names = np.repeat(
            np.array([f"{i + 1}" for i in range(Tl)]), max_channels * Bl
        )

        high_data = high_data.flatten(order="C")
        low_data = low_data.flatten(order="C")

        types = ["high"] * (Th * max_channels * Bh) + ["low"] * (Tl * max_channels * Bl)

        df = pd.DataFrame(
            {
                "values": np.concatenate([high_data, low_data]),
                "token": np.concatenate([high_token_names, low_token_names]),
                "types": types,
            }
        )
        sns.set_style("darkgrid")

        sns.violinplot(
            df, ax=ax, x="types", y="values", hue="types", split=True, inner="quart"
        )


# ============================================================================
# Inference Functions
# ============================================================================


def inference_with_sam_model(
    sam_model,
    image: np.ndarray,
    input_point: Optional[np.ndarray] = None,
    input_label: Optional[np.ndarray] = None,
    input_box: Optional[np.ndarray] = None,
    hq_token_only: bool = False,
):
    """
    Run inference directly with SAM model (without SamPredictor wrapper).

    Args:
        sam_model: SAM model instance
        image: Input image as numpy array
        input_point: Optional point prompts
        input_label: Optional labels for points
        input_box: Optional bounding box prompts
        hq_token_only: Whether to use high-quality token only

    Returns:
        Tuple of (masks, scores, logits)
    """
    # Make sure the entire model is on a single device
    device = next(sam_model.parameters()).device
    sam_model = sam_model.to(device)

    # Prepare image tensor
    input_image = torch.as_tensor(image).to(device).permute(2, 0, 1).contiguous()
    original_size = image.shape[:2]

    # Prepare batched input for Sam model
    batched_input = []
    dict_input = {"image": input_image, "original_size": original_size}

    # Add prompts if provided
    if input_point is not None and input_label is not None:
        point_coords = torch.as_tensor(input_point).to(device)
        point_labels = torch.as_tensor(input_label).to(device)
        dict_input["point_coords"] = point_coords
        dict_input["point_labels"] = point_labels

    if input_box is not None:
        boxes = torch.as_tensor(input_box).to(device)
        dict_input["boxes"] = boxes

    batched_input.append(dict_input)

    # Make sure the model is in eval mode
    sam_model.eval()

    # Force all model parameters to correct device
    for module in sam_model.modules():
        for param in module.parameters(recurse=False):
            param.data = param.data.to(device)
        for buffer in module.buffers(recurse=False):
            buffer.data = buffer.data.to(device)

    with torch.no_grad():
        outputs = sam_model(batched_input, multimask_output=False)
        if isinstance(outputs, tuple):
            outputs, interm_embeddings = outputs
        else:
            interm_embeddings = None

    # Extract results from outputs
    if len(outputs) > 0:
        output = outputs[0]
        masks = output["masks"].detach().cpu().numpy()
        scores = output["iou_predictions"].detach().cpu().numpy()
        logits = output["low_res_logits"].detach().cpu().numpy()
    else:
        # Fallback if no outputs
        h, w = original_size
        masks = np.zeros((1, h, w), dtype=bool)
        scores = np.array([0.0])
        logits = np.zeros((1, 256, 256))

    return masks, scores, logits




@torch.inference_mode()
def inference_image(
    predictor,
    image_dir: str = "./input_imgs/example1.png",
    show_image: bool = False,
    example_idx: int = 1,
):
    """
    Run inference on a single image.

    Args:
        predictor: SamPredictor or SAM model instance
        image_dir: Directory containing input images
        show_image: Whether to display and save visualization
        example_idx: Which example configuration to use (0-4)

    Returns:
        Tuple of (masks, scores, logits)
    """
    image = cv2.imread(f"{image_dir}/example{example_idx}.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Configure based on example index
    if example_idx == 0:
        input_box = np.array([[4, 13, 1007, 1023]])
        input_point, input_label = None, None
        hq_token_only = True
    elif example_idx == 1:
        input_box = np.array([[306, 132, 925, 893]])
        input_point, input_label = None, None
        hq_token_only = True
    elif example_idx == 2:
        input_point = np.array([[495, 518], [217, 140]])
        input_label = np.ones(input_point.shape[0])
        input_box = None
        hq_token_only = True
    elif example_idx == 3:
        input_point = np.array([[221, 482], [498, 633], [750, 379]])
        input_label = np.ones(input_point.shape[0])
        input_box = None
        hq_token_only = False
    elif example_idx == 4:
        input_box = np.array([[64, 76, 940, 919]])
        input_point, input_label = None, None
        hq_token_only = True
    else:
        # Default fallback
        input_box = np.array([[306, 132, 925, 893]])
        input_point, input_label = None, None
        hq_token_only = True

    # Run inference based on predictor type
    if isinstance(predictor, SamPredictor):
        print(image.shape)
        predictor.set_image(image)

        try:
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
                hq_token_only=hq_token_only,
            )
        except TypeError as e:
            if "hq_token_only" in str(e):
                print("Warning: hq_token_only not supported, using standard prediction")
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    multimask_output=False,
                )
            else:
                raise

    else:
        # Use direct SAM model inference
        masks, scores, logits = inference_with_sam_model(
            sam_model=predictor,
            image=image,
            input_point=input_point,
            input_label=input_label,
            input_box=input_box,
            hq_token_only=hq_token_only,
        )

    if show_image:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        if len(masks) > 0:
            show_mask_image(masks[0], plt.gca(), random_color=False)

        if input_box is not None:
            box = input_box[0]
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            plt.gca().add_patch(
                plt.Rectangle(
                    (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
                )
            )

        if input_point is not None and input_label is not None:
            show_points(input_point, input_label, plt.gca())

        plt.title(f"Example {example_idx} - Score: {scores[0]:.3f}")
        plt.savefig("demo.png")
        plt.axis("off")
        plt.show()

    return masks, scores, logits



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_tensor_3d(tensor, 
                   title="Tensor Visualization",
                   xlabel="Token/Position", 
                   ylabel="Channel/Feature",
                   zlabel="Absolute Value",
                   cmap='coolwarm',
                   figsize=(10, 8),
                   elev=25,
                   azim=45,
                   show_colorbar=False,
                   abs_value=True,
                   linewidth=0.5,
                   alpha=0.9):
    """
    Plot a 2D tensor as a 3D surface plot.
    
    Parameters:
    -----------
    tensor : torch.Tensor, np.ndarray, or list
        2D tensor to visualize. Shape should be (channels, tokens) or (height, width)
    title : str
        Plot title
    xlabel : str
        Label for x-axis (typically tokens/positions)
    ylabel : str
        Label for y-axis (typically channels/features)
    zlabel : str
        Label for z-axis (typically values)
    cmap : str
        Colormap name (e.g., 'coolwarm', 'viridis', 'plasma', 'RdBu')
    figsize : tuple
        Figure size (width, height)
    elev : float
        Elevation viewing angle
    azim : float
        Azimuth viewing angle
    show_colorbar : bool
        Whether to show colorbar
    abs_value : bool
        Whether to take absolute value of the tensor
    linewidth : float
        Width of grid lines
    alpha : float
        Transparency of surface (0-1)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Convert to numpy array if it's a PyTorch tensor
    if hasattr(tensor, 'detach'):  # PyTorch tensor
        data = tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        data = np.array(tensor)
    else:  # Already numpy
        data = np.array(tensor)
    
    # Ensure 2D
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        raise ValueError(f"Expected 2D tensor, got shape {data.shape}")
    
    # Take absolute value if requested
    if abs_value:
        data = np.abs(data)
    
    # Get dimensions
    channels, tokens = data.shape
    
    # Create meshgrid
    X = np.arange(0, tokens, 1)
    Y = np.arange(0, channels, 1)
    X, Y = np.meshgrid(X, Y)
    Z = data
    
    # Create the 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, 
                           linewidth=linewidth, 
                           alpha=alpha, antialiased=True)
    
    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Adjust the grid
    ax.grid(True, linewidth=0.5, alpha=0.5)
    
    # Set tick parameters
    ax.tick_params(labelsize=9)
    
    # Optional colorbar
    if show_colorbar:
        fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    return fig, ax


def plot_weight_activation_comparison(weight,
                                       activation_pre,
                                       activation_post,
                                       title_prefix="",
                                       cmap='coolwarm',
                                       figsize=(18, 5),
                                       elev=25,
                                       azim=45,
                                       show_colorbar=True,
                                       abs_value=True):
    """
    Plot weight, pre-projection activation, and post-projection activation side by side.

    Parameters:
    -----------
    weight : torch.Tensor or np.ndarray
        Weight matrix (out_features, in_features)
    activation_pre : torch.Tensor or np.ndarray
        Pre-projection activation (can be 2D, 3D, or 4D)
    activation_post : torch.Tensor or np.ndarray
        Post-projection activation (can be 2D, 3D, or 4D)
    title_prefix : str
        Prefix for subplot titles (e.g., "Layer 0 - Q")
    cmap : str
        Colormap name
    figsize : tuple
        Figure size (width, height)
    elev : float
        Elevation viewing angle
    azim : float
        Azimuth viewing angle
    show_colorbar : bool
        Whether to show colorbar
    abs_value : bool
        Whether to take absolute value

    Returns:
    --------
    fig : matplotlib figure object
    """

    # Convert tensors to numpy
    def to_numpy(tensor):
        if hasattr(tensor, 'detach'):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    weight_data = to_numpy(weight)
    pre_data = to_numpy(activation_pre)
    post_data = to_numpy(activation_post)

    # Process activations to 2D
    def process_activation(data):
        if data.ndim == 3:
            # (batch, tokens, channels) -> average over batch
            return data.mean(axis=0)
        elif data.ndim == 4:
            # (batch, heads, tokens, channels_per_head) -> average over batch and heads
            return data.mean(axis=(0, 1))
        return data

    pre_data = process_activation(pre_data)
    post_data = process_activation(post_data)

    # Take absolute value if requested
    if abs_value:
        weight_data = np.abs(weight_data)
        pre_data = np.abs(pre_data)
        post_data = np.abs(post_data)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=figsize)

    # Subplot 1: Weight
    ax1 = fig.add_subplot(131, projection='3d')
    out_features, in_features = weight_data.shape
    X1 = np.arange(0, in_features, 1)
    Y1 = np.arange(0, out_features, 1)
    X1, Y1 = np.meshgrid(X1, Y1)
    surf1 = ax1.plot_surface(X1, Y1, weight_data, cmap=cmap, alpha=0.9, antialiased=True)
    ax1.set_xlabel('Input Features', fontsize=10)
    ax1.set_ylabel('Output Features', fontsize=10)
    ax1.set_zlabel('Absolute Value' if abs_value else 'Value', fontsize=10)
    ax1.set_title(f'{title_prefix} Weight\nShape: {weight.shape}', fontsize=11, pad=10)
    ax1.view_init(elev=elev, azim=azim)
    if show_colorbar:
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Subplot 2: Pre-projection Activation
    ax2 = fig.add_subplot(132, projection='3d')
    # Transpose to (channels, tokens) for conventional view
    pre_data_T = pre_data.T
    channels_pre, tokens_pre = pre_data_T.shape
    X2 = np.arange(0, tokens_pre, 1)
    Y2 = np.arange(0, channels_pre, 1)
    X2, Y2 = np.meshgrid(X2, Y2)
    surf2 = ax2.plot_surface(X2, Y2, pre_data_T, cmap=cmap, alpha=0.9, antialiased=True)
    ax2.set_xlabel('Token/Position', fontsize=10)
    ax2.set_ylabel('Channel/Feature', fontsize=10)
    ax2.set_zlabel('Absolute Value' if abs_value else 'Value', fontsize=10)
    ax2.set_title(f'{title_prefix} Pre-Projection\nShape: {activation_pre.shape}', fontsize=11, pad=10)
    ax2.view_init(elev=elev, azim=azim)
    if show_colorbar:
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # Subplot 3: Post-projection Activation
    ax3 = fig.add_subplot(133, projection='3d')
    # Transpose to (channels, tokens) for conventional view
    post_data_T = post_data.T
    channels_post, tokens_post = post_data_T.shape
    X3 = np.arange(0, tokens_post, 1)
    Y3 = np.arange(0, channels_post, 1)
    X3, Y3 = np.meshgrid(X3, Y3)
    surf3 = ax3.plot_surface(X3, Y3, post_data_T, cmap=cmap, alpha=0.9, antialiased=True)
    ax3.set_xlabel('Token/Position', fontsize=10)
    ax3.set_ylabel('Channel/Feature', fontsize=10)
    ax3.set_zlabel('Absolute Value' if abs_value else 'Value', fontsize=10)
    ax3.set_title(f'{title_prefix} Post-Projection\nShape: {activation_post.shape}', fontsize=11, pad=10)
    ax3.view_init(elev=elev, azim=azim)
    if show_colorbar:
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

    plt.tight_layout()

    return fig

