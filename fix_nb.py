import json
import sys
import os

nb_path = '/media/volume/Chau/SAM_Quantization/notebook.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

# The cell we want to replace contains "def plot_local_attention"
target_cell_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'def plot_local_attention' in source:
            target_cell_idx = i
            break

new_source = [
    "import torch.nn as nn\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "\n",
    "def plot_local_attention(model, img_tensor, block_idx=1, head_idx=0, window_idx=None):\n",
    "    \"\"\"\n",
    "    Plots the attention map of the center token for a selected window.\n",
    "    \"\"\"\n",
    "    block = model.image_encoder.blocks[block_idx]\n",
    "    if block.window_size == 0:\n",
    "        print(f\"Block {block_idx} is a Global block. Choose a block with window_size > 0.\")\n",
    "        return\n",
    "\n",
    "    captured_attn = {}\n",
    "    orig_forward = block.attn.forward\n",
    "\n",
    "    def temp_forward(self, x, *args, **kwargs):\n",
    "        # Handle both 4D (B, H, W, C) and 3D (B, N, C) inputs\n",
    "        if x.dim() == 4:\n",
    "            B, H, W, C = x.shape\n",
    "            N = H * W\n",
    "            x_reshaped = x.reshape(B, N, C)\n",
    "        else:\n",
    "            B, N, C = x.shape\n",
    "            x_reshaped = x\n",
    "\n",
    "        qkv = self.qkv(x_reshaped).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)\n",
    "        q, k, v = qkv.unbind(0)\n",
    "        \n",
    "        attn = (q * self.scale) @ k.transpose(-2, -1)\n",
    "        attn = attn.softmax(dim=-1)\n",
    "        captured_attn['matrix'] = attn.detach()\n",
    "        \n",
    "        return orig_forward(x, *args, **kwargs)\n",
    "\n",
    "    block.attn.forward = temp_forward.__get__(block.attn, type(block.attn))\n",
    "\n",
    "    with torch.no_grad():\n",
    "        model.image_encoder(img_tensor)\n",
    "\n",
    "    block.attn.forward = orig_forward\n",
    "\n",
    "    attn_matrix = captured_attn['matrix'] \n",
    "    ws = block.window_size\n",
    "    num_heads = block.attn.num_heads\n",
    "    \n",
    "    num_windows = attn_matrix.shape[0] // num_heads\n",
    "    if window_idx is None:\n",
    "        window_idx = num_windows // 2\n",
    "    \n",
    "    window_attn = attn_matrix[window_idx * num_heads + head_idx]\n",
    "    q_idx = (ws // 2) * ws + (ws // 2)\n",
    "    q_attn = window_attn[q_idx].reshape(ws, ws).cpu().numpy()\n",
    "\n",
    "    plt.figure(figsize=(6, 5))\n",
    "    plt.title(f\"Block {block_idx}, Head {head_idx}, Window {window_idx}\\nAttention for center token\")\n",
    "    im = plt.imshow(q_attn, cmap='viridis')\n",
    "    plt.colorbar(im)\n",
    "    plt.axis('off')\n",
    "    plt.show()\n",
    "\n",
    "def plot_permuted_attention(model, img_tensor, block_idx=23, query_coords=(512, 512)):\n",
    "    \"\"\"\n",
    "    Plots the attention map after applying permute order of tile_stride_matching.\n",
    "    \"\"\"\n",
    "    block = model.image_encoder.blocks[block_idx]\n",
    "    captured = {}\n",
    "    \n",
    "    def hook_fn(module, input, output):\n",
    "        captured['perm'] = output[1].detach()\n",
    "        captured['inv_perm'] = output[2].detach()\n",
    "        return output[0]\n",
    "\n",
    "    handle = block.attn.register_forward_hook(hook_fn)\n",
    "    \n",
    "    qkv_captured = {}\n",
    "    def qkv_hook(module, input, output):\n",
    "        qkv_captured['qkv'] = output.detach()\n",
    "        \n",
    "    qkv_handle = block.attn.qkv.register_forward_hook(qkv_hook)\n",
    "\n",
    "    with torch.no_grad():\n",
    "        model.image_encoder(img_tensor)\n",
    "        \n",
    "    handle.remove()\n",
    "    qkv_handle.remove()\n",
    "    \n",
    "    grid_size = 64 if block.window_size == 0 else block.window_size\n",
    "    ty, tx = query_coords[0] // (1024 // grid_size), query_coords[1] // (1024 // grid_size)\n",
    "    raster_idx = ty * grid_size + tx\n",
    "    \n",
    "    head_idx = 0 \n",
    "    perm = captured['perm'][head_idx] \n",
    "    if perm.dim() > 1: perm = perm[0]\n",
    "    \n",
    "    qkv = qkv_captured['qkv']\n",
    "    B, N, _ = qkv.shape\n",
    "    H = block.attn.num_heads\n",
    "    D = _ // (3 * H)\n",
    "    qkv = qkv.view(B, N, 3, H, D).permute(2, 0, 3, 1, 4)\n",
    "    q, k, v = qkv.unbind(0)\n",
    "    \n",
    "    perm_e = perm.view(1, 1, N, 1).expand(B, H, -1, D)\n",
    "    q_p = q.gather(2, perm_e)\n",
    "    k_p = k.gather(2, perm_e)\n",
    "    \n",
    "    inv_perm = captured['inv_perm'][head_idx]\n",
    "    if inv_perm.dim() > 1: inv_perm = inv_perm[0]\n",
    "    q_p_idx = inv_perm[raster_idx]\n",
    "    \n",
    "    q_vec = q_p[0, head_idx, q_p_idx]\n",
    "    k_mat = k_p[0, head_idx]\n",
    "    \n",
    "    logits = (q_vec * block.attn.scale) @ k_mat.t()\n",
    "    attn_vec_perm = logits.softmax(dim=-1)\n",
    "    \n",
    "    attn_raster = torch.zeros(grid_size * grid_size, device=attn_vec_perm.device)\n",
    "    attn_raster.scatter_(0, perm, attn_vec_perm)\n",
    "    \n",
    "    attn_map = attn_raster.reshape(grid_size, grid_size).cpu().numpy()\n",
    "    \n",
    "    plt.figure(figsize=(6, 5))\n",
    "    plt.imshow(attn_map, cmap='viridis')\n",
    "    plt.title(f\"Permuted Attention Map (Block {block_idx})\\nQuery at {query_coords}\")\n",
    "    plt.colorbar()\n",
    "    plt.show()\n",
    "\n",
    "# Example usage:\n",
    "# plot_local_attention(sam_cs, img_tensor, block_idx=1, head_idx=5)\n",
    "plot_permuted_attention(sam_cs, img_tensor, block_idx=23, query_coords=(512, 512))\n"
]

if target_cell_idx != -1:
    nb['cells'][target_cell_idx]['source'] = new_source
else:
    # Append if not found
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_source
    })

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Updated cell {target_cell_idx}")
