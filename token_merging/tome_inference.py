from hilbert_utils import get_hilbert_order, get_hilbert_inverse
import torch
import torch.nn.functional as F

INSPECT_BLOCK = 11 
captured = {}

def _capturing_forward(self, x, _orig=ToMeSAMBlock.forward):
    out = _orig(self, x)
    m = self._tome_info.get("x_attn")
    if m is not None:
        captured["x_attn"] = m.detach().cpu()
    return out

target_block = sam.image_encoder.blocks[INSPECT_BLOCK]
target_block.forward = types.MethodType(_capturing_forward, target_block)

with torch.no_grad():
    predictor.set_image(image)
    predictor.predict(
        point_coords=None, point_labels=None,
        box=np.array([[306, 132, 925, 893]]),
        multimask_output=False, hq_token_only=True,
    )

del target_block.forward  # restore class method

m = captured["x_attn"]          # [B_win, T, hd]
B_win, T, hd = m.shape
ws = int(T ** 0.5)              # window size (14 for windowed, 64 for global)
input = input[0]

# ── 2. Permute to Hilbert order ───────────────────────────────────────────────
perm     = get_hilbert_order(ws, ws)     # [T]  raster→Hilbert
inv_perm = get_hilbert_inverse(ws, ws)   # [T]  Hilbert→raster

x = input[perm]                     # [T, hd]  tokens in Hilbert order



input= input.T.reshape(-1, 64,64)[None,...]# [B=1, C=1, H=ws, W=ws]
print(input.shape)
print(input.shape)  # [1, 64, 64]
# Sobel kernels
sobel_x = torch.tensor([[[[1.,  0., -1.],
                           [2.,  0., -2.],
                           [1.,  0., -1.]]]])  # horizontal edges

sobel_y = torch.tensor([[[[1.,  2.,  1.],
                           [0.,  0.,  0.],
                           [-1., -2., -1.]]]])  # vertical edges
sobel_x = sobel_x.view(1, 1, 3, 3).expand(-1, input.shape[1], -1, -1)
sobel_y = sobel_y.view(1, 1, 3, 3).expand(-1, input.shape[1], -1, -1)
print(sobel_x.shape)  # [1, 1, 3, 3]
print(sobel_y.shape)  # [1, 1, 3, 3]

x_pad = F.pad(input, (1, 1, 1, 1), mode='reflect')
edge_x = F.conv2d(x_pad, sobel_x, padding=0)
edge_y = F.conv2d(x_pad, sobel_y, padding=0)

# Gradient magnitude
magnitude = torch.sqrt(edge_x**2 + edge_y**2)

plt.imshow(magnitude.squeeze_())

order_1 = 2
order_2 = 4
order_3 = 8 
order_4 = 16 
orders = [order_1, order_2, order_3, order_4]
order_level = 1  # change to 0, 1, or 2 for different levels of aggregation
inv_perm = get_hilbert_inverse(64,64)
h_magnitude = magnitude.reshape(64**2)[perm].reshape(-1, orders[order_level]**2)
h_magnitude = h_magnitude.max(dim=-1)[0][..., None].expand(-1, orders[order_level]**2).flatten() 
print(h_magnitude.shape)  # [T//(order^2), 1]
plt.imshow(h_magnitude[inv_perm].cpu().numpy().reshape(64, 64), aspect="auto")


values = h_magnitude[inv_perm].reshape(64, 64).flatten()
p75 = np.percentile(values, 80) 

high_mask = torch.where(values > p75, 1, 0).reshape(64, 64)
plt.imshow(high_mask.reshape(64, 64), aspect="auto")