import model_utils
import torch
import typing
import utils
import transformers
import tqdm, math

from hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
from fast_hadamard_transform import hadamard_transform

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
            


################## move this function to model_utils.py ##################
def get_embeddings_sam(model) -> list[torch.nn.Module]:
    
    """
    Get embedding layers from SAM model.
    SAM has different embedding components in image_encoder, prompt_encoder, and mask_decoder.
    """
    embeddings = []
    
    #Image encoder embeddings
    if hasattr(model.image_encoder, 'patch_embed') and hasattr(model.image_encoder.patch_embed, 'proj'):
        embeddings.append(model.image_encoder.patch_embed.proj)
    
    
    # Mask decoder embeddings
    # if hasattr(model.mask_decoder, 'iou_token'):
    #     embeddings.append(model.mask_decoder.iou_token)
    # if hasattr(model.mask_decoder, 'mask_tokens'):
    #     embeddings.append(model.mask_decoder.mask_tokens)
    # if hasattr(model.mask_decoder, 'hf_token'):
    #     embeddings.append(model.mask_decoder.hf_token)
    
    return embeddings

##################################################################################

def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

def fuse_layer_norms_sam(model):
    """
    Fuse layer normalization operations into adjacent linear layers for SAM model.
    SAM uses LayerNorm in various components of image_encoder, prompt_encoder, and mask_decoder.
    """
    
    # Get embedding layers for fusion
    embeddings = get_embeddings_sam(model)
    for W in embeddings:
        if hasattr(W, 'weight'):
            W_ = W.weight.data.double()
            W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    
    # Process image encoder blocks
    if hasattr(model.image_encoder, 'blocks'):
        for block in model.image_encoder.blocks[:-1]:
            
            fuse_ln_linear(block.norm1, [block.attn.qkv])
            fuse_ln_linear(block.norm2, [block.mlp.lin1])
            bake_mean_into_linear(block.attn.proj)
            bake_mean_into_linear(block.mlp.lin2)
        
        ## TODO: Find the final layer or block to fuse ( this to ensure the same output)
        
        #fuse the final block
        final_block = model.image_encoder.blocks[-1]
        fuse_ln_linear(final_block.norm1, [final_block.attn.qkv])
        bake_mean_into_linear(final_block.attn.proj)
        fuse_ln_linear(final_block.norm2, [final_block.mlp.lin1])
    
    # Process mask decoder transformer layers with CORRECTED ORDER
    if hasattr(model.mask_decoder, 'transformer') and hasattr(model.mask_decoder.transformer, 'layers'):
        for layer in model.mask_decoder.transformer.layers[:-1]:
            
            pass
            # Order 1: norm1 -> cross_attn_token_to_image
            
            ## TODO: Reimplement this as this work not the same as the self attention 
            # if hasattr(layer, 'norm1') and hasattr(layer, 'cross_attn_token_to_image'):
            #     linear_layers = []
            #     if hasattr(layer.cross_attn_token_to_image, 'q_proj'):
            #         linear_layers.append(layer.cross_attn_token_to_image.q_proj)
            #     if hasattr(layer.cross_attn_token_to_image, 'k_proj'):
            #         linear_layers.append(layer.cross_attn_token_to_image.k_proj)
            #     if hasattr(layer.cross_attn_token_to_image, 'v_proj'):
            #         linear_layers.append(layer.cross_attn_token_to_image.v_proj)
            #     if linear_layers:
            #         fuse_ln_linear(layer.norm1, linear_layers)
                
            #     # Bake mean into output projection
            #     if hasattr(layer.cross_attn_token_to_image, 'out_proj'):
            #         bake_mean_into_linear(layer.cross_attn_token_to_image.out_proj)
            
            # Order 2: norm2 -> mlp.lin1 
            # This may be not good as we center the embedding is far from here
            # if hasattr(layer, 'norm2') and hasattr(layer, 'mlp'):
            #     fuse_ln_linear(layer.norm2, [layer.mlp.lin1])
            #     bake_mean_into_linear(layer.mlp.lin2)

            
            # Order 3: norm3 -> cross_attn_image_to_token
            ## TODO: Reimplement this part as this work not the same as the self attention 
            # if hasattr(layer, 'norm3') and hasattr(layer, 'cross_attn_image_to_token'):
            #     linear_layers = []
            #     if hasattr(layer.cross_attn_image_to_token, 'q_proj'):
            #         linear_layers.append(layer.cross_attn_image_to_token.q_proj)
            #     if hasattr(layer.cross_attn_image_to_token, 'k_proj'):
            #         linear_layers.append(layer.cross_attn_image_to_token.k_proj)
            #     if hasattr(layer.cross_attn_image_to_token, 'v_proj'):
            #         linear_layers.append(layer.cross_attn_image_to_token.v_proj)
            #     if linear_layers:
            #         fuse_ln_linear(layer.norm3, linear_layers)
                
            #     if hasattr(layer.cross_attn_image_to_token, 'out_proj'):
            #         bake_mean_into_linear(layer.cross_attn_image_to_token.out_proj)
            
        # final_layer = model.mask_decoder.transformer.layers[-1]
        # fuse_ln_linear(final_layer.norm2, [final_layer.mlp.lin1])
    
   
    
    # Replace all remaining LayerNorm modules with RMSN (following the original pattern)
    def create_rmsn_factory():
        """Factory function to create RMSN from LayerNorm"""
        def create_rmsn(layernorm_module):
            if hasattr(layernorm_module, 'normalized_shape'):
                if isinstance(layernorm_module.normalized_shape, int):
                    mean_dim = layernorm_module.normalized_shape
                else:
                    mean_dim = layernorm_module.normalized_shape[0]
            else:
                # Fallback for LayerNorm2d or other variants
                mean_dim = layernorm_module.weight.shape[0] if hasattr(layernorm_module, 'weight') else 256
            
            return model_utils.RMSN(mean_dim=mean_dim, eps=getattr(layernorm_module, 'eps', 1e-5))
        return create_rmsn
    
    # Replace all LayerNorm instances with RMSN (following original code pattern)
    # TODO: replace only fused LayerNorms
    model_utils.replace_modules(
        model,
        torch.nn.LayerNorm,
        create_rmsn_factory(),
        replace_layers=False,
    )


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

    
def rotate_embeddings(model, Q_image_encoder: torch.Tensor, Q_mask_decoder: torch.Tensor, args) -> None:
    """
    Rotate the embeddings using specific Q matrices for different SAM components.
    
    Args:
        model: SAM model instance
        Q_image_encoder: Rotation matrix for image encoder embeddings
        Q_mask_decoder: Rotation matrix for mask decoder embeddings  
        args: Arguments containing device information
    """
    
    # Image encoder embeddings - use Q_image_encoder
    if hasattr(model.image_encoder, 'patch_embed') and hasattr(model.image_encoder.patch_embed, 'proj'):
        W = model.image_encoder.patch_embed.proj
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q_image_encoder).to(device="cpu", dtype=dtype)
    
    # Prompt encoder embeddings - use Q_mask_decoder (since they interact with mask decoder)
    # Point embeddings (ModuleList of Embedding layers)
    # if hasattr(model.prompt_encoder, 'point_embeddings'):
    #     for point_embed in model.prompt_encoder.point_embeddings:
    #         dtype = point_embed.weight.data.dtype
    #         W_ = point_embed.weight.data.to(device=args.device, dtype=torch.float64)
    #         point_embed.weight.data = torch.matmul(W_, Q_mask_decoder).to(device="cpu", dtype=dtype)
    
    # # Not-a-point embedding
    # if hasattr(model.prompt_encoder, 'not_a_point_embed'):
    #     W = model.prompt_encoder.not_a_point_embed
    #     dtype = W.weight.data.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(W_, Q_mask_decoder).to(device="cpu", dtype=dtype)
    
    # # No mask embedding
    # if hasattr(model.prompt_encoder, 'no_mask_embed'):
    #     W = model.prompt_encoder.no_mask_embed
    #     dtype = W.weight.data.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(W_, Q_mask_decoder).to(device="cpu", dtype=dtype)
    
    # # Mask decoder embeddings - use Q_mask_decoder
    # # IoU token
    # if hasattr(model.mask_decoder, 'iou_token'):
    #     W = model.mask_decoder.iou_token
    #     dtype = W.weight.data.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(W_, Q_mask_decoder).to(device="cpu", dtype=dtype)
    
    # # Mask tokens
    # if hasattr(model.mask_decoder, 'mask_tokens'):
    #     W = model.mask_decoder.mask_tokens
    #     dtype = W.weight.data.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(W_, Q_mask_decoder).to(device="cpu", dtype=dtype)
    
    # # HQ token (HQ-SAM specific)
    # if hasattr(model.mask_decoder, 'hf_token'):
    #     W = model.mask_decoder.hf_token
    #     dtype = W.weight.data.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(W_, Q_mask_decoder).to(device="cpu", dtype=dtype)
@torch.inference_mode()
def rotate_model_sam(model, Q_image_encoder, Q_mask_decoder,args):
    """
    Rotate the weights of the SAM model using the rotation matrix Q.
    
    Args:
        model: The SAM model instance.
        Q: The rotation matrix.
    """
    
    
    # block_test = block_test
    # W = block_test.attn.qkv
    # dtype = W.weight.dtype
    # W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    # W.weight.data = torch.matmul(W_, Q_image_encoder).to(device="cpu", dtype=dtype)
    # for block_test in model.image_encoder.blocks[1:-1]:
    #     W = block_test.attn.proj
    #     dtype = W.weight.data.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(Q_image_encoder.T, W_).to(device="cpu", dtype=dtype)
    #     if W.bias is not None:
    #         b = W.bias.data.to(device=args.device, dtype=torch.float64)
    #         W.bias.data = torch.matmul(Q_image_encoder.T, b).to(device="cpu", dtype=dtype)
            
    #     W = block_test.mlp.lin1
    #     dtype = W.weight.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(W_, Q_image_encoder).to(device="cpu", dtype=dtype)
    
   
    # Image encoder attention input (qkv)
    for block in model.image_encoder.blocks[1:]:
        W = block.attn.qkv
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q_image_encoder).to(device="cpu", dtype=dtype)
       
    
    # Image encoder attention output (proj)
    for block in model.image_encoder.blocks[1:]:
        W = block.attn.proj
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q_image_encoder.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=args.device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q_image_encoder.T, b).to(device="cpu", dtype=dtype)
            
    # Mask decoder attention input (q_proj, k_proj, v_proj)
    # for layer in model.mask_decoder.transformer.layers:
    #     for W in [
    #         layer.cross_attn_token_to_image.q_proj,
    #         layer.cross_attn_token_to_image.k_proj,
    #         layer.cross_attn_token_to_image.v_proj,
    #         layer.cross_attn_image_to_token.q_proj,
    #         layer.cross_attn_image_to_token.k_proj,
    #         layer.cross_attn_image_to_token.v_proj,
    #         layer.self_attn.q_proj,
    #         layer.self_attn.k_proj,
    #         layer.self_attn.v_proj,
    #     ]:
    #         dtype = W.weight.dtype
    #         W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #         W.weight.data = torch.matmul(W_, Q_mask_decoder).to(device="cpu", dtype=dtype)

    
    
    # Mask decoder attention output (out_proj)
    # for layer in model.mask_decoder.transformer.layers:
    #     for W in [
    #         layer.cross_attn_token_to_image.out_proj,
    #         layer.cross_attn_image_to_token.out_proj,
    #         layer.self_attn.out_proj,
    #     ]:
    #         dtype = W.weight.data.dtype
    #         W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #         W.weight.data = torch.matmul(Q_mask_decoder.T, W_).to(device="cpu", dtype=dtype)
    #         if W.bias is not None:
    #             b = W.bias.data.to(device=args.device, dtype=torch.float64)
    #             W.bias.data = torch.matmul(Q_mask_decoder.T, b).to(device="cpu", dtype=dtype)

   
    # Image encoder MLP input (lin1)
    # for block in model.image_encoder.blocks[1:-1]:
    #     W = block.mlp.lin1
    #     dtype = W.weight.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(W_, Q_image_encoder).to(device="cpu", dtype=dtype)
        
    # # Image encoder MLP output (lin2)
    # for block in model.image_encoder.blocks[1:-1]:
    #     W = block.mlp.lin2
    #     dtype = W.weight.data.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(Q_image_encoder.T, W_).to(device="cpu", dtype=dtype)
    #     apply_exact_had_to_linear(W, had_dim=-1, output=False)  # Apply exact (inverse) hadamard
    #     if W.bias is not None:
    #         b = W.bias.data.to(device=args.device, dtype=torch.float64)
    #         W.bias.data = torch.matmul(Q_image_encoder.T, b).to(device="cpu", dtype=dtype)
    
    
    # # Mask decoder MLP input (first layer in MLP)
    # for layer in model.mask_decoder.transformer.layers:
    #     W = layer.mlp.lin1  # First layer in MLP
    #     dtype = W.weight.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(W_, Q_mask_decoder).to(device="cpu", dtype=dtype)    
    
    # # Mask decoder MLP output (last layer in MLP)
    # for layer in model.mask_decoder.transformer.layers:
    #     W = layer.mlp.lin2  # Last layer in MLP
    #     dtype = W.weight.data.dtype
    #     W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    #     W.weight.data = torch.matmul(Q_mask_decoder.T, W_).to(device="cpu", dtype=dtype)
    #     apply_exact_had_to_linear(W, had_dim=-1, output=False)  # Apply exact (inverse) hadamard
    #     if W.bias is not None:
    #         b = W.bias.data.to(device=args.device, dtype=torch.float64)
    #         W.bias.data = torch.matmul(Q_mask_decoder.T, b).to(device="cpu", dtype=dtype)

    # rotate qv projections
    # this is  possible to apply had to v and output projections as they are applied consecutively 
    # head_dim = (args.hidden_size_image_en // args.num_attention_head_image_en)
    # for block in model.image_encoder.blocks:
    #     if hasattr(block.attn, 'qkv'):
    #         embed_dim = args.hidden_size_image_en
    #         qkv_weight = block.attn.qkv.weight
    #         v_start = 2 * embed_dim  # V starts at 2/3
    #         v_end = 3 * embed_dim    # V ends at full dimension
            
    #         v_proj_temp = torch.nn.Linear(embed_dim, embed_dim, bias=False)
    #         v_proj_temp.weight.data = qkv_weight[v_start:v_end, :].clone()
    #         apply_exact_had_to_linear(v_proj_temp, had_dim=head_dim, output=True)
            
    #         qkv_weight[v_start:v_end, :] = v_proj_temp.weight.data
            
    #         # Apply Hadamard to output projection
    #         if hasattr(block.attn, 'proj'):
    #             apply_exact_had_to_linear(block.attn.proj, had_dim=-1, output=False)
    # this is  possible to apply had to v and output projections as they are applied consecutively
    # head_dim = args.hidden_size_mask_de // args.num_attention_head_mask_de
    # for layer in model.mask_decoder.transformer.layers:
    #     apply_exact_had_to_linear(layer.self_attn.v_proj, had_dim=head_dim, output=True)
    #     apply_exact_had_to_linear(layer.self_attn.out_proj, had_dim=-1, output=False)
    #     apply_exact_had_to_linear(layer.cross_attn_token_to_image.v_proj, had_dim=head_dim, output=True)
    #     apply_exact_had_to_linear(layer.cross_attn_token_to_image.out_proj, had_dim=-1, output=False)
    #     apply_exact_had_to_linear(layer.cross_attn_image_to_token.v_proj, had_dim=head_dim, output=True)
    #     apply_exact_had_to_linear(layer.cross_attn_image_to_token.out_proj, had_dim=-1, output=False)

def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation. 
    It reshapes X and applies Walsh-Hadamard transform to the last dimension. 
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from fast_hadamard_transform import hadamard_transform
    from hadamard_utils import get_had172
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1/math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input 
    return input.to(X.device).to(X.dtype).reshape(
        X.shape) 




def rotate_head_sam(model, Q: torch.Tensor,args):
    """
    Rotate the output heads for SAM model.
    SAM has multiple output heads: IoU prediction head, mask output heads, and HQ-specific heads.
    """
    
    # Rotate IoU prediction head (final layer)
    # From mask_decoder_hq.py: self.iou_prediction_head = MLP(...)
    iou_head = model.mask_decoder.iou_prediction_head
    W = iou_head.layers[-1]  # Final layer of IoU prediction MLP
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    
    # Rotate mask output heads (output_hypernetworks_mlps final layers)
    # From mask_decoder_hq.py: self.output_hypernetworks_mlps = nn.ModuleList([MLP(...) for i in range(self.num_mask_tokens)])
    for mlp in model.mask_decoder.output_hypernetworks_mlps:
        W = mlp.layers[-1]  # Final layer of each output hypernetwork MLP
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    
    # Rotate HQ-specific head (HQ-SAM specific)
    # From mask_decoder_hq.py: self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
    hf_mlp = model.mask_decoder.hf_mlp
    W = hf_mlp.layers[-1]  # Final layer of HQ MLP
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=args.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)




@torch.inference_mode()
def rotate_model(model, args):
   
    Q_image_encoder = get_orthogonal_matrix(args.hidden_size_image_en,args.rotate_mode,device = args.device)
    Q_mask_decoder = get_orthogonal_matrix(args.hidden_size_mask_de,args.rotate_mode,device = args.device)
    
    
    # rotate_embeddings(model, Q_image_encoder,Q_mask_decoder,args)
    # rotate_head_sam(model, Q_mask_decoder,args)
    utils.cleanup_memory()
    rotate_model_sam(model,Q_image_encoder,Q_mask_decoder,args)
    utils.cleanup_memory()


@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]

def register_online_rotation(module, Q:torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape
        

        if self.k_groupsize == -1: #token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else: #head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
        
        self.k_quantizer.free()
            
        return q, k



def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)