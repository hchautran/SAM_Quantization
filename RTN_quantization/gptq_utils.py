import math
import time
import tqdm
import torch
import torch.nn as nn
import logging
import numpy as np
import ipdb

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

def find_qlayers(module, layers=[torch.nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 4:
            B, H, W, C = inp.shape
            inp = inp.reshape(-1, C)  # Flatten spatial dimensions
            
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        cleanup_memory(verbos=False)
        
        
@torch.no_grad()
def gptq_fwrd_sam(model, dataloader, dev, args):

    logging.info('-----GPTQ Quantization-----')
    
    image_encoder = model.image_encoder
    blocks = image_encoder.blocks

    image_encoder.patch_embed = image_encoder.patch_embed.to(dev)
    blocks[0] = blocks[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    sample_batch = next(iter(dataloader))
    _, sample_image, _, _, _ = sample_batch['imidx'], sample_batch['image'], sample_batch['label'], sample_batch['shape'], sample_batch['ori_label']
    
    sample_img = sample_image[0].permute(1, 2, 0).cpu().numpy() # shape (H, W, C)
    sample_tensor = torch.as_tensor(sample_img.astype(dtype=np.uint8), device=dev).permute(2, 0, 1).contiguous() # shape (C, H, W)
    model.to(dev)
    preprocessed_image = model.preprocess(sample_tensor).unsqueeze(0).to(dev)

    with torch.no_grad():
        x = image_encoder.patch_embed(preprocessed_image)
        feature_shape = x.shape
    
    inps = torch.zeros(
        (args.nsamples, feature_shape[1], feature_shape[2], feature_shape[3]), 
        dtype=dtype, device=dev
    )
    # store the input as format : nu_image , H , W ,C ( as each image is )  nuimage * 64 * 64 * 1024
    # ipdb.set_trace()
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, x):
            if cache['i'] < args.nsamples:
                inps[cache['i']] = x
                cache['i'] += 1
            raise ValueError
        
    blocks[0] = Catcher(blocks[0])
    
    sample_count = 0
    for batch in dataloader:
        try:
            if sample_count >= args.nsamples:
                break

            _, image_batch, _, _, _ = batch['imidx'], batch['image'], batch['label'], batch['shape'], batch['ori_label']

            sample_img = image_batch[0].permute(1, 2, 0).cpu().numpy() # shape (H, W, C)
            sample_tensor = torch.as_tensor(sample_img.astype(dtype=np.uint8), device=dev).permute(2, 0, 1).contiguous() # shape (C, H, W)
            
            preprocessed_image = model.preprocess(sample_tensor).unsqueeze(0).to(dev)
            x = image_encoder.patch_embed(preprocessed_image).squeeze(0)
            if model.image_encoder.pos_embed is not None:
                x = x + model.image_encoder.pos_embed
            blocks[0](x)
            sample_count += 1
        except ValueError:
            pass
    blocks[0] = blocks[0].module

    # Move initial layers back to CPU
    blocks[0] = blocks[0].cpu()
    image_encoder.patch_embed = image_encoder.patch_embed.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    quantizers = {}
    
    sequential = [  
        ['attn.qkv'],           # Attention QKV projection
        ['attn.proj'],          # Attention output projection  
        ['mlp.lin1'],           # MLP first layer
        ['mlp.lin2'] ,           # MLP second layer
        ['attn.qkv.module'],           # Attention QKV projection quarot
        ['attn.proj.module'],          # Attention output projection  quarot
        ['mlp.lin1.module'],           # MLP first layer quarot
        ['mlp.lin2.module']             # MLP second layer quarot
    ]
    
    for i in range(len(blocks)):
        print(f'\nBlock {i}:', flush=True, end=' ')
        block = blocks[i].to(dev)
        full = find_qlayers(block, layers=[torch.nn.Linear])
        
        for names in sequential:
            # Filter out names that don't exist in this block
            subset = {n: full[n] for n in names if n in full}
                  
            if not subset:  # Skip if no matching layers found
                continue

            gptq = {}
          
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
                )
            
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            for j in range(min(args.nsamples, cache['i'])):
                inp_tensor = inps[j].unsqueeze(0)  
                out_tensor = block(inp_tensor)
                
                # Handle the output properly 
                if isinstance(out_tensor, tuple):
                    outs[j] = out_tensor[0].squeeze(0)
                else:
                    outs[j] = out_tensor.squeeze(0)
            
            
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, 
                    actorder=args.act_order, static_groups=False
                )
                quantizers[f'image_encoder.blocks.{i}.{name}'] = gptq[name].quantizer
                gptq[name].free()

        # Forward pass for next block
        for j in range(min(args.nsamples, cache['i'])):
            inp_tensor = inps[j].unsqueeze(0)  # Add batch dimension
            out_tensor = block(inp_tensor)
            
            # Handle output properly
            if isinstance(out_tensor, tuple):
                outs[j] = out_tensor[0].squeeze(0)
            else:
                outs[j] = out_tensor.squeeze(0)

        blocks[i] = block.cpu()
        del block
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization for SAM Image Encoder Done-----\n')
    return quantizers

@torch.no_grad()
def gptq_fwrd_sam_maskdecoder(module,
                    image_embeddings,
                    image_pe,
                    sparse_prompt_embeddings,
                    dense_prompt_embeddings,
                    multimask_output=False,
                    hq_token_only=False,
                    interm_embeddings=None,
                    dev=None,
                    args=None
                ):
    
    logging.info('-----GPTQ Quantization for Mask Decoder-----')
    
    module = module.to(dev)
    dtype = next(iter(module.parameters())).dtype
    
    # Prepare inputs for decoder
    batch_len = len(image_embeddings)
    
    # Store inputs for GPTQ calibration
    inps = []
    cache = {'i': 0}
    
    class Catcher(nn.Module):
        def __init__(self, original_module):
            super().__init__()
            self.module = original_module
        
        def forward(self, *forward_args, **kwargs):
            # Fix: use args from outer scope, not forward arguments
            if cache['i'] < args.nsamples:
                # Store the inputs for this forward pass
                inps.append({
                    'args': forward_args,
                    'kwargs': kwargs
                })
                cache['i'] += 1
            raise ValueError
    
    # Replace the transformer with catcher to collect inputs
    original_transformer = module.transformer
    module.transformer = Catcher(original_transformer)
    
    # Collect calibration data by running forward passes
    sample_count = 0
    for i in range(min(args.nsamples, batch_len)):
        try:
            # Prepare batch inputs for this sample
            batch_image_embeddings = image_embeddings[i:i+1]
            batch_image_pe = [image_pe[i]] if isinstance(image_pe, list) else image_pe[i:i+1]
            batch_sparse_embeddings = [sparse_prompt_embeddings[i]] if isinstance(sparse_prompt_embeddings, list) else sparse_prompt_embeddings[i:i+1]
            batch_dense_embeddings = [dense_prompt_embeddings[i]] if isinstance(dense_prompt_embeddings, list) else dense_prompt_embeddings[i:i+1]
            batch_interm_embeddings = [interm_embeddings[i]] if isinstance(interm_embeddings, list) else interm_embeddings[i:i+1]
            
            # Run forward to collect inputs
            module(
                image_embeddings=batch_image_embeddings,
                image_pe=batch_image_pe,
                sparse_prompt_embeddings=batch_sparse_embeddings,
                dense_prompt_embeddings=batch_dense_embeddings,
                multimask_output=multimask_output,
                hq_token_only=hq_token_only,
                interm_embeddings=batch_interm_embeddings,
            )
            sample_count += 1
        except ValueError:
            pass
    
    # Restore original transformer
    module.transformer = original_transformer
    
    # Define layers to quantize in the decoder
    sequential = [
        ['transformer.layers.0.self_attn.q_proj'],
        ['transformer.layers.0.self_attn.k_proj'], 
        ['transformer.layers.0.self_attn.v_proj'],
        ['transformer.layers.0.self_attn.out_proj'],
        ['transformer.layers.0.mlp.lin1'],
        ['transformer.layers.0.mlp.lin2'],
        ['transformer.layers.0.cross_attn_token_to_image.q_proj'],
        ['transformer.layers.0.cross_attn_token_to_image.k_proj'],
        ['transformer.layers.0.cross_attn_token_to_image.v_proj'],
        ['transformer.layers.0.cross_attn_token_to_image.out_proj'],
        ['transformer.layers.0.cross_attn_image_to_token.q_proj'],
        ['transformer.layers.0.cross_attn_image_to_token.k_proj'],
        ['transformer.layers.0.cross_attn_image_to_token.v_proj'],
        ['transformer.layers.0.cross_attn_image_to_token.out_proj'],
        ['transformer.layers.1.self_attn.q_proj'],
        ['transformer.layers.1.self_attn.k_proj'],
        ['transformer.layers.1.self_attn.v_proj'],
        ['transformer.layers.1.self_attn.out_proj'],
        ['transformer.layers.1.mlp.lin1'],
        ['transformer.layers.1.mlp.lin2'],
        ['transformer.layers.1.cross_attn_token_to_image.q_proj'],
        ['transformer.layers.1.cross_attn_token_to_image.k_proj'],
        ['transformer.layers.1.cross_attn_token_to_image.v_proj'],
        ['transformer.layers.1.cross_attn_token_to_image.out_proj'],
        ['transformer.layers.1.cross_attn_image_to_token.q_proj'],
        ['transformer.layers.1.cross_attn_image_to_token.k_proj'],
        ['transformer.layers.1.cross_attn_image_to_token.v_proj'],
        ['transformer.layers.1.cross_attn_image_to_token.out_proj'],
        ['output_hypernetworks_mlps.0.layers.0'],
        ['output_hypernetworks_mlps.0.layers.1'],
        ['output_hypernetworks_mlps.0.layers.2'],
        ['output_hypernetworks_mlps.1.layers.0'],
        ['output_hypernetworks_mlps.1.layers.1'],
        ['output_hypernetworks_mlps.1.layers.2'],
        ['output_hypernetworks_mlps.2.layers.0'],
        ['output_hypernetworks_mlps.2.layers.1'],
        ['output_hypernetworks_mlps.2.layers.2'],
        ['output_hypernetworks_mlps.3.layers.0'],
        ['output_hypernetworks_mlps.3.layers.1'],
        ['output_hypernetworks_mlps.3.layers.2'],
        ['iou_prediction_head.layers.0'],
        ['iou_prediction_head.layers.1'],
        ['iou_prediction_head.layers.2'],
        
        ['transformer.layers.0.self_attn.q_proj.module'],
        ['transformer.layers.0.self_attn.k_proj.module'], 
        ['transformer.layers.0.self_attn.v_proj.module'],
        ['transformer.layers.0.self_attn.out_proj.module'],
        ['transformer.layers.0.mlp.lin1.module'],
        ['transformer.layers.0.mlp.lin2.module'],
        ['transformer.layers.0.cross_attn_token_to_image.q_proj.module'],
        ['transformer.layers.0.cross_attn_token_to_image.k_proj.module'],
        ['transformer.layers.0.cross_attn_token_to_image.v_proj.module'],
        ['transformer.layers.0.cross_attn_token_to_image.out_proj.module'],
        ['transformer.layers.0.cross_attn_image_to_token.q_proj.module'],
        ['transformer.layers.0.cross_attn_image_to_token.k_proj.module'],
        ['transformer.layers.0.cross_attn_image_to_token.v_proj.module'],
        ['transformer.layers.0.cross_attn_image_to_token.out_proj.module'],
        ['transformer.layers.1.self_attn.q_proj.module'],
        ['transformer.layers.1.self_attn.k_proj.module'],
        ['transformer.layers.1.self_attn.v_proj.module'],
        ['transformer.layers.1.self_attn.out_proj.module'],
        ['transformer.layers.1.mlp.lin1.module'],
        ['transformer.layers.1.mlp.lin2.module'],
        ['transformer.layers.1.cross_attn_token_to_image.q_proj.module'],
        ['transformer.layers.1.cross_attn_token_to_image.k_proj.module'],
        ['transformer.layers.1.cross_attn_token_to_image.v_proj.module'],
        ['transformer.layers.1.cross_attn_token_to_image.out_proj.module'],
        ['transformer.layers.1.cross_attn_image_to_token.q_proj.module'],
        ['transformer.layers.1.cross_attn_image_to_token.k_proj.module'],
        ['transformer.layers.1.cross_attn_image_to_token.v_proj.module'],
        ['transformer.layers.1.cross_attn_image_to_token.out_proj.module'],
        ['output_hypernetworks_mlps.0.layers.0.module'],
        ['output_hypernetworks_mlps.0.layers.1.module'],
        ['output_hypernetworks_mlps.0.layers.2.module'],
        ['output_hypernetworks_mlps.1.layers.0.module'],
        ['output_hypernetworks_mlps.1.layers.1.module'],
        ['output_hypernetworks_mlps.1.layers.2.module'],
        ['output_hypernetworks_mlps.2.layers.0.module'],
        ['output_hypernetworks_mlps.2.layers.1.module'],
        ['output_hypernetworks_mlps.2.layers.2.module'],
        ['output_hypernetworks_mlps.3.layers.0.module'],
        ['output_hypernetworks_mlps.3.layers.1.module'],
        ['output_hypernetworks_mlps.3.layers.2.module'],
        ['iou_prediction_head.layers.0.module'],
        ['iou_prediction_head.layers.1.module'],
        ['iou_prediction_head.layers.2.module'],
        
    ]
    
    quantizers = {}
    
    # Find all linear layers in the decoder
    full = find_qlayers(module, layers=[torch.nn.Linear])
    
    for names in sequential:
        # Filter out names that don't exist in this module
        subset = {n: full[n] for n in names if n in full}
        
        if not subset:  # Skip if no matching layers found
            continue
            
        print(f'Quantizing: {names}', flush=True)
        
        gptq = {}
        for name in subset:
            layer_weight_bits = args.w_bits
            layer_weight_sym = not(args.w_asym)
            
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = WeightQuantizer()
            gptq[name].quantizer.configure(
                layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip
            )
        
        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        # Run forward passes with collected inputs
        for i in range(min(len(inps), args.nsamples)):
            batch_image_embeddings = image_embeddings[i:i+1]
            batch_image_pe = [image_pe[i]] if isinstance(image_pe, list) else image_pe[i:i+1]
            batch_sparse_embeddings = [sparse_prompt_embeddings[i]] if isinstance(sparse_prompt_embeddings, list) else sparse_prompt_embeddings[i:i+1]
            batch_dense_embeddings = [dense_prompt_embeddings[i]] if isinstance(dense_prompt_embeddings, list) else dense_prompt_embeddings[i:i+1]
            batch_interm_embeddings = [interm_embeddings[i]] if isinstance(interm_embeddings, list) else interm_embeddings[i:i+1]
            
            with torch.no_grad():
                module(
                    image_embeddings=batch_image_embeddings,
                    image_pe=batch_image_pe,
                    sparse_prompt_embeddings=batch_sparse_embeddings,
                    dense_prompt_embeddings=batch_dense_embeddings,
                    multimask_output=multimask_output,
                    hq_token_only=hq_token_only,
                    interm_embeddings=batch_interm_embeddings,
                )
        
        for h in handles:
            h.remove()
        
        # Quantize the layers
        for name in subset:
            layer_w_groupsize = args.w_groupsize
            gptq[name].fasterquant(
                percdamp=args.percdamp, groupsize=layer_w_groupsize, 
                actorder=args.act_order, static_groups=False
            )
            quantizers[f'mask_decoder.{name}'] = gptq[name].quantizer
            gptq[name].free()
        
        del gptq
        torch.cuda.empty_cache()
    
    cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization for Mask Decoder Done-----\n')
    return quantizers