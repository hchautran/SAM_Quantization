from segment_anything.modeling.image_encoder import (
    Attention as EncoderAttention,
    Block as EncoderBlock,
    ImageEncoderViT,
    window_partition,
    window_unpartition,
)
from segment_anything.modeling.mask_decoder_hq import MaskDecoderHQ as OriginalMaskDecoderHQ
from typing import Any, Dict, List, Tuple
from segment_anything.modeling import Sam
from .ddp import DiffPruneRate
import torch

class MaskDecoderHQ(OriginalMaskDecoderHQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified forward to match training behavior while allowing gradient flow.
        """
        # Enable gradient computation even though parameters are frozen
        with torch.set_grad_enabled(True):
     
            vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
            hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

            batch_len = len(image_embeddings)
            masks = []
            iou_preds = []
            for i_batch in range(batch_len):
                mask, iou_pred = self.predict_masks(
                    image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                    image_pe=image_pe[i_batch],
                    sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                    dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                    hq_feature = hq_features[i_batch].unsqueeze(0)
                )
                masks.append(mask)
                iou_preds.append(iou_pred)
            masks = torch.cat(masks,0)
            iou_preds = torch.cat(iou_preds,0)

            # Select the correct mask or masks for output
            if multimask_output:
                # mask with highest score
                mask_slice = slice(1,self.num_mask_tokens-1)
                iou_preds = iou_preds[:, mask_slice]
                iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
                iou_preds = iou_preds.unsqueeze(1)
                masks_multi = masks[:, mask_slice, :, :]
                masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)
            else:
                # singale mask output, default
                mask_slice = slice(0, 1)
                masks_sam = masks[:,mask_slice]

            masks_hq = masks[:,slice(self.num_mask_tokens-1, self.num_mask_tokens), :, :]
            
            if hq_token_only:
                return masks_hq
            else:
                return masks_sam, masks_hq
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
     
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape

        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4:] @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
class DiffPruneRateAttention(EncoderAttention):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
    def introduce_prune_diff(self,head_number,prune_granularity):
        self.prune_ddp = DiffPruneRate(head_number,prune_granularity)
    def set_processor(self, processor, module_name,train_rate_prune=False):
        self.processor = processor
        self.module_name = module_name 
        self.training = train_rate_prune
    def _calculate_qkv_flops(self, B, H, W):
        """Calculate FLOPs for QKV linear transformation."""
        # Input: (B, H*W, dim), Weight: (dim, 3*dim)
        # FLOPs = B * H * W * dim * 3 * dim = B * H * W * 3 * dim^2
        dim = self.qkv.in_features
        return B * H * W * 3 * dim * dim
    def _calculate_attention_flops(self, H, W, active_heads):
        """Calculate FLOPs for attention computation with pruned heads."""
        head_dim = self.qkv.in_features // self.num_heads
        seq_len = H * W
        
        # Q @ K^T: ( active_heads, seq_len, head_dim) @ ( active_heads, head_dim, seq_len)
        # FLOPs = active_heads * seq_len * seq_len * head_dim
        qk_flops =  active_heads * seq_len * seq_len * head_dim
        
        # Softmax: approximately seq_len operations per element
        # FLOPs = B * active_heads * seq_len * seq_len
    
        softmax_flops =  active_heads * seq_len * seq_len
        
        # Attn @ V: ( active_heads, seq_len, seq_len) @ (active_heads, seq_len, head_dim)
        # FLOPs = active_heads * seq_len * seq_len * head_dim
        attn_v_flops =  active_heads * seq_len * seq_len * head_dim
        
        return qk_flops + softmax_flops + attn_v_flops
    def _calculate_projection_flops(self, B, H, W):
        """Calculate FLOPs for final projection."""
        # Input: (B, H*W, dim), Weight: (dim, dim)
        # FLOPs = B * H * W * dim * dim
        dim = self.proj.in_features
        return B * H * W * dim * dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        if self.training:
            
            prune_kept_num = self.prune_ddp.update_kept_head_number() 
            # Get sorted head indices from processor
            sorted_indicies = self.processor.final_entropy_stats.get(self.module_name, None)            
            # Create reordering indices for B * num_heads dimension 
            nu_heads = k.shape[0]
            nuheads_per_image= len(sorted_indicies)
            nu_images = nu_heads // nuheads_per_image
       
            # calculate number of batch of only one image
            sorted_indicies_mul_images = []
            for i in range(nu_images):
                # Add offset for each image in the batch
                offset = i * nuheads_per_image
                batch_indices = [idx + offset for idx in sorted_indicies]
                sorted_indicies_mul_images.extend(batch_indices)
            
            # Reorder q, k, v according to importance scores
            q_reordered = q[sorted_indicies_mul_images, :, :]
            k_reordered = k[sorted_indicies_mul_images, :, :]
            v_reordered = v[sorted_indicies_mul_images, :, :]
            
            # Get trainable mask from DiffPruneRate
            batch_masks = []
            for i in range(nu_images):
                single_mask = self.prune_ddp.get_head_mask(nuheads_per_image).to(q.device)
                batch_masks.append(single_mask)
            prune_mask = torch.cat(batch_masks, dim=0)
            # Apply mask to q and k (element-wise multiplication for training)
            q_masked = q_reordered * prune_mask.unsqueeze(-1).unsqueeze(-1)
            k_masked = k_reordered * prune_mask.unsqueeze(-1).unsqueeze(-1)
            v_masked = v_reordered  # v doesn't need masking for attention computation
            
            # Compute attention normally
            attn = (q_masked * self.scale) @ k_masked.transpose(-2, -1)
            if self.use_rel_pos:
                from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
                attn = add_decomposed_rel_pos(attn, q_masked, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            attn = attn.softmax(dim=-1)
            
            # Apply attention to values
            x_reordered = attn @ v_masked
            
            # Create inverse reordering indices to restore original order
            inverse_indices = torch.argsort(torch.tensor(sorted_indicies_mul_images))

            x = x_reordered[inverse_indices, :, :]
            
        
        else:
            nu_images =1
            
            prune_kept_num=  int(self.prune_ddp.update_kept_head_number() )
           
            prune_mask_data = None
            should_prune = True
            if not self.processor.prune_global:
                if self.processor.model_type == "vit_b" and any(num in self.module_name for num in ["2", "5", "8", "11"]):
                    should_prune = False
                elif self.processor.model_type == "vit_l" and any(num in self.module_name for num in ["5", "11", "17", "23"]):
                    should_prune = False
                elif self.processor.model_type == "vit_h" and any(num in self.module_name for num in ["7", "15", "23", "31"]):
                    should_prune = False
            
            if should_prune:
                non_prune_mask = self.processor.final_entropy_stats.get(self.module_name, None)[:prune_kept_num]
                prune_mask = self.processor.final_entropy_stats.get(self.module_name, None)[prune_kept_num:]
                
             
            if prune_mask is not None:
                q_attn = q[non_prune_mask, :, :]
                k_attn = k[non_prune_mask, :, :]
                v_attn = v[non_prune_mask, :, :]
                v_pruned = v[prune_mask, :, :]

                attn = (q_attn * self.scale) @ k_attn.transpose(-2, -1)
                if self.use_rel_pos:
                    from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
                    attn = add_decomposed_rel_pos(attn, q_attn, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
                
                attn= attn.softmax(dim=-1)
                x_attn = attn @ v_attn
                x = torch.zeros_like(v).to(v.device)
                x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
                x[non_prune_mask] = x_attn
                
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                if self.use_rel_pos:
                    from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
                    attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
                attn = attn.softmax(dim=-1)
                x = attn @ v
        
        # Reshape output to original spatial dimensions
        
        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        
        qkvflops = self._calculate_qkv_flops(B // nu_images , H, W)
        projectflops = self._calculate_projection_flops(B // nu_images, H, W)
        attention_flops = self._calculate_attention_flops( H, W, prune_kept_num)
        # attention_flops = self._calculate_attention_flops( H, W, self.prune_ddp.head_number//2)
        total_flops = qkvflops + projectflops + attention_flops
        # print("number kept heads/ total heads:",prune_kept_num,"/",B*self.num_heads)
        # print("B* self.num_heads",B*self.num_heads)
        # print("B// nu_images:",B//nu_images)
        # print("prune_kept_num:",prune_kept_num)
        # print("qkv flops:",qkvflops)
        # print("projectflops flops:",projectflops)
        # print("attention_flops: ",attention_flops)
        return x, total_flops

class DiffPruneRateBlock(EncoderBlock):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Block class."""
        super().__init__(*args, **kwargs)
    def _calculate_mlp_flops(self, x: torch.Tensor) -> int:
        """Calculate FLOPs for the MLP block."""
        B, H, W, C = x.shape
        sequence_length = B * H * W
        
        # MLP has two linear layers: embedding_dim -> mlp_dim -> embedding_dim
        embedding_dim = self.mlp.lin1.in_features
        mlp_dim = self.mlp.lin1.out_features
        
        # FLOPs for lin1: embedding_dim * mlp_dim * sequence_length
        lin1_flops = embedding_dim * mlp_dim * sequence_length
        
        # FLOPs for lin2: mlp_dim * embedding_dim * sequence_length  
        lin2_flops = mlp_dim * embedding_dim * sequence_length
        
        total_flops = lin1_flops + lin2_flops
        return total_flops
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x, attn_flops = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        
        
        mlp_input = self.norm2(x)
        mlp_flops = self._calculate_mlp_flops(mlp_input)
        
        x = x + self.mlp(mlp_input)
        # flops = attn_flops + mlp_flops
        flops = attn_flops
        # print("mlp flops:",mlp_flops)
        return x , flops

class DiffPruneRateImageEncoderViT(ImageEncoderViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def set_training(self, mode: bool = True):
        self.training = mode
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        total_flops =0
        interm_embeddings=[]
        for blk in self.blocks:
            x ,flops= blk(x)
            if blk.window_size == 0:
                interm_embeddings.append(x)
            total_flops += flops
        x = self.neck(x.permute(0, 3, 1, 2))
        # print("total flops:", total_flops)
        # exit()
        if self.training:
            return x, interm_embeddings, total_flops
        else:
            return x, interm_embeddings

class DiffPruneRateSam(Sam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        hq_token_only: bool =False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        
        if self.image_encoder.training:
            image_embeddings, interm_embeddings, total_flops = self.image_encoder(input_images)
        else:
            image_embeddings, interm_embeddings = self.image_encoder(input_images)
            total_flops = None
            
       

        # Process each image individually since embeddings have different shapes
        all_sparse_embeddings = []
        all_dense_embeddings = []
        all_image_pe = []
        all_image_embeddings =[]
        for  image_record, curr_embedding in zip(batched_input,image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
                
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            img_pre = self.prompt_encoder.get_dense_pe()
         
            all_sparse_embeddings.append(sparse_embeddings)
            all_dense_embeddings.append(dense_embeddings)
            all_image_pe.append(img_pre)
            all_image_embeddings.append(curr_embedding.unsqueeze(0))
           
        image_embeddings= torch.cat(all_image_embeddings)  
        # Call mask decoder for single image
        mask_hq = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=all_image_pe,
            sparse_prompt_embeddings=all_sparse_embeddings,
            dense_prompt_embeddings=all_dense_embeddings,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=interm_embeddings,
        )
       
        return mask_hq, total_flops

class DiffPruneRateSamInference(Sam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
        hq_token_only: bool =False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings, interm_embeddings = self.image_encoder(input_images)
        interm_embeddings = interm_embeddings[0] # early layer

        outputs = []
        for image_record, curr_embedding, curr_interm in zip(batched_input, image_embeddings, interm_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                hq_token_only=hq_token_only,
                interm_embeddings=curr_interm.unsqueeze(0).unsqueeze(0),
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs
def image_encoder_monkey_patch_train( model, processor=None,  device="cuda",  args_yaml=None, train = False):
    """
    Apply monkey-patching to SAM image encoder for quantization and observation.

    Args:
        model: SAM model to patch
        processor: Processing strategy for activations
        n_bits: Number of bits for quantization
        weight_quant: Weight quantization strategy
        k_preserve: Number of channels to preserve in selective quantization
    """
    # First, freeze all original SAM parameters
   
        # print(f"Frozen: {name}")
    for name, module in model.named_modules():
        if isinstance(module, (EncoderAttention)):
            module.__class__ = DiffPruneRateAttention
            module.set_processor(processor, name, train)
            if args_yaml.model.model_type == "vit_b":
                if not any(num in name for num in ["2", ".5", "8", "11"]):
                    module.introduce_prune_diff(300,1)
                else:
                    module.introduce_prune_diff(12,1)
            elif args_yaml.model.model_type =="vit_l":
                if not any(num in name for num in [".5", "11", "17", "23"]):
                    module.introduce_prune_diff(400,1)
                else:
                    module.introduce_prune_diff(16,1)
            elif args_yaml.model.model_type == "vit_h":
                if not any(num in name for num in [".7", "15", "23", "31"]):
                    module.introduce_prune_diff(400,1)
                else:
                    module.introduce_prune_diff(16,1)
        if isinstance(module, (EncoderBlock)):
            module.__class__ = DiffPruneRateBlock
        if isinstance(module, (ImageEncoderViT)):
            module.__class__ = DiffPruneRateImageEncoderViT
            module.set_training(train)
        if isinstance(module, Sam):
            if train:
                module.__class__ = DiffPruneRateSam
            
        if isinstance(module, OriginalMaskDecoderHQ):
            if train:
                module.__class__ = MaskDecoderHQ
    # Now enable gradients only for selected_probability parameters
    if train:
        
        for name, param in model.named_parameters():
            if 'selected_probability' in name:
                param.requires_grad = True
                print(f"Enabled training for: {name}")
    
    