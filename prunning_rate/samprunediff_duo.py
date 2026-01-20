from segment_anything.modeling.image_encoder import (
    Attention as EncoderAttention,
    Block as EncoderBlock,
    ImageEncoderViT,
    window_partition,
    window_unpartition,
)
from .ddp import DiffPruneRate
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from segment_anything.modeling.mask_decoder_hq import MaskDecoderHQ as OriginalMaskDecoderHQ
from .samprune import MaskDecoderHQ
from typing import Any, Dict, List, Tuple
from segment_anything.modeling import Sam
import torch
import torch.nn as nn
from utils.quant_utils import quantize_activation_per_channel_absmax, quantize_activation_per_token_absmax


class DuoDiffPruneRateAttention(EncoderAttention):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
    def introduce_prune_diff(self,head_number,prune_granularity):
        self.prune_ddp = DiffPruneRate(head_number,prune_granularity)
    def set_processor(self, processor, module_name,args,train_rate_prune=False):
        self.processor = processor
        self.module_name = module_name 
        self.training = train_rate_prune
        self.threshold = args.train_prune_rate.threshold
        self.global_threshold = args.train_prune_rate.threshold_globle
        self.model_type = args.model.model_type
        self.prune_global = args.quantization.prune_global
        self.positional_quant = args.quantization.positional_quant
        self.use_percentage = args.quantization.use_percentage
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.training:
            B_2, H, W, _ = x.shape

            assert B_2 % 2 == 0
            B= B_2 // 2
            
            # qkv with shape (3, B, nHead, H * W, C)
            qkv = self.qkv(x).reshape(B_2, H * W, 3, self.num_heads, -1)
            
            qkv_non_prune, qkv_prune = qkv[:B], qkv[B:]

            

            with torch.no_grad():
                # q, k, v with shape (B * nHead, H * W, C)
                q, k, v = qkv_non_prune.permute(2, 0, 3, 1, 4).reshape(3, B * self.num_heads, H * W, -1).unbind(0)
                attn = (q * self.scale) @ k.transpose(-2, -1)
                if self.use_rel_pos:
                    attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

                attn = attn.softmax(dim=-1) 
                x = (attn @ v)
            
            q_prune, k_prune, v_prune = qkv_prune.permute(2, 0, 3, 1, 4).reshape(3, B * self.num_heads, H * W, -1).unbind(0)
            x_prune= v_prune.mean(-2, keepdim=True).expand(-1, x.shape[-2], x.shape[-1])

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

            with torch.no_grad():
                sorted_x = x[sorted_indicies_mul_images, :, :]
            sorted_x_prune = x_prune[sorted_indicies_mul_images, :, :]
            
            # Get trainable mask from DiffPruneRate
            batch_masks_probability = []
            for i in range(nu_images):
                single_mask_probablity = self.prune_ddp.get_head_probability_diff_duo()
                batch_masks_probability.append(single_mask_probablity)
            prune_ordered_probality_mask = torch.cat(batch_masks_probability, dim=0).view(-1, 1, 1)
            sorted_x_prune = (1-prune_ordered_probality_mask) * sorted_x_prune + prune_ordered_probality_mask * sorted_x

            
            inverse_indices = torch.argsort(torch.tensor(sorted_indicies_mul_images))

            x = sorted_x[inverse_indices, :, :]
            x_prune = sorted_x_prune[inverse_indices, :, :]
            
            with torch.no_grad():
                x=x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
            x_prune = x_prune.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

            x = torch.cat([x, x_prune], dim=0)
            x = self.proj(x)

            return x
        
        else:
            B, H, W, _ = x.shape
            
            # qkv with shape (3, B, nHead, H * W, C)
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            # q, k, v with shape (B * nHead, H * W, C)
            q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
            nu_images =1
            
            # prune_kept_num=  int(self.prune_ddp.update_kept_head_number() )
            if self.use_percentage:

                
                
                sorted_indices = self.processor.final_entropy_stats.get(self.module_name, None)
                prune_head_num = self.processor.prune_counts_per_layer.get(self.module_name, self.num_heads)
                kept_head_num = len(sorted_indices)- prune_head_num
                non_prune_mask = sorted_indices[:kept_head_num]
                prune_mask = sorted_indices[kept_head_num:]
                single_mask_probability = torch.zeros(self.num_heads) # Dummy for determining n_bits size check
            else:
                should_prune = True
                if not self.prune_global:
                    if self.model_type == "vit_b" and any(num in self.module_name for num in ["2", "5", "8", "11"]):
                        should_prune = False
                    elif self.model_type == "vit_l" and any(num in self.module_name for num in [".5", "11", "17", "23"]):
                        should_prune = False
                    elif self.model_type == "vit_h" and any(num in self.module_name for num in [".7", "15", "23", "31"]):
                        should_prune = False
                
                if should_prune:
                    single_mask_probability = self.prune_ddp.get_head_probability_diff_duo()

                    if self.model_type == "vit_b":
                        if not any(num in self.module_name for num in [".2", ".5", "8", "11"]):
                            mask = single_mask_probability > self.threshold
                        else:
                            mask = single_mask_probability > self.global_threshold

                    elif self.model_type == "vit_l":
                        if not any(num in self.module_name for num in [".5", "11", "17", "23"]):
                            mask = single_mask_probability > self.threshold
                        else:
                            mask = single_mask_probability > self.global_threshold

                    elif self.model_type == "vit_h":
                        if not any(num in self.module_name for num in [".7", "15", "23", "31"]):
                            mask = single_mask_probability > self.threshold
                        else:
                            mask = single_mask_probability > self.global_threshold

                    kept_head_num = mask.sum().item()
                    non_prune_mask = self.processor.final_entropy_stats.get(self.module_name, None)[:kept_head_num]
                    prune_mask = self.processor.final_entropy_stats.get(self.module_name, None)[kept_head_num:]
                
            
            if prune_mask is not None:
                if not self.positional_quant :
                    q_attn = q[non_prune_mask, :, :]
                    k_attn = k[non_prune_mask, :, :]
                    v_attn = v[non_prune_mask, :, :]
                    v_pruned = v[prune_mask, :, :]

                    attn = (q_attn * self.scale) @ k_attn.transpose(-2, -1)
                    if self.use_rel_pos:
                        attn = add_decomposed_rel_pos(attn, q_attn, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
                    
                    attn= attn.softmax(dim=-1)
                    x_attn = attn @ v_attn
                    x = torch.zeros_like(v).to(v.device)
                    x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
                    x[non_prune_mask] = x_attn
                else:
                    if self.model_type == "vit_b" :  
                        n_bits = 4 if not single_mask_probability.shape[0] == 300 else 2
                    elif  self.model_type == "vit_l" :
                        n_bits = 4 if not single_mask_probability.shape[0] == 400 else 2
                    elif  self.model_type == "vit_h" :
                        n_bits = 4 if not single_mask_probability.shape[0] == 400 else 2
                    q_attn = q[non_prune_mask, :, :]
                    k_attn = k[non_prune_mask, :, :]
                    v_attn = v[non_prune_mask, :, :]
                    q_prune = quantize_activation_per_token_absmax(q[prune_mask, :, :], n_bits)
                    k_prune = quantize_activation_per_token_absmax(k[prune_mask, :, :], n_bits)
                    v_prune = quantize_activation_per_channel_absmax(v[prune_mask, :, :], n_bits)

                    attn = (q_attn * self.scale) @ k_attn.transpose(-2, -1)
                    if self.use_rel_pos:
                        attn = add_decomposed_rel_pos(attn, q_attn, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
                    
                    attn= attn.softmax(dim=-1)
                    x_attn = attn @ v_attn
                    x = torch.zeros_like(v).to(v.device)
                    attn_prune = ((q_prune * self.scale) @ k_prune.transpose(-2, -1)).softmax(dim=-1)
                    x_prune = quantize_activation_per_token_absmax(attn_prune, n_bits) @ v_prune
                    x[prune_mask] = x_prune
                    x[non_prune_mask] = x_attn
            else:
                attn = (q * self.scale) @ k.transpose(-2, -1)
                if self.use_rel_pos:
                    attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
                attn = attn.softmax(dim=-1)
                x = attn @ v
        
            # Reshape output to original spatial dimensions
            
            x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
            x = self.proj(x)
      
            return x

class DuoDiffPruneRateSamDistillation(Sam):
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
        double_images= torch.cat([input_images, input_images], dim=0)
        image_embeddings, interm_embeddings = self.image_encoder(double_images)
        batched_input.extend(batched_input)
        # import ipdb; ipdb.set_trace()
        
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
        
        return mask_hq

def image_encoder_monkey_patch_train_duo_diff( model, processor=None,  device="cuda",  args_yaml=None, train = False):
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
            module.__class__ = DuoDiffPruneRateAttention
            module.set_processor(processor, name, args_yaml , train)
            if args_yaml.model.model_type == "vit_b":
                if not any(num in name for num in [".2", ".5", "8", "11"]):
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
        # if isinstance(module, (EncoderBlock)):
        #     module.__class__ = DiffPruneRateBlock
        # if isinstance(module, (ImageEncoderViT)):
        #     module.__class__ = DuoPruneRateImageEncoderViT
        #     module.set_training(train)
        if isinstance(module, Sam):
            if train:
                # module.__class__ = DuoPruneRateSam
                module.__class__ = DuoDiffPruneRateSamDistillation
        if isinstance(module, OriginalMaskDecoderHQ):
            if train:
                module.__class__ = MaskDecoderHQ
    # Now enable gradients only for selected_probability parameters
    if train:
        
        for name, param in model.named_parameters():
            if 'selected_probability' in name:
                param.requires_grad = True
                print(f"Enabled training for: {name}")
            else:
                param.requires_grad = False