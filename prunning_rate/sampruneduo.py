from segment_anything.modeling.image_encoder import (
    Attention as EncoderAttention,
    Block as EncoderBlock,
    ImageEncoderViT,
    window_partition,
    window_unpartition,
)
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from segment_anything.modeling.mask_decoder_hq import MaskDecoderHQ as OriginalMaskDecoderHQ
from .samprune import MaskDecoderHQ
from typing import Any, Dict, List, Tuple
from segment_anything.modeling import Sam
import torch
import torch.nn as nn



class DuoPruneRateAttention(EncoderAttention):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
    def introduce_full_attention_heads(self,head_number, train_rate_prune= False,initial_value=1.0, device='cpu', dtype=torch.float32):
        self.full_attention_heads = nn.Parameter(
                torch.ones(
                    head_number,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                * initial_value
            )
    
    def set_processor(self, processor, module_name, args,train_rate_prune=False):
        # self.processor = processor
        self.module_name = module_name 
        self.training = train_rate_prune
        self.batch_size = args.train_prune_rate.batch_size_train
        self.threshold = args.train_prune_rate.threshold  # You can set this to any desired value
        self.global_threshold = args.train_prune_rate.threshold_globle
        self.model_type = args.model.model_type
    
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

            attention_weights_multiple_images = torch.cat([self.full_attention_heads for _ in range(self.batch_size)])
            
            attention_weights = (
                attention_weights_multiple_images.clamp(0, 1)
                .view(-1, 1, 1)  # Shape: (total_nu_heads, 1, 1)
            )
            
            x_prune = (1 - attention_weights) * x_prune + attention_weights * x

            with torch.no_grad():
                x=x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
            x_prune = x_prune.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

            x = torch.cat([x, x_prune], dim=0)
            x = self.proj(x)

            return x
        else:
            # Inference mode - single batch, no duplication
            B, H, W, _ = x.shape
            
            # Compute QKV for all heads
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
            
            # Determine which heads to prune based on threshold
            if hasattr(self, 'full_attention_heads') and self.full_attention_heads is not None:
                # Create pruning mask: True for heads to prune (below threshold)
                head_weights = self.full_attention_heads.clamp(0, 1)

                if self.model_type == "vit_b":
                    if not any(num in self.module_name for num in [".2", ".5", "8", "11"]):
                        prune_mask = head_weights < self.threshold
                    else:
                        prune_mask = head_weights < self.global_threshold
                elif self.model_type =="vit_l":
                    if not any(num in self.module_name for num in [".5", "11", "17", "23"]):
                        prune_mask = head_weights < self.threshold
                    else:
                        prune_mask = head_weights < self.global_threshold
                elif self.model_type == "vit_h":
                    if not any(num in self.module_name for num in [".7", "15", "23", "31"]):
                        prune_mask = head_weights < self.threshold
                    else:
                        prune_mask = head_weights < self.global_threshold
                
                
                # Expand mask to match batch dimension
                prune_mask = prune_mask.repeat(q.shape[0]// len(head_weights))
                
                # Split heads into pruned and non-pruned
                
                q_attn = q[~prune_mask, :, :]
                k_attn = k[~prune_mask, :, :]
                v_attn = v[~prune_mask, :, :]
                v_pruned = v[prune_mask, :, :]
                
                # print(q_attn.shape, k_attn.shape, v_attn.shape, v_pruned.shape )
                
            else:
                # No pruning - process all heads normally
                q_attn, k_attn, v_attn = q, k, v
                prune_mask = None
            
            # Compute attention for non-pruned heads
            attn = (q_attn * self.scale) @ k_attn.transpose(-2, -1)
            if self.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q_attn, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            attn = attn.softmax(dim=-1)
            x_attn = attn @ v_attn
            
            # Merge outputs
            if prune_mask is not None:
                x = torch.zeros_like(v).to(v.device)
                # For pruned heads: use mean of value vectors
                x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
                # For non-pruned heads: use computed attention
                x[~prune_mask] = x_attn
            else:
                x = x_attn
            
            # Reshape and project
            x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
            x = self.proj(x)
            
            return x
            

class DuoPruneRateSam(Sam):
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
        
        if self.image_encoder.blocks[0].attn.training:
            nu_images = input_images.shape[0]
            image_embeddings = image_embeddings[nu_images:]  # Take only pruned images
            for i in range(len(interm_embeddings)):
                interm_embeddings[i] = interm_embeddings[i][nu_images:]

            ##############################
            # # these lines forcus on right after the loss function

            # nu_images = input_images.shape[0]
            # teacher_image_embeddings = image_embeddings[:nu_images]  # Take only pruned images
            # prune_image_embeddings = image_embeddings[nu_images:]  # Take only non-pruned images

            # teacher_interm_embeddings = []
            # prune_interm_embeddings = []
            # for i in range(len(interm_embeddings)):
            #     teacher_interm_embeddings.append(interm_embeddings[i][:nu_images])  
            #     prune_interm_embeddings.append(interm_embeddings[i][nu_images:])
            # return teacher_image_embeddings, teacher_interm_embeddings, prune_image_embeddings, prune_interm_embeddings


        # else:

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
        return mask_hq

class DuoPruneRateSamDistillation(Sam):
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

def image_encoder_monkey_patch_train_duo( model, processor=None,  device="cuda",  args_yaml=None, train = False):
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
            module.__class__ = DuoPruneRateAttention
            module.set_processor(processor, name, args_yaml , train)
            if args_yaml.model.model_type == "vit_b":
                if not any(num in name for num in [".2", ".5", "8", "11"]):
                    module.introduce_full_attention_heads(300,1)
                else:
                    module.introduce_full_attention_heads(12,1)
            elif args_yaml.model.model_type =="vit_l":
                if not any(num in name for num in [".5", "11", "17", "23"]):
                    module.introduce_full_attention_heads(400,1)
                else:
                    module.introduce_full_attention_heads(16,1)
            elif args_yaml.model.model_type == "vit_h":
                if not any(num in name for num in [".7", "15", "23", "31"]):
                    module.introduce_full_attention_heads(400,1)
                else:
                    module.introduce_full_attention_heads(16,1)
        # if isinstance(module, (EncoderBlock)):
        #     module.__class__ = DiffPruneRateBlock
        # if isinstance(module, (ImageEncoderViT)):
        #     module.__class__ = DuoPruneRateImageEncoderViT
        #     module.set_training(train)
        if isinstance(module, Sam):
            if train:
                # module.__class__ = DuoPruneRateSam
                module.__class__ = DuoPruneRateSamDistillation
        if isinstance(module, OriginalMaskDecoderHQ):
            if train:
                module.__class__ = MaskDecoderHQ
    # Now enable gradients only for selected_probability parameters
    if train:
        
        for name, param in model.named_parameters():
            if 'full_attention_heads' in name:
                param.requires_grad = True
                print(f"Enabled training for: {name}")
            else:
                param.requires_grad = False
    
    