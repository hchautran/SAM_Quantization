from decoder_quant import (
    mask_decoder_monkey_patch, 
    inference_image,
    re_cal_attn,
    TwoWayTransformerObserver,
    get_activation_boxplot
) 

from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt

model_type = 'vit_l'
checkpoint_path= '/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/pretrained_checkpoint/sam_hq_vit_l.pth'
sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
mask_decoder_monkey_patch(sam)
predictor = SamPredictor(sam)
results = inference_image(predictor, image_dir='./input_imgs/', example_idx=3, show_image=True)
import torch.nn.functional as F
import torch
k1 = TwoWayTransformerObserver.attention_score['p2i_k'][1]
print(k1.shape)
exit()
q = TwoWayTransformerObserver.attention_score['p2i_q'][1].permute(0,2,1,3).reshape(1,-1,128 )
k = TwoWayTransformerObserver.attention_score['p2i_k'][1].permute(0,2,1,3).reshape(1,-1,128 )
print("k.shape,", k.shape)
print("q.shape,", q.shape)
exit()
fig, ax= plt.subplots(4,1, figsize=(20,10), sharey=True)
head =0


get_activation_boxplot(high_activations=q, low_activations=k, ax=ax[0], offset=(16*head), max_channels=128,token_wise=True,pertoken=False)
get_activation_boxplot(high_activations=q, low_activations=k, ax=ax[1], offset=(16*head), max_channels=10,token_wise=True,pertoken=True)
# get_activation_boxplot(high_activations=q, low_activations=torch.abs(k), ax=ax[1], offset=(16*head), max_channels=16, token_wise=False)
get_activation_boxplot(high_activations=q, low_activations=torch.abs(k), ax=ax[2], offset=(16*head), max_channels=128, token_wise=True,pertoken= False)
get_activation_boxplot(high_activations=q, low_activations=torch.abs(k), ax=ax[3], offset=(16*head), max_channels=10, token_wise=True,pertoken= True)