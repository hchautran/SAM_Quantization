'''
Differentiable Discrte Proxy 
'''

import torch.nn as nn
import torch



class STE_Ceil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x = torch.ceil(x_in)
        return x
    
    @staticmethod
    def backward(ctx, g):
        return g, None
    
    
ste_ceil = STE_Ceil.apply


class DiffPruneRate(nn.Module):
    def __init__(self, head_number=400, granularity=1) -> None:
        '''
        head_number: the origianl input patch head of each block, it is same for each block for standard ViT
        class_head: weather there is a class head
        granularity: the granularity of searched compression rate, 1 means the gap between each candidate is 1 head
        '''
        super().__init__()
        self.head_number = head_number

        
        # for more clean code, we directly set the candidate as kept head number, which can perform same as compression rate
        # at least one head should be kept
        self.kept_head_candidate =  nn.Parameter(torch.arange(head_number, 0,-1*granularity).float())
        self.kept_head_candidate.requires_grad_(False)
        self.selected_probability =  nn.Parameter(torch.zeros_like(self.kept_head_candidate))   
        # self.selected_probability.data[0] = 10.0  # initialize to keep all heads
        self.selected_probability.requires_grad_(True)
        
        # the learn target, which can be directly applied to the off-the-shlef pre-trained models
        self.kept_head_number = self.head_number 
        
        self.update_kept_head_number()
    
    
    def update_kept_head_number(self):
        self.selected_probability_softmax = self.selected_probability.softmax(dim=-1)
        # which will be used to calculate FLOPs, leveraging STE in Ceil to keep gradient backpropagation
        kept_head_number = ste_ceil(torch.matmul(self.kept_head_candidate,self.selected_probability_softmax))
        self.kept_head_number = int(kept_head_number)
        return kept_head_number
        
    def get_head_probability(self):
        head_probability =  torch.zeros((self.head_number), device=self.selected_probability_softmax.device) 
        for kept_head_number, prob in zip(self.kept_head_candidate, self.selected_probability_softmax):
            head_probability[: int(kept_head_number)] += prob
        return head_probability
    
    def get_head_mask(self, head_number=None):
        # self.update_kept_head_number()
        head_probability = self.get_head_probability()
        
        # translate probability to 0/1 mask
        head_mask = torch.ones_like(head_probability)
        if head_number is not None:    # only set the compressed head  in this operation as 0, which can keep gradient backward
            head_mask[int(self.kept_head_number):int(head_number)] = 0     
        else:
            head_mask[int(self.kept_head_number):] = 0
        head_mask = head_mask - head_probability.detach() + head_probability   # ste trick, similar to gumbel softmax
        return head_mask
    