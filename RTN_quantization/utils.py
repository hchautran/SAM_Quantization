import torch
import torch.nn as nn
from per_tensor_channel_group import W8A8Linear

def replace_linear_with_target_and_quantize(module, 
                               target_class, module_name_to_exclude, 
                               weight_quant="per_channel", act_quant="per_token", 
                               quantize_output=False, group_size=None):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            
            # Use from_float method instead of manual creation
            new_module = target_class.from_float(
                child, 
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                quantize_output=quantize_output,
                group_size=group_size
            )
            setattr(module, name, new_module) # replace the module with the new module
            
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, 
                     target_class, module_name_to_exclude, 
                     weight_quant, act_quant, quantize_output, group_size)
            
### example usage
class DummyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = torch.nn.Embedding(1, 1)
    # Try with bias
    self.linear_1 = nn.Linear(1, 1)
    # Try without bias
    self.linear_2 = nn.Linear(1, 1, bias=False)
    # Lm prediction head
    self.lm_head = nn.Linear(1, 1, bias=False)

  def forward(self, x):
    x = self.emb(x)
    x = self.linear_1(x)
    x = self.linear_2(x)
    x = self.lm_head(x)
    return x

if __name__ == "__main__":
    model = DummyModel()  # Fixed: Added parentheses to instantiate
    print("Before quantization:")
    print(model)
    print("\nLinear layers before:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: {type(module)}")
    
    replace_linear_with_target_and_quantize(
        model, 
        W8A8Linear, 
        module_name_to_exclude=["emb", "lm_head"],
        weight_quant="per_channel",
        act_quant="per_token"
    )
    
    print("\nAfter quantization:")
    print(model)
    print("\nLinear layers after:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, W8A8Linear)):
            print(f"  {name}: {type(module)}")
            
    # Test forward pass
    test_input = torch.randint(0, 1, (1,))
    output = model(test_input)
    print(f"\nTest forward pass successful: {output.shape}")