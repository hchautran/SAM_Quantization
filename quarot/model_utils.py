
import torch



def get_rope_function_name(model):
    # SAM models don't use rotary position embeddings
    # They use standard attention with positional encodings added to inputs
    # Return None to indicate no RoPE function exists
    return None

def get_layers(model):
    return model.mask_decoder.transformer.layers
def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    # Collect modules to replace
    modules_to_replace = []

    for name, module in root.named_modules():
        
        # if ("mask_decoder" in name and "norm2" in name) or ("image_encoder" in name and "norm" in name):
        if "image_encoder" in name and "norm" in name :

            if isinstance(module, type_to_replace):
                modules_to_replace.append((name, module))

    # Perform replacements
    for name, module in modules_to_replace:
        new_module = None
        if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
            new_module = new_module_factory(module, int(name))
        else:  # layernorm_fusion.fuse_modules case where layernorms are fused
            new_module = new_module_factory(module)
        if new_module is not None:
            # Use `setattr` to replace the module
            parent_module, attr_name = _get_parent_module_and_attr(root, name)
            setattr(parent_module, attr_name, new_module)


def _get_parent_module_and_attr(root: torch.nn.Module, full_name: str):
    """Helper function to get the parent module and attribute name for a given module."""
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)
