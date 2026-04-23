import torch
import torch.nn as nn
import os
import psutil
import warnings
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
warnings.filterwarnings("ignore")

# Check for available fusion libraries
APEX_AVAILABLE = False
XFORMERS_AVAILABLE = False
TORCH_COMPILE_AVAILABLE = False

try:
    import apex
    from apex.normalization import FusedLayerNorm
    APEX_AVAILABLE = True
except ImportError:
    pass

try:
    import xformers
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    pass

# Check PyTorch version for torch.compile
if hasattr(torch, 'compile'):
    TORCH_COMPILE_AVAILABLE = True

# ===============================
# Device & Memory Utility
# ===============================
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for better performance on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    device = torch.device('cpu')
    torch.set_num_threads(min(8, os.cpu_count()))
    print("CUDA not available, using CPU")

print(f"\n🔧 Kernel Fusion Support:")
print(f"  - APEX (Fused LayerNorm): {APEX_AVAILABLE}")
print(f"  - xFormers (Fused Attention): {XFORMERS_AVAILABLE}")
print(f"  - torch.compile (PyTorch 2.0+): {TORCH_COMPILE_AVAILABLE}")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024 / 1024
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024
        return {
            'cpu_memory_mb': cpu_memory,
            'gpu_memory_mb': gpu_memory,
            'gpu_memory_cached_mb': gpu_memory_cached
        }
    return {'cpu_memory_mb': cpu_memory}

def shrink_model_architecture(base_config, shrink_config):
    """
    Shrink model architecture based on shrink_config parameters.
    
    Args:
        base_config: Original LlamaConfig from pretrained model
        shrink_config: Dict with shrinking parameters
            - num_layers: Number of layers to keep (default: keep all)
            - num_heads: Number of attention heads (default: keep all)
            - hidden_size: Hidden dimension size (default: keep original)
            - intermediate_size: MLP intermediate size (default: keep original)
    
    Returns:
        Modified LlamaConfig
    """
    config = LlamaConfig.from_dict(base_config.to_dict())
    
    # Shrink layers
    if shrink_config.get('num_layers') is not None:
        config.num_hidden_layers = shrink_config['num_layers']
    
    # Shrink attention heads (must be divisor of hidden_size)
    if shrink_config.get('num_heads') is not None:
        # Ensure head_dim remains valid
        head_dim = config.hidden_size // config.num_attention_heads
        new_num_heads = shrink_config['num_heads']
        
        # Adjust num_key_value_heads proportionally
        if hasattr(config, 'num_key_value_heads'):
            ratio = config.num_key_value_heads / config.num_attention_heads
            new_kv_heads = max(1, int(new_num_heads * ratio))
            config.num_key_value_heads = new_kv_heads
        
        config.num_attention_heads = new_num_heads
    
    # Shrink hidden size (must be multiple of num_heads)
    if shrink_config.get('hidden_size') is not None:
        new_hidden_size = shrink_config['hidden_size']
        # Ensure it's divisible by num_attention_heads
        new_hidden_size = (new_hidden_size // config.num_attention_heads) * config.num_attention_heads
        config.hidden_size = new_hidden_size
        
        # Update head_dim
        head_dim = config.hidden_size // config.num_attention_heads
    
    # Shrink intermediate size (MLP)
    if shrink_config.get('intermediate_size') is not None:
        config.intermediate_size = shrink_config['intermediate_size']
    elif shrink_config.get('hidden_size') is not None:
        # Auto-adjust intermediate size proportionally
        original_ratio = base_config.intermediate_size / base_config.hidden_size
        config.intermediate_size = int(config.hidden_size * original_ratio)
        # Round to nearest multiple of 256 for efficiency
        config.intermediate_size = ((config.intermediate_size + 127) // 256) * 256
    
    return config

def copy_weights_with_shrinking(source_model, target_model, shrink_config):
    """
    Copy weights from source model to shrunken target model.
    Only copies the layers/dimensions that exist in the target model.
    """
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    
    num_layers_target = shrink_config.get('num_layers', len([k for k in target_dict.keys() if 'layers.0.' in k]))
    
    for key in target_dict.keys():
        if key in source_dict:
            source_param = source_dict[key]
            target_param = target_dict[key]
            
            # Handle layer-specific weights
            if 'layers.' in key:
                layer_num = int(key.split('layers.')[1].split('.')[0])
                if layer_num >= num_layers_target:
                    continue  # Skip layers beyond target
            
            # Copy with dimension matching
            if source_param.shape == target_param.shape:
                target_dict[key] = source_param.clone()
            else:
                # Truncate dimensions to match target
                if len(source_param.shape) == 2:  # Weight matrices
                    target_dict[key] = source_param[:target_param.shape[0], :target_param.shape[1]].clone()
                elif len(source_param.shape) == 1:  # Biases/norms
                    target_dict[key] = source_param[:target_param.shape[0]].clone()
                else:
                    target_dict[key] = source_param  # Keep as is for other shapes
    
    target_model.load_state_dict(target_dict)
    return target_model

def apply_kernel_fusion_optimizations(model, fusion_mode='torch_compile'):
    """
    Apply REAL kernel fusion optimizations to the model.
    
    Args:
        model: The LlamaForCausalLM model
        fusion_mode: 'torch_compile', 'apex', 'xformers', or 'all'
    
    Returns:
        Optimized model with fused kernels
    """
    print(f"\n⚡ Applying Kernel Fusion: {fusion_mode}")
    
    if fusion_mode == 'torch_compile' or fusion_mode == 'all':
        if TORCH_COMPILE_AVAILABLE and device.type == 'cuda':
            print("  ✓ Compiling model with torch.compile (fuses kernels automatically)")
            # torch.compile with max-autotune mode provides aggressive kernel fusion
            model = torch.compile(
                model,
                mode='max-autotune',  # Aggressive optimization
                fullgraph=False,  # Allow partial graphs
                dynamic=False  # Static shapes for better fusion
            )
            print("    → Fused attention, LayerNorm+Linear, and other ops")
        else:
            print("  ✗ torch.compile not available (requires PyTorch 2.0+ and CUDA)")
    
    if fusion_mode == 'apex' or fusion_mode == 'all':
        if APEX_AVAILABLE and device.type == 'cuda':
            print("  ✓ Replacing LayerNorm with Fused LayerNorm (APEX)")
            # Replace all LayerNorm modules with FusedLayerNorm
            def replace_layernorm(module):
                for name, child in module.named_children():
                    if isinstance(child, nn.LayerNorm):
                        fused_ln = FusedLayerNorm(
                            child.normalized_shape,
                            eps=child.eps,
                            elementwise_affine=child.elementwise_affine
                        )
                        # Copy weights
                        if child.elementwise_affine:
                            fused_ln.weight.data.copy_(child.weight.data)
                            fused_ln.bias.data.copy_(child.bias.data)
                        setattr(module, name, fused_ln)
                    else:
                        replace_layernorm(child)
            
            if hasattr(model, 'model'):
                replace_layernorm(model.model)
            else:
                replace_layernorm(model)
            print("    → Fused all LayerNorm operations")
        else:
            print("  ✗ APEX not available (install with: pip install apex)")
    
    if fusion_mode == 'xformers' or fusion_mode == 'all':
        if XFORMERS_AVAILABLE and device.type == 'cuda':
            print("  ✓ xFormers available for memory-efficient attention")
            print("    → Will use fused attention kernels during forward pass")
            # Note: xFormers integration typically requires modifying attention implementation
            # For HuggingFace models, this is often done via config settings
            if hasattr(model, 'model') and hasattr(model.model, 'config'):
                # Enable if the model supports it
                if hasattr(model.model.config, '_attn_implementation'):
                    model.model.config._attn_implementation = 'flash_attention_2'
                    print("    → Enabled Flash Attention 2")
        else:
            print("  ✗ xFormers not available (install with: pip install xformers)")
    
    return model

# ===============================
# HuggingFace-backed Transformer
# ===============================
class LlamaCompatibleTransformer(nn.Module):
    _shared_instance = None  # singleton

    def __new__(cls, *args, **kwargs):
        if cls._shared_instance is None:
            cls._shared_instance = super().__new__(cls)
        return cls._shared_instance

    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct",
                 shrink_config=None,
                 enable_kernel_fusion=False,
                 fusion_mode='torch_compile',
                 **hf_kwargs):
        super().__init__()

        # Only initialize once, but allow fusion config to run if requested
        if not hasattr(self, "_initialized"):
            self._initialized = False

        # First-time setup
        if not self._initialized:
            print(f"Loading HuggingFace model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            base_config = LlamaConfig.from_pretrained(model_name)

            if shrink_config is not None and any(v is not None for v in shrink_config.values()):
                print(f"Applying model shrinking: {shrink_config}")
                target_config = shrink_model_architecture(base_config, shrink_config)

                source_model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="cpu",
                    **hf_kwargs
                )
                
                self.model = LlamaForCausalLM(target_config)
                self.model = copy_weights_with_shrinking(source_model, self.model, shrink_config)
                del source_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = self.model.to(
                    device=device,
                    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
                )
            else:
                self.model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    **hf_kwargs
                )

            self._initialized = True

        # Apply kernel fusion if requested, **even if _initialized was True**
        if enable_kernel_fusion:
            self.model = apply_kernel_fusion_optimizations(self.model, fusion_mode)
            self.kernel_fusion_enabled = True
        else:
            self.kernel_fusion_enabled = False

        self.model.train()
        self.to(device)

        print(f"Model loaded on {device}: {model_name}")
        if enable_kernel_fusion:
            print(f"⚡ Kernel Fusion: ENABLED ({fusion_mode})")
        print(f"Memory usage: {get_memory_usage()}")


    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass (compatible with fine-tuning)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.logits if labels is None else outputs

    def generate_text(self, prompt, **gen_kwargs):
        """Convenience wrapper for inference/generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ===============================
# Debug Utility
# ===============================
def print_tensor_shapes(state_dict, stage):
    print(f"Tensors at {stage}:")
    for k, v in state_dict.items():
        print(f"{k}: {tuple(v.shape)}")