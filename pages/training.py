# pages/training.py - Enhanced with 8-bit optimizers, model shrinking, and dual training modes

import streamlit as st
import torch
import torch.optim as optim
import time
import gc
import os
from model import (
    LlamaCompatibleTransformer, 
    get_memory_usage, 
    device
)
from performance_enchancer import perform_garbage_collection, cuda_memory_stats
from dataset_loader import FastDatasetIterator
from training_loop import run_training_loop
from standard_training_loop import run_standard_training_loop
from training_preload import load_tokenizer

# Try to import bitsandbytes for 8-bit optimizers
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    bnb = None

def display_device_info():
    if device.type == 'cuda':
        st.sidebar.success(f"🚀 Using CUDA: {torch.cuda.get_device_name()}")
        memory_stats = cuda_memory_stats()
        if memory_stats:
            st.sidebar.info(f"GPU Memory: {memory_stats['allocated']:.1f}GB / {memory_stats['reserved']:.1f}GB")
    else:
        st.sidebar.info("💻 Using CPU")

def optimize_cuda_settings():
    if device.type == 'cuda':
        # Enable memory optimization features
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        
        # Set CUDA memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_ui_configuration():
    st.sidebar.header("Configuration")
    display_device_info()
    
    # Display 8-bit optimizer availability
    if BITSANDBYTES_AVAILABLE:
        st.sidebar.success("✅ 8-bit optimizers available")
    else:
        st.sidebar.warning("⚠️ bitsandbytes not installed - using standard optimizers")
        st.sidebar.info("Install with: pip install bitsandbytes")
    
    # ===== NEW: Training Mode Selection =====
    st.sidebar.subheader("🎯 Training Mode")
    training_mode = st.sidebar.radio(
        "Select Training Mode",
        ["Standard LM Training", "Contrastive Learning"],
        index=0,
        help="Standard: Next-token prediction | Contrastive: Learn from positive/negative examples"
    )
    
    use_standard_training = (training_mode == "Standard LM Training")
    
    # Display mode-specific info
    if use_standard_training:
        st.sidebar.info("📖 Standard language modeling: trains to predict next tokens")
    else:
        st.sidebar.info("🔄 Contrastive learning: learns from positive vs negative examples")
    
    # ===== Model Shrinking Configuration =====
    st.sidebar.subheader("🔧 Model Architecture")
    
    enable_shrinking = st.sidebar.checkbox("Enable Model Shrinking", value=False,
                                          help="Reduce model size by removing layers/heads/dimensions")
    
    shrink_config = {}
    if enable_shrinking:
        with st.sidebar.expander("Model Shrinking Settings", expanded=True):
            st.write("**Reduce model architecture:**")
            
            # Base model selection
            base_model = st.selectbox(
                "Base Model",
                ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"],
                index=0,
                help="Start from this model and shrink it"
            )
            
            # Layer reduction
            use_custom_layers = st.checkbox("Custom Layer Count", value=False)
            if use_custom_layers:
                num_layers = st.number_input(
                    "Number of Layers",
                    min_value=1,
                    max_value=32,
                    value=8,
                    help="Fewer layers = faster training, less capacity"
                )
                shrink_config['num_layers'] = num_layers
            
            # Attention head reduction
            use_custom_heads = st.checkbox("Custom Attention Heads", value=False)
            if use_custom_heads:
                num_heads = st.number_input(
                    "Number of Attention Heads",
                    min_value=1,
                    max_value=32,
                    value=8,
                    help="Must divide hidden size evenly"
                )
                shrink_config['num_heads'] = num_heads
            
            # Hidden size reduction
            use_custom_hidden = st.checkbox("Custom Hidden Size", value=False)
            if use_custom_hidden:
                hidden_size = st.number_input(
                    "Hidden Size",
                    min_value=128,
                    max_value=4096,
                    value=512,
                    step=128,
                    help="Must be divisible by number of heads"
                )
                shrink_config['hidden_size'] = hidden_size
            
            # Intermediate size (MLP)
            use_custom_intermediate = st.checkbox("Custom MLP Size", value=False)
            if use_custom_intermediate:
                intermediate_size = st.number_input(
                    "MLP Intermediate Size",
                    min_value=256,
                    max_value=16384,
                    value=1024,
                    step=256,
                    help="Typically 2-4x hidden size"
                )
                shrink_config['intermediate_size'] = intermediate_size
            
            if shrink_config:
                st.info(f"📊 Shrinking enabled with {len(shrink_config)} modifications")
        
        # Store base model in config
        shrink_config['_base_model'] = base_model
    
    st.sidebar.subheader("Training Settings")
    num_epochs = st.sidebar.number_input("Epochs", value=10, min_value=1, max_value=1000)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001, 0.00001], index=2, 
                                       help="Lower learning rates prevent NaN losses")
    batch_accumulation = st.sidebar.number_input("Gradient Accumulation", value=4, min_value=1, max_value=8)
    max_grad_norm = st.sidebar.number_input("Max Gradient Norm", value=0.25, min_value=0.1, max_value=2.0, step=0.1,
                                           help="Gradient clipping to prevent exploding gradients")
    
    # Standard training specific settings
    label_smoothing = 0.1
    if use_standard_training:
        label_smoothing = st.sidebar.slider(
            "Label Smoothing", 
            min_value=0.0, 
            max_value=0.3, 
            value=0.1, 
            step=0.01,
            help="Smoothing for cross-entropy loss (prevents overconfidence)"
        )
    
    # ===== Breakdown Settings =====
    st.sidebar.subheader("🎯 Context/Response Breakdown")
    enable_breakdown = st.sidebar.checkbox(
        "Enable Breakdown", 
        value=False,
        help="Only learn the response part after a delimiter, not the input context"
    )
    
    breakdown_delimiter = ""
    if enable_breakdown:
        breakdown_delimiter = st.sidebar.text_input(
            "Breakdown Delimiter",
            value="<|start_header_id|>reply<|end_header_id|>\n",
            help="Everything after the LAST occurrence of this delimiter will be learned"
        )
        st.sidebar.info("💡 Model will read all tokens but only learn to generate tokens after the delimiter")
        st.sidebar.caption("Example: Use '<|start_header_id|>reply<|end_header_id|>\\n' to only learn the reply portion")
    
    # ===== Parameter Freezing Settings =====
    st.sidebar.subheader("❄️ Parameter Freezing")
    st.sidebar.caption("Freeze parameter categories after N epochs to speed up training")
    
    enable_attention_freeze = st.sidebar.checkbox(
        "Freeze Attention",
        value=False,
        help="Freeze attention parameters (Q, K, V, O projections)"
    )
    
    freeze_attention_after_epoch = 0
    if enable_attention_freeze:
        freeze_attention_after_epoch = st.sidebar.number_input(
            "Freeze Attention After Epoch",
            min_value=1,
            max_value=1000,
            value=5,
            help="After this many epochs, attention weights will be frozen"
        )
    
    enable_embedding_freeze = st.sidebar.checkbox(
        "Freeze Embeddings",
        value=False,
        help="Freeze embedding layers (token and position embeddings)"
    )
    
    freeze_embedding_after_epoch = 0
    if enable_embedding_freeze:
        freeze_embedding_after_epoch = st.sidebar.number_input(
            "Freeze Embeddings After Epoch",
            min_value=1,
            max_value=1000,
            value=3,
            help="After this many epochs, embedding layers will be frozen"
        )
    
    enable_mlp_freeze = st.sidebar.checkbox(
        "Freeze MLP/FFN",
        value=False,
        help="Freeze MLP/feed-forward network layers"
    )
    
    freeze_mlp_after_epoch = 0
    if enable_mlp_freeze:
        freeze_mlp_after_epoch = st.sidebar.number_input(
            "Freeze MLP After Epoch",
            min_value=1,
            max_value=1000,
            value=10,
            help="After this many epochs, MLP layers will be frozen"
        )
    
    enable_lmhead_freeze = st.sidebar.checkbox(
        "Freeze LM Head",
        value=False,
        help="Freeze output/LM head layer (usually not recommended)"
    )
    
    freeze_lmhead_after_epoch = 0
    if enable_lmhead_freeze:
        freeze_lmhead_after_epoch = st.sidebar.number_input(
            "Freeze LM Head After Epoch",
            min_value=1,
            max_value=1000,
            value=15,
            help="After this many epochs, LM head will be frozen"
        )
        st.sidebar.warning("⚠️ Freezing LM head may impact output quality")
    
    # Optimization options for freezing
    st.sidebar.subheader("🚀 Freezing Optimizations")
    
    use_static_graph = False
    if enable_embedding_freeze:
        use_static_graph = st.sidebar.checkbox(
            "Static Graph (Embeddings)",
            value=False,
            help="Use static computation graph for frozen embeddings (PyTorch 2.0+ optimization)"
        )
    
    use_attention_cache = False
    if enable_attention_freeze:
        use_attention_cache = st.sidebar.checkbox(
            "Attention Cache",
            value=False,
            help="Cache attention computations for frozen attention (faster but uses more memory)"
        )
    
    # Show info if any freezing is enabled
    if enable_attention_freeze or enable_embedding_freeze or enable_mlp_freeze or enable_lmhead_freeze:
        st.sidebar.info("💡 Freezing parameters can speed up training significantly after initial learning")
        st.sidebar.caption("Parameter statistics will be shown during training")
        if use_static_graph:
            st.sidebar.caption("⚡ Static graph enabled - PyTorch will optimize frozen embedding lookups")
        if use_attention_cache:
            st.sidebar.caption("⚡ Attention cache enabled - will use more memory but speed up frozen attention")
    
    st.sidebar.subheader("Optimizer Settings")
    
    # 8-bit optimizer options
    optimizer_type = st.sidebar.selectbox(
        "Optimizer Type",
        options=["AdamW", "AdamW8bit", "Adam8bit", "Lion8bit"] if BITSANDBYTES_AVAILABLE else ["AdamW"],
        index=1 if BITSANDBYTES_AVAILABLE else 0,
        help="8-bit optimizers reduce memory usage by ~50% with minimal accuracy loss"
    )
    
    # Advanced optimizer settings
    with st.sidebar.expander("Advanced Optimizer Settings"):
        weight_decay = st.number_input("Weight Decay", value=0.01, min_value=0.0, max_value=0.1, step=0.001,
                                     help="L2 regularization strength")
        beta1 = st.number_input("Beta1 (momentum)", value=0.9, min_value=0.1, max_value=0.999, step=0.001)
        beta2 = st.number_input("Beta2 (RMSprop)", value=0.999, min_value=0.9, max_value=0.9999, step=0.0001)
        eps = st.number_input("Epsilon", value=1e-8, min_value=1e-10, max_value=1e-6, format="%.2e",
                            help="Small constant for numerical stability")
        
        if optimizer_type in ["AdamW8bit", "Adam8bit"]:
            # 8-bit specific settings
            st.write("**8-bit Optimizer Settings:**")
            block_wise = st.checkbox("Block-wise quantization", value=True,
                                   help="More accurate but slightly slower")
            percentile_clipping = st.number_input("Percentile Clipping", value=100, min_value=90, max_value=100,
                                                help="Outlier clipping percentile (100 = no clipping)")
            optim_bits = st.selectbox("Optimizer bits", [32, 8], index=1,
                                    help="8-bit saves memory, 32-bit for full precision")
    
    st.sidebar.subheader("Memory Management")
    gc_frequency = st.sidebar.number_input("Garbage Collection Every N Epochs", value=5, min_value=1, max_value=20)
    memory_chunk_size = st.sidebar.number_input("Memory Chunk Size", value=5000, min_value=10, max_value=20000,
                                               help="Process dataset in chunks of this size to reduce memory usage")
    
    if device.type == 'cuda':
        sync_cuda = st.sidebar.checkbox("Sync CUDA", value=False)
        enable_mixed_precision = st.sidebar.checkbox("Mixed Precision (AMP)", value=True, 
                                                    help="Reduces memory usage and can speed up training")
    else:
        sync_cuda = False
        enable_mixed_precision = False
    
    st.sidebar.subheader("Batches Management")
    batch_size = st.sidebar.number_input("Batching", value=5000, min_value=1, max_value=20000)

    st.sidebar.subheader("Fusion Management")
    kernel_fusion = st.sidebar.checkbox("Enable Kernel Fusion", value=False,
                                          help="Reuse kernels to speed up cache usage")
    fusion_mode = st.sidebar.selectbox(
        "Fusion Mode",
        options=["apex"],
        index=0,
        help="Enable apex fusion mode"
    )

    config = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_accumulation': batch_accumulation,
        'max_grad_norm': max_grad_norm,
        'gc_frequency': gc_frequency,
        'sync_cuda': sync_cuda,
        'memory_chunk_size': memory_chunk_size,
        'batch_size': batch_size,
        'enable_mixed_precision': enable_mixed_precision,
        'optimizer_type': optimizer_type,
        'weight_decay': weight_decay,
        'beta1': beta1,
        'beta2': beta2,
        'eps': eps,
        'shrink_config': shrink_config if enable_shrinking else None,
        'use_standard_training': use_standard_training,
        'label_smoothing': label_smoothing,
        'kernel_fusion': kernel_fusion,
        'enable_kernel_fusion': kernel_fusion,
        'fusion_mode': fusion_mode,
        'enable_breakdown': enable_breakdown,
        'breakdown_delimiter': breakdown_delimiter,
        'enable_attention_freeze': enable_attention_freeze,
        'freeze_attention_after_epoch': freeze_attention_after_epoch,
        'enable_embedding_freeze': enable_embedding_freeze,
        'freeze_embedding_after_epoch': freeze_embedding_after_epoch,
        'enable_mlp_freeze': enable_mlp_freeze,
        'freeze_mlp_after_epoch': freeze_mlp_after_epoch,
        'enable_lmhead_freeze': enable_lmhead_freeze,
        'freeze_lmhead_after_epoch': freeze_lmhead_after_epoch,
        'use_static_graph': use_static_graph,
        'use_attention_cache': use_attention_cache
    }
    
    # Add 8-bit specific settings if applicable
    if optimizer_type in ["AdamW8bit", "Adam8bit"]:
        config.update({
            'block_wise': block_wise,
            'percentile_clipping': percentile_clipping,
            'optim_bits': optim_bits
        })
    
    return config

def create_optimizer(model, config):
    """Create optimizer based on configuration"""
    optimizer_type = config['optimizer_type']
    params = model.parameters()
    lr = config['learning_rate']
    weight_decay = config.get('weight_decay', 0.01)
    betas = (config.get('beta1', 0.9), config.get('beta2', 0.999))
    eps = config.get('eps', 1e-8)
    
    if optimizer_type == "AdamW":
        # Standard PyTorch AdamW
        optimizer = optim.AdamW(
            params, 
            lr=lr, 
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        st.info("✅ Using standard AdamW optimizer")
        
    elif optimizer_type == "AdamW8bit" and BITSANDBYTES_AVAILABLE:
        # 8-bit AdamW from bitsandbytes
        optimizer = bnb.optim.AdamW8bit(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            block_wise=config.get('block_wise', True),
            percentile_clipping=config.get('percentile_clipping', 100),
            optim_bits=config.get('optim_bits', 8)
        )
        st.info("✅ Using 8-bit AdamW optimizer (50% memory reduction)")
        
    elif optimizer_type == "Adam8bit" and BITSANDBYTES_AVAILABLE:
        # 8-bit Adam from bitsandbytes
        optimizer = bnb.optim.Adam8bit(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            block_wise=config.get('block_wise', True),
            percentile_clipping=config.get('percentile_clipping', 100),
            optim_bits=config.get('optim_bits', 8)
        )
        st.info("✅ Using 8-bit Adam optimizer (50% memory reduction)")
        
    elif optimizer_type == "Lion8bit" and BITSANDBYTES_AVAILABLE:
        # 8-bit Lion optimizer (newer, more memory efficient)
        optimizer = bnb.optim.Lion8bit(
            params,
            lr=lr * 0.3,  # Lion typically needs lower LR
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.99)),
            weight_decay=weight_decay
        )
        st.info("✅ Using 8-bit Lion optimizer (even more memory efficient)")
        
    else:
        # Fallback to standard AdamW if 8-bit not available
        optimizer = optim.AdamW(
            params, 
            lr=lr, 
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        st.warning(f"⚠️ {optimizer_type} not available, using standard AdamW")
    
    return optimizer

def create_model_and_optimizer(config):
    # Clear any existing CUDA cache before loading model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Get shrinking configuration
    shrink_config = config.get('shrink_config')
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    
    if shrink_config:
        base_model = shrink_config.pop('_base_model', base_model)
        st.info(f"🔧 Creating shrunken model from {base_model}")
        model = LlamaCompatibleTransformer(model_name=base_model, shrink_config=shrink_config)
    else:
        model = LlamaCompatibleTransformer(model_name=base_model)
    
    # Additional memory cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # Create optimizer with 8-bit support
    optimizer = create_optimizer(model, config)
    
    # Setup mixed precision scaler if enabled
    scaler = None
    if config.get('enable_mixed_precision', False) and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        st.info("✅ Mixed precision training enabled")
    
    return model, optimizer, scaler

def display_optimizer_memory_savings():
    """Display estimated memory savings from 8-bit optimizers"""
    if BITSANDBYTES_AVAILABLE and device.type == 'cuda':
        with st.expander("💡 8-bit Optimizer Benefits"):
            st.write("**Memory Savings:**")
            st.write("- AdamW8bit: ~50% optimizer memory reduction")
            st.write("- Adam8bit: ~50% optimizer memory reduction") 
            st.write("- Lion8bit: ~60% optimizer memory reduction")
            st.write("")
            st.write("**Performance:**")
            st.write("- Minimal accuracy loss (<1%)")
            st.write("- Slightly slower per step (~5-10%)")
            st.write("- Enables training larger models")
            st.write("- Reduces CUDA OOM errors")

def display_shrinking_info():
    """Display information about model shrinking"""
    with st.expander("🔧 Model Shrinking Benefits"):
        st.write("**Why Shrink Models?**")
        st.write("- Faster training and inference")
        st.write("- Much lower memory usage")
        st.write("- Still effective for fine-tuning")
        st.write("- Perfect for custom use cases")
        st.write("")
        st.write("**Recommended Shrinking Strategies:**")
        st.write("- **Tiny Model**: 6 layers, 8 heads, 512 hidden")
        st.write("- **Small Model**: 12 layers, 12 heads, 768 hidden")
        st.write("- **Medium Model**: 16 layers, 16 heads, 1024 hidden")
        st.write("")
        st.write("**Note**: Llama is just the skeleton - shrinking won't hurt pre-trained knowledge since you're training from scratch!")

def setup_training_ui():
    st.subheader("Training Progress")
    main_progress_bar = st.progress(0)
    epoch_progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart_placeholder = st.empty()
    memory_info = st.empty()
    
    if device.type == 'cuda':
        initial_gpu_stats = cuda_memory_stats()
        st.info(f"Initial GPU Memory: {initial_gpu_stats['allocated']:.1f}GB allocated, {initial_gpu_stats['reserved']:.1f}GB reserved")
    
    return {
        'main_progress_bar': main_progress_bar,
        'epoch_progress_bar': epoch_progress_bar,
        'status_text': status_text,
        'loss_chart_placeholder': loss_chart_placeholder,
        'memory_info': memory_info
    }

def prepare_training_data(config):
    """Prepare training data indices based on training mode"""
    indices = torch.randperm(len(st.session_state.dataset_iterator))
    valid_indices = []
    
    if config['use_standard_training']:
        # Standard training: use all examples (negatives are ignored in the loss function)
        valid_indices = indices.tolist()
    else:
        # Contrastive training: only use examples with negatives
        for idx in indices:
            item = st.session_state.dataset_iterator[idx.item()]
            if len(item['negatives']) > 0:
                valid_indices.append(idx.item())
    
    return valid_indices

def display_training_results(start_time, initial_memory, config):
    final_collected = perform_garbage_collection()
    total_time = time.time() - start_time
    final_memory = get_memory_usage()
    
    mode_info = f" | Mode: {'Standard LM' if config['use_standard_training'] else 'Contrastive'}"
    optimizer_info = f" | Optimizer: {config['optimizer_type']}"
    
    if config.get('shrink_config'):
        shrink_info = f" | Shrunken: {config['shrink_config']}"
    else:
        shrink_info = ""
    
    if device.type == 'cuda':
        final_gpu_stats = cuda_memory_stats()
        st.success(f"✅ Training completed! "
                  f"Time: {total_time:.1f}s | "
                  f"CPU: {initial_memory['cpu_memory_mb']:.1f}MB → {final_memory['cpu_memory_mb']:.1f}MB | "
                  f"GPU: {final_gpu_stats['allocated']:.1f}GB used | "
                  f"Final cleanup: {final_collected} objects"
                  f"{mode_info}{optimizer_info}{shrink_info}")
    else:
        st.success(f"✅ Training completed! "
                  f"Time: {total_time:.1f}s | "
                  f"Memory: {initial_memory['cpu_memory_mb']:.1f}MB → {final_memory['cpu_memory_mb']:.1f}MB | "
                  f"Final cleanup: {final_collected} objects"
                  f"{mode_info}{optimizer_info}{shrink_info}")

def main():
    st.header("Training")
    
    config = setup_ui_configuration()
    
    if st.session_state.dataset is None:
        st.warning("Please upload a dataset on the Dataset page before training")
        return
    
    # Display training mode information
    if config['use_standard_training']:
        st.info("📖 **Standard Language Modeling**: Training with next-token prediction loss. Negative examples in dataset will be ignored.")
    else:
        st.info("🔄 **Contrastive Learning**: Training with positive and negative examples. Only samples with negatives will be used.")
    
    # Display memory optimization tips
    with st.expander("💡 Memory Optimization Tips"):
        st.write("**If you encounter CUDA OOM errors:**")
        st.write("- **Enable Model Shrinking** (most effective!)")
        st.write("- Use 8-bit optimizers (AdamW8bit/Lion8bit)")
        st.write("- Reduce Memory Chunk Size (try 25-50)")
        st.write("- Increase Gradient Accumulation (try 8)")
        st.write("- Enable Mixed Precision")
        st.write("- Reduce GC frequency (try 3-5)")
        
        if BITSANDBYTES_AVAILABLE:
            st.write("")
            st.write("**8-bit Optimizer Recommendations:**")
            st.write("- AdamW8bit: Best general purpose")
            st.write("- Lion8bit: Most memory efficient")
            st.write("- Enable block-wise quantization for accuracy")
    
    # Display optimizer and shrinking info
    display_optimizer_memory_savings()
    display_shrinking_info()
    
    # Display training mode comparison
    with st.expander("🎯 Training Mode Comparison"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Standard LM Training**")
            st.write("✅ Uses all examples")
            st.write("✅ Simple next-token prediction")
            st.write("✅ Traditional approach")
            st.write("✅ Works with any text data")
            st.write("⚠️ No contrastive learning")
        
        with col2:
            st.write("**Contrastive Learning**")
            st.write("✅ Learns from comparisons")
            st.write("✅ Uses positive/negative pairs")
            st.write("✅ Can learn preferences")
            st.write("⚠️ Requires negative examples")
            st.write("⚠️ More complex loss")
    
    if st.button("Start Training", type="primary"):
        try:
            optimize_cuda_settings()
            start_total_time = time.time()
            initial_memory = get_memory_usage()
            
            tokenizer = load_tokenizer()
            model, optimizer, scaler = create_model_and_optimizer(config)
            
            st.session_state.model = model
            st.session_state.dataset_iterator = FastDatasetIterator(
                st.session_state.dataset, 
                tokenizer,
                breakdown_delimiter=config.get('breakdown_delimiter', ''),
                enable_breakdown=config.get('enable_breakdown', False)
            )
            
            ui_components = setup_training_ui()
            
            valid_indices = prepare_training_data(config)
            
            training_mode_str = "Standard LM" if config['use_standard_training'] else "Contrastive"
            st.info(f"Starting {training_mode_str} training with {len(valid_indices)} samples in chunks of {config['memory_chunk_size']}")
            
            # Display optimizer memory footprint estimate
            if config['optimizer_type'] in ['AdamW8bit', 'Adam8bit', 'Lion8bit']:
                estimated_savings = "~50%" if config['optimizer_type'] != 'Lion8bit' else "~60%"
                st.info(f"Using {config['optimizer_type']} - estimated optimizer memory savings: {estimated_savings}")
            
            # Run appropriate training loop based on mode
            if config['use_standard_training']:
                losses = run_standard_training_loop(
                    model, optimizer, config, ui_components, valid_indices
                )
            else:
                losses, pos_losses, neg_losses = run_training_loop(
                    model, optimizer, config, ui_components, valid_indices
                )
            
            ui_components['main_progress_bar'].progress(1.0)
            ui_components['epoch_progress_bar'].progress(1.0)
            
            display_training_results(start_total_time, initial_memory, config)
            
            st.session_state.training_complete = True
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                st.error("❌ CUDA Out of Memory! Try enabling Model Shrinking or using 8-bit optimizers.")
                if not BITSANDBYTES_AVAILABLE:
                    st.info("💡 Install bitsandbytes for 8-bit optimizers: pip install bitsandbytes")
                else:
                    st.info("💡 Best solution: Enable Model Shrinking with 6-12 layers, or use AdamW8bit optimizer with Memory Chunk Size 25.")
            else:
                st.error(f"❌ Training failed: {str(e)}")
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                perform_garbage_collection()
        
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()