# training_loop.py - Enhanced with proper batching support

import torch
import time
import streamlit as st
from contrastive_loss import optimized_contrastive_loss
from performance_enchancer import perform_garbage_collection, cuda_memory_stats
from model import get_memory_usage, device


def apply_static_graph_to_embeddings(model):
    """
    Apply REAL static graph compilation to frozen embedding layers using torch.compile().
    This compiles the forward pass for frozen embeddings while still allowing gradients to flow through.
    """
    st.info("⚡ Applying static graph compilation to embeddings...")
    
    if not hasattr(torch, 'compile'):
        st.warning("⚠️ torch.compile() not available (requires PyTorch 2.0+)")
        st.info("Embeddings are frozen but not compiled")
        return 0
    
    compiled_count = 0
    
    # Find embedding modules and compile them
    for name, module in model.named_modules():
        if any(emb in name.lower() for emb in ['embed_tokens', 'wte', 'wpe']) and hasattr(module, 'forward'):
            try:
                # Get parent and attribute name
                parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else None
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                else:
                    parent = model
                
                # Freeze the module's parameters (already done, but ensure it)
                for param in module.parameters():
                    param.requires_grad = False
                
                # Compile the module with static graph (dynamic=False)
                compiled_module = torch.compile(module, mode='default', dynamic=False)
                
                # Replace the original module with compiled version
                setattr(parent, attr_name, compiled_module)
                
                compiled_count += 1
                st.success(f"✓ Compiled embedding: {name}")
                
            except Exception as e:
                st.warning(f"Could not compile {name}: {e}")
    
    if compiled_count > 0:
        st.success(f"✓ Compiled {compiled_count} embedding modules with static graph")
        st.info("💡 Frozen embeddings now use compiled forward pass (gradients still flow through)")
    else:
        st.warning("⚠️ No embedding modules found to compile")
        st.info("Embeddings are frozen but not compiled - forward pass not optimized")
    
    return compiled_count


def apply_attention_cache(model):
    """
    Attempt to compile frozen attention layers.
    WARNING: This is experimental and may break training or not provide benefits.
    Attention layers are complex and may not compile well.
    """
    st.info("⚡ Attempting to compile attention layers...")
    
    if not hasattr(torch, 'compile'):
        st.warning("⚠️ torch.compile() not available (requires PyTorch 2.0+)")
        st.info("Attention is frozen but not compiled")
        return 0
    
    st.warning("⚠️ Compiling attention is EXPERIMENTAL and may fail or break training")
    st.info("💡 Consider leaving this disabled unless you're testing")
    
    compiled_count = 0
    
    # Try to compile attention modules
    for name, module in model.named_modules():
        if 'self_attn' in name.lower() and hasattr(module, 'forward'):
            try:
                # Get parent and attribute name
                parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else None
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                else:
                    parent = model
                
                # Freeze parameters
                for param in module.parameters():
                    param.requires_grad = False
                
                # Compile with dynamic=True for attention (it has variable seq lengths)
                compiled_module = torch.compile(module, mode='default', dynamic=True)
                
                setattr(parent, attr_name, compiled_module)
                compiled_count += 1
                st.info(f"Compiled attention: {name}")
                
            except Exception as e:
                st.warning(f"Could not compile {name}: {e}")
    
    if compiled_count > 0:
        st.success(f"✓ Compiled {compiled_count} attention modules")
        st.warning("Monitor loss carefully - compilation may cause issues")
    else:
        st.info("No attention modules compiled (this is often safer)")
    
    return compiled_count


def analyze_model_parameters(model):
    """
    Analyze and categorize all model parameters.
    
    Returns:
        Dictionary with parameter categories and their counts/names
    """
    categories = {
        'attention': {'count': 0, 'params': []},
        'embeddings': {'count': 0, 'params': []},
        'layernorm': {'count': 0, 'params': []},
        'mlp': {'count': 0, 'params': []},
        'output': {'count': 0, 'params': []},
        'other': {'count': 0, 'params': []}
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        
        # Categorize parameters
        if any(attn in name.lower() for attn in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn', 'attention']):
            categories['attention']['count'] += param_count
            categories['attention']['params'].append((name, param_count, param))
        elif any(emb in name.lower() for emb in ['embed', 'wte', 'wpe', 'position', 'token']):
            categories['embeddings']['count'] += param_count
            categories['embeddings']['params'].append((name, param_count, param))
        elif any(norm in name.lower() for norm in ['norm', 'ln']):
            categories['layernorm']['count'] += param_count
            categories['layernorm']['params'].append((name, param_count, param))
        elif any(mlp in name.lower() for mlp in ['mlp', 'ffn', 'fc', 'feed_forward', 'gate_proj', 'up_proj', 'down_proj']):
            categories['mlp']['count'] += param_count
            categories['mlp']['params'].append((name, param_count, param))
        elif any(out in name.lower() for out in ['lm_head', 'output', 'score']):
            categories['output']['count'] += param_count
            categories['output']['params'].append((name, param_count, param))
        else:
            categories['other']['count'] += param_count
            categories['other']['params'].append((name, param_count, param))
    
    return categories


def display_parameter_analysis(model):
    """
    Display comprehensive parameter analysis.
    """
    categories = analyze_model_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    st.markdown("### 📊 Model Parameter Analysis")
    
    # Overall statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Parameters", f"{total_params:,}")
    with col2:
        trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0
        st.metric("🔥 Trainable", f"{trainable_params:,} ({trainable_pct:.1f}%)")
    with col3:
        frozen_pct = (frozen_params / total_params * 100) if total_params > 0 else 0
        st.metric("❄️ Frozen", f"{frozen_params:,} ({frozen_pct:.1f}%)")
    
    # Category breakdown
    st.markdown("#### Parameter Categories:")
    
    # Create table data
    table_data = []
    for category_name, category_data in categories.items():
        count = category_data['count']
        if count > 0:
            pct = (count / total_params * 100) if total_params > 0 else 0
            # Check if any params in this category are trainable
            trainable_in_cat = sum(p.numel() for _, _, p in category_data['params'] if p.requires_grad)
            frozen_in_cat = count - trainable_in_cat
            status = "🔥" if trainable_in_cat > 0 else "❄️"
            table_data.append({
                'Category': category_name.upper(),
                'Parameters': f"{count:,}",
                'Percentage': f"{pct:.1f}%",
                'Trainable': f"{trainable_in_cat:,}",
                'Frozen': f"{frozen_in_cat:,}",
                'Status': status
            })
    
    # Display as columns for better readability
    for data in table_data:
        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 2, 2, 1])
        with col1:
            st.write(f"**{data['Category']}**")
        with col2:
            st.write(data['Parameters'])
        with col3:
            st.write(data['Percentage'])
        with col4:
            st.write(data['Trainable'])
        with col5:
            st.write(data['Frozen'])
        with col6:
            st.write(data['Status'])


def freeze_parameters_by_category(model, category_name, optimizer):
    """
    Freeze parameters in a specific category.
    Does NOT recreate the optimizer - just sets requires_grad=False.
    
    Args:
        model: The model
        category_name: 'attention', 'embeddings', 'layernorm', 'mlp', 'output', or 'other'
        optimizer: The optimizer (not modified, just for compatibility)
    
    Returns:
        Number of parameters frozen
    """
    categories = analyze_model_parameters(model)
    
    if category_name not in categories:
        st.error(f"Unknown category: {category_name}")
        return 0
    
    category_data = categories[category_name]
    frozen_count = 0
    frozen_names = []
    
    for name, param_count, param in category_data['params']:
        if param.requires_grad:
            param.requires_grad = False
            frozen_count += param_count
            frozen_names.append(name)
    
    if frozen_count > 0:
        st.success(f"❄️ Froze {frozen_count:,} parameters in category: {category_name.upper()}")
        st.caption(f"Frozen {len(frozen_names)} parameter groups in this category")
    
    return frozen_count


def collate_batch(batch_items, tokenizer):
    """
    Collate a batch of items into padded tensors.
    Returns: (batch_positive_ids, batch_negative_ids_list, batch_valid_lengths, 
              pos_breakdown_indices, neg_breakdown_indices_list)
    """
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    # Find max lengths
    max_pos_len = max(item['positive'].size(0) for item in batch_items)
    max_neg_len = 0
    for item in batch_items:
        for neg in item['negatives']:
            if neg is not None and neg.numel() > 0:
                max_neg_len = max(max_neg_len, neg.size(0))
    
    # Pad positives and collect breakdown indices
    batch_positives = []
    pos_lengths = []
    pos_breakdown_indices = []
    for item in batch_items:
        pos = item['positive']
        pos_lengths.append(pos.size(0))
        pos_breakdown_indices.append(item.get('positive_breakdown_idx', 0))
        if pos.size(0) < max_pos_len:
            padding = torch.full((max_pos_len - pos.size(0),), pad_token_id, dtype=pos.dtype)
            pos = torch.cat([pos, padding])
        batch_positives.append(pos)
    
    batch_positives = torch.stack(batch_positives)
    
    # Organize negatives and their breakdown indices by batch item
    batch_negatives_list = []
    neg_breakdown_indices_list = []
    for item in batch_items:
        item_negs = []
        item_neg_breakdown_indices = []
        neg_breakdown_indices = item.get('negative_breakdown_indices', [])
        
        for idx, neg in enumerate(item['negatives']):
            if neg is None or neg.numel() <= 1:
                continue
            if neg.size(0) < max_neg_len:
                padding = torch.full((max_neg_len - neg.size(0),), pad_token_id, dtype=neg.dtype)
                neg = torch.cat([neg, padding])
            item_negs.append(neg)
            # Get breakdown index for this negative, or 0 if not available
            breakdown_idx = neg_breakdown_indices[idx] if idx < len(neg_breakdown_indices) else 0
            item_neg_breakdown_indices.append(breakdown_idx)
        
        batch_negatives_list.append(item_negs)
        neg_breakdown_indices_list.append(item_neg_breakdown_indices)
    
    return batch_positives, batch_negatives_list, pos_lengths, pos_breakdown_indices, neg_breakdown_indices_list


def process_training_batch(model, batch_items, tokenizer, optimizer, config, step_count):
    """Process a batch of training items"""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    batch_positives, batch_negatives_list, pos_lengths, pos_breakdown_indices, neg_breakdown_indices_list = collate_batch(batch_items, tokenizer)
    
    total_loss = 0
    total_pos_loss = 0
    total_neg_loss = 0
    valid_samples = 0
    
    # Process each item in the batch
    for i in range(len(batch_items)):
        pos_ids = batch_positives[i][:pos_lengths[i]]  # Remove padding
        neg_ids = batch_negatives_list[i]
        pos_breakdown_idx = pos_breakdown_indices[i]
        neg_breakdown_indices = neg_breakdown_indices_list[i]
        
        loss, pos_loss, neg_loss = optimized_contrastive_loss(
            model, pos_ids, neg_ids, tokenizer,
            pos_breakdown_idx=pos_breakdown_idx,
            neg_breakdown_indices=neg_breakdown_indices
        )
        
        # Check for NaN
        if torch.isnan(loss):
            continue
        
        total_loss += loss.item()
        total_pos_loss += pos_loss
        total_neg_loss += neg_loss
        valid_samples += 1
        
        # Accumulate gradients
        loss = loss / config['batch_accumulation']
        loss.backward()
        
        del loss
    
    step_count += 1
    
    # Clear intermediate tensors
    del batch_positives, batch_negatives_list
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Update weights after accumulation
    if step_count % config['batch_accumulation'] == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('max_grad_norm', 0.5))
        optimizer.step()
        optimizer.zero_grad()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Return average losses
    if valid_samples > 0:
        avg_pos_loss = total_pos_loss / valid_samples
        avg_neg_loss = total_neg_loss / valid_samples
    else:
        avg_pos_loss = 0.0
        avg_neg_loss = 0.0
    
    return avg_pos_loss, avg_neg_loss, step_count, valid_samples


def handle_remaining_gradients(model, optimizer, config, step_count):
    if step_count % config['batch_accumulation'] != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('max_grad_norm', 0.5))
        optimizer.step()
        optimizer.zero_grad()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def perform_garbage_collection_if_needed(epoch, config, ui_components):
    if (epoch + 1) % config['gc_frequency'] == 0:
        pre_gc_memory = get_memory_usage()
        collected_objects = perform_garbage_collection()
        post_gc_memory = get_memory_usage()
        
        if device.type == 'cuda':
            gpu_stats = cuda_memory_stats()
            ui_components['memory_info'].info(f"🗑️ GC (Epoch {epoch + 1}): "
                                            f"Collected {collected_objects} objects | "
                                            f"CPU: {pre_gc_memory['cpu_memory_mb']:.1f}MB → {post_gc_memory['cpu_memory_mb']:.1f}MB | "
                                            f"GPU: {gpu_stats['allocated']:.1f}GB used, {gpu_stats['reserved']:.1f}GB reserved")
        else:
            memory_saved = pre_gc_memory['cpu_memory_mb'] - post_gc_memory['cpu_memory_mb']
            ui_components['memory_info'].info(f"🗑️ GC (Epoch {epoch + 1}): "
                                            f"Collected {collected_objects} objects, "
                                            f"Memory: {pre_gc_memory['cpu_memory_mb']:.1f}MB → {post_gc_memory['cpu_memory_mb']:.1f}MB "
                                            f"({'freed' if memory_saved > 0 else 'no change'}: {abs(memory_saved):.1f}MB)")


def update_training_status(epoch, config, epoch_time, avg_loss, avg_pos_loss, avg_neg_loss, optimizer, ui_components, batch_size):
    current_lr = optimizer.param_groups[0]['lr']
    current_memory = get_memory_usage()
    
    if device.type == 'cuda':
        gpu_stats = cuda_memory_stats()
        ui_components['status_text'].text(
            f"Epoch {epoch + 1}/{config['num_epochs']} | "
            f"Time: {epoch_time:.4f}s | "
            f"Batch: {batch_size} | "
            f"Loss: {avg_loss:.4f} | "
            f"Pos: {avg_pos_loss:.4f} | "
            f"Neg: {avg_neg_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"GPU: {gpu_stats['allocated']:.1f}GB"
        )
    else:
        ui_components['status_text'].text(
            f"Epoch {epoch + 1}/{config['num_epochs']} | "
            f"Time: {epoch_time:.4f}s | "
            f"Batch: {batch_size} | "
            f"Loss: {avg_loss:.4f} | "
            f"Pos: {avg_pos_loss:.4f} | "
            f"Neg: {avg_neg_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Memory: {current_memory['cpu_memory_mb']:.1f}MB"
        )


def update_loss_chart(epoch, config, losses, pos_losses, neg_losses, ui_components):
    if epoch % 1 == 0 or epoch == config['num_epochs'] - 1:
        with ui_components['loss_chart_placeholder'].container():
            st.line_chart({
                'Total Loss': losses,
                'Positive Loss': pos_losses,
                'Negative Loss': neg_losses
            })


def run_training_loop(model, optimizer, config, ui_components, valid_indices):
    """
    Main training loop with batching support for contrastive learning.
    """
    losses = []
    pos_losses = []
    neg_losses = []
    
    total_samples = len(valid_indices)
    batch_size = config.get('batch_size', 4)  # Default batch size
    chunk_size = config.get('memory_chunk_size', 100)
    
    # Track freezing state
    attention_frozen = False
    embedding_frozen = False
    mlp_frozen = False
    lmhead_frozen = False
    
    freeze_attention_epoch = config.get('freeze_attention_after_epoch', 0)
    freeze_embedding_epoch = config.get('freeze_embedding_after_epoch', 0)
    freeze_mlp_epoch = config.get('freeze_mlp_after_epoch', 0)
    freeze_lmhead_epoch = config.get('freeze_lmhead_after_epoch', 0)
    
    enable_attention_freeze = config.get('enable_attention_freeze', False)
    enable_embedding_freeze = config.get('enable_embedding_freeze', False)
    enable_mlp_freeze = config.get('enable_mlp_freeze', False)
    enable_lmhead_freeze = config.get('enable_lmhead_freeze', False)
    
    # Show initial parameter analysis if any freezing is enabled
    any_freezing_enabled = enable_attention_freeze or enable_embedding_freeze or enable_mlp_freeze or enable_lmhead_freeze
    
    if any_freezing_enabled:
        st.info("📊 Initial parameter analysis:")
        display_parameter_analysis(model)
        
        if enable_attention_freeze and freeze_attention_epoch > 0:
            st.info(f"❄️ Attention will freeze after epoch {freeze_attention_epoch}")
        if enable_embedding_freeze and freeze_embedding_epoch > 0:
            st.info(f"❄️ Embeddings will freeze after epoch {freeze_embedding_epoch}")
        if enable_mlp_freeze and freeze_mlp_epoch > 0:
            st.info(f"❄️ MLP will freeze after epoch {freeze_mlp_epoch}")
        if enable_lmhead_freeze and freeze_lmhead_epoch > 0:
            st.info(f"❄️ LM Head will freeze after epoch {freeze_lmhead_epoch}")
    
    for epoch in range(config['num_epochs']):
        # Check if we should freeze attention this epoch
        if enable_attention_freeze and not attention_frozen and epoch + 1 >= freeze_attention_epoch:
            st.warning(f"❄️ Freezing ATTENTION parameters at epoch {epoch + 1}...")
            freeze_parameters_by_category(model, 'attention', optimizer)
            attention_frozen = True
            
            # Apply attention cache if enabled
            if config.get('use_attention_cache', False):
                apply_attention_cache(model)
            
            display_parameter_analysis(model)
        
        # Check if we should freeze embeddings this epoch
        if enable_embedding_freeze and not embedding_frozen and epoch + 1 >= freeze_embedding_epoch:
            st.warning(f"❄️ Freezing EMBEDDING parameters at epoch {epoch + 1}...")
            freeze_parameters_by_category(model, 'embeddings', optimizer)
            embedding_frozen = True
            
            # Apply static graph optimization if enabled
            if config.get('use_static_graph', False):
                apply_static_graph_to_embeddings(model)
            
            display_parameter_analysis(model)
        
        # Check if we should freeze MLP this epoch
        if enable_mlp_freeze and not mlp_frozen and epoch + 1 >= freeze_mlp_epoch:
            st.warning(f"❄️ Freezing MLP parameters at epoch {epoch + 1}...")
            freeze_parameters_by_category(model, 'mlp', optimizer)
            mlp_frozen = True
            display_parameter_analysis(model)
        
        # Check if we should freeze LM Head this epoch
        if enable_lmhead_freeze and not lmhead_frozen and epoch + 1 >= freeze_lmhead_epoch:
            st.warning(f"❄️ Freezing OUTPUT/LM HEAD parameters at epoch {epoch + 1}...")
            freeze_parameters_by_category(model, 'output', optimizer)
            lmhead_frozen = True
            display_parameter_analysis(model)
        
        epoch_start_time = time.time()
        model.train()
        
        epoch_loss = 0
        epoch_pos_loss = 0
        epoch_neg_loss = 0
        epoch_samples = 0
        step_count = 0
        
        ui_components['main_progress_bar'].progress(epoch / config['num_epochs'])
        
        # Process dataset in chunks (for memory safety)
        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_indices = valid_indices[chunk_start:chunk_end]
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Process chunk in batches
            for batch_start in range(0, len(chunk_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(chunk_indices))
                batch_indices = chunk_indices[batch_start:batch_end]
                
                try:
                    # Load batch items
                    batch_items = [st.session_state.dataset_iterator[idx] for idx in batch_indices]
                    
                    # Process batch
                    pos_loss, neg_loss, step_count, valid_samples = process_training_batch(
                        model, batch_items, st.session_state.tokenizer, optimizer, config, step_count
                    )
                    
                    if valid_samples > 0:
                        # Calculate actual loss for tracking
                        loss = pos_loss - 0.01 * neg_loss
                        
                        epoch_loss += loss * valid_samples
                        epoch_pos_loss += pos_loss * valid_samples
                        epoch_neg_loss += neg_loss * valid_samples
                        epoch_samples += valid_samples
                    
                    # Update progress
                    overall_progress = (chunk_start + batch_end) / total_samples
                    ui_components['epoch_progress_bar'].progress(overall_progress)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        st.error(f"CUDA OOM at batch starting at sample {batch_indices[0]}. Clearing cache and skipping...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        perform_garbage_collection()
                        continue
                    else:
                        raise e
            
            if config['sync_cuda'] and device.type == 'cuda':
                torch.cuda.synchronize()
        
        handle_remaining_gradients(model, optimizer, config, step_count)
        
        # Calculate epoch averages
        if epoch_samples > 0:
            avg_loss = float(epoch_loss / epoch_samples)
            avg_pos_loss = float(epoch_pos_loss / epoch_samples)
            avg_neg_loss = float(epoch_neg_loss / epoch_samples)
        else:
            avg_loss = 0.0
            avg_pos_loss = 0.0
            avg_neg_loss = 0.0
        
        perform_garbage_collection_if_needed(epoch, config, ui_components)
        
        epoch_time = time.time() - epoch_start_time
        losses.append(avg_loss)
        pos_losses.append(avg_pos_loss)
        neg_losses.append(avg_neg_loss)
        
        update_training_status(epoch, config, epoch_time, avg_loss, avg_pos_loss, avg_neg_loss, optimizer, ui_components, batch_size)
        update_loss_chart(epoch, config, losses, pos_losses, neg_losses, ui_components)
    
    return losses, pos_losses, neg_losses