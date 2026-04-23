import torch
import torch.nn as nn
import torch.nn.functional as F

def _extract_logits(model_output):
    """Helper to accept raw tensors, tuples, or transformer ModelOutput"""
    if isinstance(model_output, tuple):
        logits = model_output[0]
    elif hasattr(model_output, "logits"):
        logits = model_output.logits
    else:
        logits = model_output
    return logits


def standard_language_model_loss(model, input_ids, tokenizer, label_smoothing=0.1, breakdown_idx=0):
    """
    Standard causal language modeling loss (next-token prediction).
    Kernel fusion happens at the model level, not here.
    
    Args:
        model: The language model (may have fused CUDA kernels if enabled)
        input_ids: Token IDs tensor [seq_len]
        tokenizer: Tokenizer (pad_token_id will NOT be used as ignore_index)
        label_smoothing: Label smoothing factor for CrossEntropyLoss
        breakdown_idx: Token index where learning starts (context before this is ignored)
    
    Returns:
        (loss_tensor, loss_value): Tuple of (loss tensor with grad, scalar loss value)
    """
    device = next(model.parameters()).device
    
    # Basic input validation
    if input_ids is None or input_ids.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    if input_ids.size(0) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    
    input_ids = input_ids.to(device, non_blocking=True)
    
    try:
        # Forward pass with mixed precision support
        # Kernel fusion happens here if model was compiled with torch.compile
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            raw_output = model(input_ids.unsqueeze(0))
        
        logits = _extract_logits(raw_output)
        logits = logits.float()  # Ensure float32 for stability
        
        # Handle NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Validate logits shape
        if logits.dim() != 3:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        
        # Prepare inputs and targets for next-token prediction
        # Input: tokens[:-1], Target: tokens[1:]
        logits_seq_len = logits.size(1)
        targets = input_ids[1:].contiguous()
        
        if logits_seq_len - 1 <= 0 or targets.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0
        
        # Align logits and targets
        logits_for_loss = logits[0, :logits_seq_len-1, :].contiguous()
        min_len = min(logits_for_loss.size(0), targets.size(0))
        logits_for_loss = logits_for_loss[:min_len]
        targets = targets[:min_len]
        
        # Apply breakdown masking: set targets before breakdown_idx to -100 (ignore_index)
        if breakdown_idx > 0:
            # Create a masked targets tensor where context tokens are ignored
            masked_targets = targets.clone()
            # Mask all positions before breakdown_idx
            # Note: We need to account for the shift (targets are input_ids[1:])
            # So if breakdown_idx=10, we want to ignore targets[0:9] (which correspond to input_ids[1:10])
            mask_until = max(0, breakdown_idx - 1)
            if mask_until < len(masked_targets):
                masked_targets[:mask_until] = -100
            targets = masked_targets
        
        # Compute cross-entropy loss with label smoothing
        # -100 is the default ignore_index for CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
        loss = loss_fn(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            targets.view(-1)
        )
        
        # Fallback to more stable computation if loss is invalid
        if torch.isnan(loss) or torch.isinf(loss):
            log_probs = F.log_softmax(logits_for_loss, dim=-1)
            nll_loss = F.nll_loss(
                log_probs.view(-1, log_probs.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            if torch.isnan(nll_loss) or torch.isinf(nll_loss):
                print(f"WARNING: Standard loss computation failed, returning zero loss")
                return torch.tensor(0.0, device=device, requires_grad=True), 0.0
            loss = nll_loss
        
        # Clean up
        try:
            del raw_output, logits, logits_for_loss, targets
        except Exception:
            pass
        
        return loss, loss.item()
        
    except Exception as e:
        print(f"ERROR in standard loss computation: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0