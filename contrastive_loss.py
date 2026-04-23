import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _extract_logits(model_output):
    # Helper to accept raw tensors, tuples, or transformer ModelOutput
    if isinstance(model_output, tuple):
        logits = model_output[0]
    elif hasattr(model_output, "logits"):
        logits = model_output.logits
    else:
        logits = model_output
    return logits

def optimized_contrastive_loss(model, positive_ids, negative_ids_list, tokenizer, base_weight=0.01, 
                               pos_breakdown_idx=0, neg_breakdown_indices=None):
    """
    Numerically robust contrastive loss with breakdown support.
    Returns (total_loss_tensor, positive_loss_item, negative_loss_item)
    
    Args:
        model: The language model
        positive_ids: Positive example token IDs
        negative_ids_list: List of negative example token IDs
        tokenizer: Tokenizer
        base_weight: Base weight for contrastive component
        pos_breakdown_idx: Token index where learning starts for positive example
        neg_breakdown_indices: List of token indices where learning starts for each negative example
    """
    device = next(model.parameters()).device

    # Basic input checks
    if positive_ids is None or positive_ids.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0.0
    if positive_ids.size(0) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0.0

    positive_ids = positive_ids.to(device, non_blocking=True)
    pad_token_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else getattr(tokenizer, "eos_token_id", None)
    num_negatives = len(negative_ids_list)
    base_weight = float(max(0.001, min(0.1, base_weight)))
    
    if neg_breakdown_indices is None:
        neg_breakdown_indices = [0] * num_negatives

    # ---------- Positive example ----------
    try:
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            raw_pos_out = model(positive_ids.unsqueeze(0))
        pos_logits = _extract_logits(raw_pos_out)

        # Ensure tensor and float dtype for stability
        pos_logits = pos_logits.float()

        # defensive nan/inf repair (replace extreme/infinite with large finite numbers)
        if torch.isnan(pos_logits).any() or torch.isinf(pos_logits).any():
            pos_logits = torch.nan_to_num(pos_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        # Expect shape [batch, seq_len, vocab]
        if pos_logits.dim() != 3:
            raise RuntimeError(f"Unexpected logits shape for positive example: {tuple(pos_logits.shape)}")

        # Align lengths
        logits_seq_len = pos_logits.size(1)
        targets = positive_ids[1:].contiguous()
        if logits_seq_len - 1 <= 0 or targets.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0.0

        logits_for_loss = pos_logits[0, :logits_seq_len-1, :].contiguous()
        min_len = min(logits_for_loss.size(0), targets.size(0))
        logits_for_loss = logits_for_loss[:min_len]
        targets = targets[:min_len]

        # Apply breakdown masking for positive example
        if pos_breakdown_idx > 0:
            masked_targets = targets.clone()
            mask_until = max(0, pos_breakdown_idx - 1)
            if mask_until < len(masked_targets):
                masked_targets[:mask_until] = -100
            targets = masked_targets

        # Loss function with ignore_index for masked tokens
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
        pos_loss = loss_fn(logits_for_loss.view(-1, logits_for_loss.size(-1)), targets.view(-1))

        # final guard: replace NaN/Inf if they somehow appear
        if torch.isnan(pos_loss) or torch.isinf(pos_loss):
            log_probs = F.log_softmax(logits_for_loss, dim=-1)
            nll_loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), targets.view(-1), ignore_index=-100)
            if torch.isnan(nll_loss) or torch.isinf(nll_loss):
                print(f"WARNING: positive loss still invalid after fallback: {nll_loss}")
                return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0.0
            pos_loss = nll_loss

    except Exception as e:
        print(f"ERROR in positive loss computation: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0, 0.0

    # free pos logits if large
    try:
        del raw_pos_out, pos_logits, logits_for_loss, targets
    except Exception:
        pass

    if num_negatives == 0:
        return pos_loss, pos_loss.item(), 0.0

    # ---------- Negatives ----------
    contrastive_weight = base_weight * math.exp(-0.05 * min(num_negatives, 20))
    neg_batch_size = min(2, num_negatives)
    valid_neg_losses = []

    for batch_start in range(0, num_negatives, neg_batch_size):
        try:
            batch_end = min(batch_start + neg_batch_size, num_negatives)
            batch_negatives = negative_ids_list[batch_start:batch_end]
            batch_breakdown_indices = neg_breakdown_indices[batch_start:batch_end]
            valid_batch_negatives = []
            valid_batch_breakdown_indices = []
            
            for idx, neg_ids in enumerate(batch_negatives):
                if neg_ids is None or neg_ids.numel() <= 1:
                    continue
                valid_batch_negatives.append(neg_ids.to(device, non_blocking=True))
                valid_batch_breakdown_indices.append(batch_breakdown_indices[idx])

            if not valid_batch_negatives:
                continue

            max_len = max(neg.size(0) for neg in valid_batch_negatives)
            padded = []
            valid_lengths = []
            for neg_ids in valid_batch_negatives:
                valid_lengths.append(neg_ids.size(0))
                if neg_ids.size(0) < max_len:
                    padding_size = max_len - neg_ids.size(0)
                    padding = torch.full((padding_size,), pad_token_id, dtype=neg_ids.dtype, device=device)
                    padded_neg = torch.cat([neg_ids, padding])
                else:
                    padded_neg = neg_ids
                padded.append(padded_neg)

            batch_tensor = torch.stack(padded)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                raw_neg_out = model(batch_tensor)
            neg_logits = _extract_logits(raw_neg_out)
            neg_logits = neg_logits.float()
            if torch.isnan(neg_logits).any() or torch.isinf(neg_logits).any():
                neg_logits = torch.nan_to_num(neg_logits, nan=0.0, posinf=1e4, neginf=-1e4)

            # compute per-item losses with breakdown masking
            for j, valid_len in enumerate(valid_lengths):
                if valid_len <= 1:
                    continue
                neg_targets = valid_batch_negatives[j][1:valid_len].contiguous()
                neg_logits_for_loss = neg_logits[j, :valid_len-1, :].contiguous()
                min_len = min(neg_logits_for_loss.size(0), neg_targets.size(0))
                neg_logits_for_loss = neg_logits_for_loss[:min_len]
                neg_targets = neg_targets[:min_len]
                
                # Apply breakdown masking for negative example
                neg_breakdown_idx = valid_batch_breakdown_indices[j]
                if neg_breakdown_idx > 0:
                    masked_neg_targets = neg_targets.clone()
                    mask_until = max(0, neg_breakdown_idx - 1)
                    if mask_until < len(masked_neg_targets):
                        masked_neg_targets[:mask_until] = -100
                    neg_targets = masked_neg_targets
                
                loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
                neg_loss = loss_fn(neg_logits_for_loss.view(-1, neg_logits_for_loss.size(-1)), neg_targets.view(-1))
                if not (torch.isnan(neg_loss) or torch.isinf(neg_loss)):
                    valid_neg_losses.append(neg_loss)
                else:
                    # fallback to stable nll
                    log_probs = F.log_softmax(neg_logits_for_loss, dim=-1)
                    nll_loss = F.nll_loss(log_probs.view(-1, log_probs.size(-1)), neg_targets.view(-1), ignore_index=-100)
                    if not (torch.isnan(nll_loss) or torch.isinf(nll_loss)):
                        valid_neg_losses.append(nll_loss)
                    else:
                        print(f"WARNING: Invalid negative loss in batch {batch_start}, item {j}")
            try:
                del raw_neg_out, neg_logits, batch_tensor
            except Exception:
                pass

        except Exception as e:
            print(f"ERROR processing negative batch {batch_start}: {e}")
            continue

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------- Final aggregation ----------
    if valid_neg_losses:
        try:
            avg_neg_loss = torch.stack(valid_neg_losses).mean()
            if torch.isnan(avg_neg_loss) or torch.isinf(avg_neg_loss):
                print("WARNING: Invalid average negative loss")
                return pos_loss, pos_loss.item(), 0.0

            pos_loss_clamped = torch.clamp(pos_loss, max=1e3)
            avg_neg_loss_clamped = torch.clamp(avg_neg_loss, max=1e3)

            total_loss = pos_loss_clamped - contrastive_weight * avg_neg_loss_clamped

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("WARNING: Invalid total loss")
                return pos_loss, pos_loss.item(), avg_neg_loss.item()

            neg_loss_item = avg_neg_loss.item()
            del avg_neg_loss, valid_neg_losses
            return total_loss, pos_loss.item(), neg_loss_item

        except Exception as e:
            print(f"ERROR in final loss computation: {e}")
            return pos_loss, pos_loss.item(), 0.0

    return pos_loss, pos_loss.item(), 0.0