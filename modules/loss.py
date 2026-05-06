import torch
import torch.nn.functional as F

def ctc_loss_from_logits(
    logits_btn: torch.Tensor,          # (B, T, N) unnormalized scores
    targets_bk: torch.Tensor,          # (B, K) int64 target labels
    input_lengths_b: torch.Tensor,     # (B,) lengths in frames (<= T)
    target_lengths_b: torch.Tensor,    # (B,) lengths in symbols (<= K)
    blank_id: int = 0,
    target_pad_id=None,
    reduction: str = "mean",
    zero_infinity: bool = True,
):
    """
    Computes CTC loss from (B,T,N) logits + (B,K) padded targets from scratch.
    """
    B, T, N = logits_btn.size()
    device = logits_btn.device
    dtype = logits_btn.dtype
    
    # Clamp input lengths to valid range
    input_lengths_b = input_lengths_b.clamp(min=1, max=T)
    
    # Move to log probabilities with numerical stability
    log_probs = F.log_softmax(logits_btn, dim=-1)
    
    # Handle edge case: if all target lengths are 0
    max_target_len = target_lengths_b.max().item()
    if max_target_len == 0:
        # Loss is 0 for empty targets (or could be -log(prob of all blanks))
        if reduction == "mean":
            return torch.tensor(0.0, device=device, dtype=dtype)
        elif reduction == "sum":
            return torch.tensor(0.0, device=device, dtype=dtype)
        else:
            return torch.zeros(B, device=device, dtype=dtype)
    
    S = 2 * max_target_len + 1  # Length of extended targets with blanks
    
    # Initialize extended targets sequence (b, L1, b, L2, b, L3 ...)
    ext_targets = torch.full((B, S), blank_id, dtype=torch.long, device=device)
    ext_targets[:, 1::2] = targets_bk[:, :max_target_len]
    
    # Mask for valid positions in extended target (per sample)
    # For sample b with target_length L, valid positions are 0 to 2*L (inclusive)
    s_range = torch.arange(S, device=device).unsqueeze(0)  # (1, S)
    valid_s_mask = s_range <= (2 * target_lengths_b.unsqueeze(1))  # (B, S)
    
    # Mask to allow transitions directly between differing consecutive labels
    mask_move2 = torch.zeros((B, S), dtype=torch.bool, device=device)
    if S > 2:
        s_idx = torch.arange(2, S, device=device)
        # Can skip blank (move by 2) only if current label differs from label 2 positions back
        # and the position 2 back is a label (odd index), current is also label (odd index)
        for s in range(3, S, 2):  # only odd indices >= 3 (label positions)
            mask_move2[:, s] = (ext_targets[:, s] != ext_targets[:, s - 2])
    
    T_max = input_lengths_b.max().item()
    
    # Pre-gather all target emissions
    ext_targets_exp = ext_targets.unsqueeze(1).expand(B, T_max, S)
    emissions = log_probs[:, :T_max, :].gather(2, ext_targets_exp)  # (B, T_max, S)
    
    # Initialize alpha
    NEG_INF = -1e9  # Use large negative instead of -inf for stability
    
    alpha = torch.full((B, S), NEG_INF, device=device, dtype=dtype)
    alpha[:, 0] = log_probs[:, 0, blank_id]
    
    # For samples with target_length > 0, initialize alpha[:, 1]
    valid_b = target_lengths_b > 0
    if valid_b.any():
        first_label_idx = ext_targets[valid_b, 1]
        alpha[valid_b, 1] = log_probs[valid_b, 0, :].gather(1, first_label_idx.unsqueeze(1)).squeeze(1)
    
    # Mask out invalid positions
    alpha = torch.where(valid_s_mask, alpha, torch.full_like(alpha, NEG_INF))
    
    # DP forward loop
    for t in range(1, T_max):
        # Mask for samples where t < input_length (still processing)
        active_mask = (t < input_lengths_b).unsqueeze(1)  # (B, 1)
        
        emissions_t = emissions[:, t, :]  # (B, S)
        
        # 1. Stay in the same state
        trans_stay = alpha
        
        # 2. Move 1 state forward
        trans_move1 = torch.full_like(alpha, NEG_INF)
        trans_move1[:, 1:] = alpha[:, :-1]
        
        # 3. Move 2 states forward (only for differing consecutive labels)
        trans_move2 = torch.full_like(alpha, NEG_INF)
        trans_move2[:, 2:] = alpha[:, :-2]
        trans_move2 = torch.where(mask_move2, trans_move2, torch.full_like(trans_move2, NEG_INF))
        
        # Log-sum-exp of transitions
        stacked = torch.stack([trans_stay, trans_move1, trans_move2], dim=0)  # (3, B, S)
        trans_sum = torch.logsumexp(stacked, dim=0)  # (B, S)
        
        new_alpha = trans_sum + emissions_t
        
        # Mask out invalid positions
        new_alpha = torch.where(valid_s_mask, new_alpha, torch.full_like(new_alpha, NEG_INF))
        
        # Only update for active samples
        alpha = torch.where(active_mask, new_alpha, alpha)
    
    # Get final alpha at each sample's last timestep
    # Terminal states: position 2*L (final blank) or 2*L-1 (final label)
    final_alpha = alpha  # alpha already contains the values at the last processed timestep
    
    # For each sample, we need alpha at t = input_length - 1
    # But our loop already handled this with the active_mask
    
    idx_final_blank = 2 * target_lengths_b  # (B,)
    idx_final_label = 2 * target_lengths_b - 1  # (B,)
    
    # Clamp indices to valid range
    idx_final_blank = idx_final_blank.clamp(max=S-1)
    idx_final_label = idx_final_label.clamp(min=0, max=S-1)
    
    log_prob_blank = final_alpha.gather(1, idx_final_blank.unsqueeze(1)).squeeze(1)
    log_prob_label = final_alpha.gather(1, idx_final_label.unsqueeze(1)).squeeze(1)
    
    # For samples with target_length > 0, both are valid; otherwise only blank
    log_prob_label = torch.where(
        target_lengths_b > 0, 
        log_prob_label, 
        torch.full_like(log_prob_label, NEG_INF)
    )
    
    log_p = torch.logsumexp(torch.stack([log_prob_blank, log_prob_label], dim=0), dim=0)
    loss = -log_p
    
    # Handle invalid losses
    if zero_infinity:
        loss = torch.where(
            torch.isfinite(loss), 
            loss, 
            torch.zeros_like(loss)
        )
    
    if reduction == "mean":
        # Normalize by target length, but handle zero-length targets
        target_lengths_safe = target_lengths_b.to(dtype).clamp(min=1.0)
        return (loss / target_lengths_safe).mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
