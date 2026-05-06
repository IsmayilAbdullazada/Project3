import torch
import torch.nn.functional as F

def ctc_loss_from_logits(
    logits_btn: torch.Tensor,          # (B, T, N) unnormalized scores
    targets_bk: torch.Tensor,          # (B, K) int64 target labels
    input_lengths_b: torch.Tensor,     # (B,) lengths in frames (<= T)
    target_lengths_b: torch.Tensor,    # (B,) lengths in symbols (<= K)
    blank_id: int = 0,
    target_pad_id = None,  # if targets are padded, set this (e.g., blank_id)
    reduction: str = "mean",
    zero_infinity: bool = True,
):
    """
    Computes CTC loss from (B,T,N) logits + (B,K) padded targets from scratch.
    """
    B, T, N = logits_btn.size()
    device = logits_btn.device
    
    # Move to log probabilities
    log_probs = F.log_softmax(logits_btn, dim=-1)
    
    max_target_len = target_lengths_b.max().item()
    S = 2 * max_target_len + 1  # Length of extended targets with blanks
    
    # Initialize extended targets sequence (b, L1, b, L2, b, L3 ...)
    ext_targets = torch.full((B, S), blank_id, dtype=torch.long, device=device)
    if max_target_len > 0:
        ext_targets[:, 1::2] = targets_bk[:, :max_target_len]
        
    # Mask to allow transitions directly between differing consecutive labels (jumping over the blank state)
    mask_move2 = torch.zeros((B, S), dtype=torch.bool, device=device)
    if S > 2:
        s_idx = torch.arange(3, S, 2, device=device)
        mask_move2[:, s_idx] = (ext_targets[:, s_idx] != ext_targets[:, s_idx - 2])
        
    T_max = min(T, max(1, input_lengths_b.max().item()))
    
    # Pre-gather all target emissions up to T_max
    ext_targets_exp = ext_targets.unsqueeze(1).expand(B, T_max, S)
    emissions = log_probs[:, :T_max, :].gather(2, ext_targets_exp) # (B, T_max, S)
    
    # Store DP table
    all_alphas =[]
    
    alpha = torch.full((B, S), -float('inf'), device=device)
    alpha[:, 0] = log_probs[:, 0, blank_id]
    valid_b = target_lengths_b > 0
    if valid_b.any():
        alpha[valid_b, 1] = log_probs[valid_b, 0, ext_targets[valid_b, 1]]
        
    all_alphas.append(alpha)
    
    # DP forward loop
    for t in range(1, T_max):
        emissions_t = emissions[:, t, :]
        
        # 1. Stay in the same state
        trans_stay = alpha
        
        # 2. Move 1 state forward (blank <-> label)
        trans_move1 = torch.full_like(alpha, -float('inf'))
        trans_move1[:, 1:] = alpha[:, :-1]
        
        # 3. Move 2 states forward (label -> label directly without visiting blank, allowed only if they differ)
        trans_move2 = torch.full_like(alpha, -float('inf'))
        trans_move2[:, 2:] = alpha[:, :-2]
        trans_move2.masked_fill_(~mask_move2, -float('inf'))
        
        # Collect top transitions and sum
        stacked = torch.stack([trans_stay, trans_move1, trans_move2], dim=0)
        trans_max = torch.logsumexp(stacked, dim=0)
        
        alpha = trans_max + emissions_t
        all_alphas.append(alpha)
        
    all_alphas = torch.stack(all_alphas, dim=0) # (T_max, B, S)
    
    # Grab alphas at the respective final valid timestamps based on input lengths
    T_b_idx = (input_lengths_b - 1).clamp(min=0, max=T_max - 1)
    T_b_idx_exp = T_b_idx.unsqueeze(0).unsqueeze(-1).expand(1, B, S)
    final_alpha = all_alphas.gather(0, T_b_idx_exp).squeeze(0) # (B, S)
    
    # Terminal states can be either the trailing blank or the final label
    idx1 = 2 * target_lengths_b
    idx2 = 2 * target_lengths_b - 1
    
    valid_idx2 = idx2 >= 0
    idx2_safe = torch.clamp(idx2, min=0)
    
    log_prob1 = final_alpha.gather(1, idx1.unsqueeze(1)).squeeze(1)
    log_prob2 = final_alpha.gather(1, idx2_safe.unsqueeze(1)).squeeze(1)
    log_prob2 = torch.where(valid_idx2, log_prob2, torch.full_like(log_prob2, -float('inf')))
    
    log_p = torch.logsumexp(torch.stack([log_prob1, log_prob2], dim=0), dim=0)
    loss = -log_p
    
    # Resolve NaNs or Infinite losses generated on early steps / zero length predictions
    if zero_infinity:
        loss = torch.where(loss == float('inf'), torch.zeros_like(loss), loss)
        
    if reduction == "mean":
        return (loss / torch.clamp(target_lengths_b.to(loss.dtype), min=1.0)).mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss