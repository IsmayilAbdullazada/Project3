import math
import torch
import torch.nn.functional as F

from modules.loss import ctc_loss_from_logits


# -----------------------------
# 1) Vocab-wide min CTC loss (classification)
# -----------------------------

@torch.no_grad()
def decode_batch_ctc_vocab_minloss(logits_btn, logit_lengths_b, dataset, reduction="none"):
    """
    Scores each utterance against each vocabulary word using CTC loss.
    Returns: LongTensor (B,) = best vocab row index (NOT scr2id id).
    """
    vocab_words =[w for w in dataset.scr2id.keys() if w != "<unk>"]
    device = logits_btn.device
    B = logits_btn.size(0)
    
    # Prepare literal character matrices per vocabulary word
    word_targets =[]
    word_lengths =[]
    for w in vocab_words:
        t = dataset.encode_transcript_letters(w)
        word_targets.append(t)
        word_lengths.append(len(t))
        
    losses =[]
    for t_tensor, t_len in zip(word_targets, word_lengths):
        t_tensor = t_tensor.to(device).unsqueeze(0).expand(B, -1)
        t_lens = torch.full((B,), t_len, dtype=torch.long, device=device)
        
        # Test batch sequences explicitly against this specific word utilizing our implemented CTC Loss
        loss = ctc_loss_from_logits(
            logits_btn,
            t_tensor,
            logit_lengths_b,
            t_lens,
            blank_id=dataset.ctc_blank_id,
            reduction="none",
            zero_infinity=True
        )
        losses.append(loss)
        
    losses = torch.stack(losses, dim=1) # (B, V)
    best = losses.argmin(dim=1) # (B,)
    return best


# -----------------------------
# 2) Greedy CTC decode (sequence)
# -----------------------------

@torch.no_grad()
def decode_batch_ctc_greedy(logits_btn, logit_lengths_b, dataset, blank_id=0):
    """
    Greedy argmax per frame, then CTC collapse repeats and remove blanks.
    Returns: list[str] length B (decoded letter sequences as strings).
    """
    preds = logits_btn.argmax(dim=-1) # (B, T)
    id2letter = {i: ch for ch, i in dataset.letter2id.items()}
    
    out =[]
    for b in range(logits_btn.size(0)):
        seq_len = logit_lengths_b[b].item()
        pred_seq = preds[b, :seq_len].tolist()
        
        collapsed =[]
        prev = -1
        # Collapse sequence duplicates & subsequently omit blanks
        for p in pred_seq:
            if p != prev:
                if p != blank_id:
                    collapsed.append(p)
            prev = p
            
        s = "".join([id2letter.get(p, "") for p in collapsed])
        out.append(s)
        
    return out


# -----------------------------
# 3) Beam search CTC decode (sequence)
# -----------------------------

@torch.no_grad()
def decode_batch_ctc_beam(logits_btn, logit_lengths_b, dataset, beam=4, blank_id=0):
    """
    Simple prefix beam search CTC (no LM).
    Returns: list[str] length B.
    """
    B, T, N = logits_btn.size()
    log_probs = F.log_softmax(logits_btn, dim=-1).cpu()
    logit_lengths = logit_lengths_b.cpu().tolist()
    
    id2letter = {i: ch for ch, i in dataset.letter2id.items()}
    
    def logaddexp(x, y):
        if x == -float('inf'): return y
        if y == -float('inf'): return x
        return max(x, y) + math.log1p(math.exp(-abs(x - y)))

    results =[]
    for b in range(B):
        seq_len = logit_lengths[b]
        
        # Mapping prefix paths structure -> (probability of blank trailing state, probability of label trailing state)
        beam_state = {(): (0.0, -float('inf'))}
        
        for t in range(seq_len):
            next_beam = {}
            for prefix, (p_b, p_nb) in beam_state.items():
                p_total = logaddexp(p_b, p_nb)
                
                # Option 1: Predict blank token
                prob_blank = log_probs[b, t, blank_id].item()
                next_p_b = p_total + prob_blank
                if prefix not in next_beam:
                    next_beam[prefix] =[-float('inf'), -float('inf')]
                next_beam[prefix][0] = logaddexp(next_beam[prefix][0], next_p_b)
                
                # Option 2: Predict structural non-blank tokens
                for c in range(N):
                    if c == blank_id: continue
                    prob_c = log_probs[b, t, c].item()
                    
                    if len(prefix) > 0 and c == prefix[-1]:
                        # A. Extend same path, but NO blank encountered directly -> doesn't append to sequential prefix
                        next_p_nb_same = p_nb + prob_c
                        next_beam[prefix][1] = logaddexp(next_beam[prefix][1], next_p_nb_same)
                        
                        # B. Extend same character path, BUT a blank was encountered successively -> adds to trailing prefix
                        new_prefix = prefix + (c,)
                        next_p_nb_diff = p_b + prob_c
                        if new_prefix not in next_beam:
                            next_beam[new_prefix] =[-float('inf'), -float('inf')]
                        next_beam[new_prefix][1] = logaddexp(next_beam[new_prefix][1], next_p_nb_diff)
                    else:
                        # Append a strictly new character into the prefix array footprint
                        new_prefix = prefix + (c,)
                        next_p_nb = p_total + prob_c
                        if new_prefix not in next_beam:
                            next_beam[new_prefix] = [-float('inf'), -float('inf')]
                        next_beam[new_prefix][1] = logaddexp(next_beam[new_prefix][1], next_p_nb)
                        
            # Apply beam pruning mechanism 
            items = list(next_beam.items())
            items.sort(key=lambda x: logaddexp(x[1][0], x[1][1]), reverse=True)
            beam_state = {k: tuple(v) for k, v in items[:beam]}
            
        best_prefix = max(beam_state.items(), key=lambda x: logaddexp(x[1][0], x[1][1]))[0]
        s = "".join([id2letter.get(p, "") for p in best_prefix])
        results.append(s)
        
    return results