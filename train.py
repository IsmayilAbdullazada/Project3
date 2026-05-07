import time
from pathlib import Path
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules.dataset import CLSPDataset, clsp_collate
from modules.model import SequenceClassifier
from modules.loss import ctc_loss_from_logits
from utils.features import (
    get_mfcc_transform,
    get_delta_transform,
    extract_features,
    wav_lengths_to_logit_lengths,
)
from utils.decode import (
    decode_batch_ctc_greedy,
    decode_batch_ctc_beam,
)

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# Config
# -------------------------------------------------
n_mfcc      = 40
FEAT_DIM    = n_mfcc * 3       # static + delta + delta²  = 120
BATCH_SIZE  = 32
LR          = 3e-4             # slightly lower learning rate
MAX_EPOCHS  = 100000
TIME_LIMIT  = 20 * 60
GRAD_CLIP   = 1.0              # more aggressive clipping
WEIGHT_DECAY = 1e-3            # stronger weight decay
eps         = 1e-8

# Early stopping config
PATIENCE = 10                  # stop if no improvement for this many epochs
MIN_DELTA = 0.001              # minimum improvement to count

# Data augmentation config
SPEC_AUGMENT = True
FREQ_MASK_PARAM = 10           # max frequency mask width
TIME_MASK_PARAM = 5            # max time mask width
NUM_FREQ_MASKS = 1
NUM_TIME_MASKS = 1

CHECKPOINT_PATH  = Path("best_model.pt")
CTC_BLANK_ID     = 0
CTC_TARGET_PAD_ID = CTC_BLANK_ID
BEAM_SIZE        = 4

mfcc_transform  = get_mfcc_transform(n_mfcc).to(device)
delta_transform = get_delta_transform().to(device)

# -------------------------------------------------
# Dataset / Loader
# -------------------------------------------------
train_dataset = CLSPDataset(subset="trn")
dev_dataset   = CLSPDataset(subset="dev")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=clsp_collate,
    num_workers=0,
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=clsp_collate,
    num_workers=0,
)

# -------------------------------------------------
# Model
# -------------------------------------------------
model = SequenceClassifier(
    num_classes=len(train_dataset.scr_letters),
    feat_dim=FEAT_DIM,
).to(device)

# Adam with stronger weight decay
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

# ReduceLROnPlateau - reduces LR when validation loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
)

# -------------------------------------------------
# Data Augmentation: SpecAugment
# -------------------------------------------------
def spec_augment(features, freq_mask_param=10, time_mask_param=5, 
                 num_freq_masks=1, num_time_masks=1):
    """
    Apply SpecAugment to features (B, T, F).
    """
    B, T, F = features.shape
    augmented = features.clone()
    
    for b in range(B):
        # Frequency masking
        for _ in range(num_freq_masks):
            f = torch.randint(0, min(freq_mask_param, F) + 1, (1,)).item()
            if f > 0 and F > f:
                f0 = torch.randint(0, F - f, (1,)).item()
                augmented[b, :, f0:f0 + f] = 0
        
        # Time masking
        for _ in range(num_time_masks):
            t = torch.randint(0, min(time_mask_param, T) + 1, (1,)).item()
            if t > 0 and T > t:
                t0 = torch.randint(0, T - t, (1,)).item()
                augmented[b, t0:t0 + t, :] = 0
    
    return augmented


def add_noise(features, noise_level=0.1):
    """Add Gaussian noise to features."""
    noise = torch.randn_like(features) * noise_level
    return features + noise


# -------------------------------------------------
# CER helper
# -------------------------------------------------
def _levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def _cer(pred: str, ref: str) -> tuple[int, int]:
    pred = pred.replace("|", "")
    ref  = ref.replace("|", "")
    edits = _levenshtein(pred, ref)
    return edits, max(1, len(ref))


# -------------------------------------------------
# Feature extraction helper
# -------------------------------------------------
def get_features(wavs, training=False):
    """Extract normalised MFCC+delta+delta² features with optional augmentation."""
    feats = extract_features(wavs, mfcc_transform, delta_transform, device)
    mean  = feats.mean(dim=1, keepdim=True)
    std   = feats.std(dim=1, keepdim=True)
    feats = (feats - mean) / (std + eps)
    
    # Apply augmentation only during training
    if training and SPEC_AUGMENT:
        feats = spec_augment(
            feats, 
            freq_mask_param=FREQ_MASK_PARAM,
            time_mask_param=TIME_MASK_PARAM,
            num_freq_masks=NUM_FREQ_MASKS,
            num_time_masks=NUM_TIME_MASKS
        )
        # Optional: add small noise
        if torch.rand(1).item() < 0.3:  # 30% chance
            feats = add_noise(feats, noise_level=0.05)
    
    return feats


# -------------------------------------------------
# Label smoothing for CTC
# -------------------------------------------------
def smooth_ctc_loss(logits, targets, input_lengths, target_lengths, 
                    blank_id, smoothing=0.1):
    """
    CTC loss with label smoothing effect via confidence penalty.
    """
    base_loss = ctc_loss_from_logits(
        logits_btn=logits,
        targets_bk=targets,
        input_lengths_b=input_lengths,
        target_lengths_b=target_lengths,
        blank_id=blank_id,
        target_pad_id=CTC_TARGET_PAD_ID,
        reduction="mean",
        zero_infinity=True,
    )
    
    # Confidence penalty: penalize very confident predictions
    log_probs = torch.log_softmax(logits, dim=-1)
    confidence_penalty = -smoothing * log_probs.mean()
    
    return base_loss + confidence_penalty


# -------------------------------------------------
# Training
# -------------------------------------------------
def train_epoch():
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(tqdm(train_loader)):
        wavs = batch["wavs"].to(device)
        feats = get_features(wavs, training=True)  # with augmentation

        if torch.isnan(feats).any():
            print(f"NaN in features at batch {i}; skipping.")
            continue

        logits = model(feats)

        if torch.isnan(logits).any():
            print(f"NaN in logits at batch {i}; skipping.")
            continue

        logit_lengths  = wav_lengths_to_logit_lengths(batch["wav_lengths"]).to(device)
        targets        = batch["letters"].to(device)
        target_lengths = (targets != CTC_TARGET_PAD_ID).sum(dim=1).to(device)

        loss = ctc_loss_from_logits(
            logits_btn=logits,
            targets_bk=targets,
            input_lengths_b=logit_lengths,
            target_lengths_b=target_lengths,
            blank_id=CTC_BLANK_ID,
            target_pad_id=CTC_TARGET_PAD_ID,
            reduction="mean",
            zero_infinity=True,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(train_loader), 1)


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
@torch.no_grad()
def evaluate():
    model.eval()
    total_loss   = 0.0
    greedy_edits = 0
    greedy_chars = 0
    beam_edits   = 0
    beam_chars   = 0

    id2letter = {i: ch for ch, i in dev_dataset.letter2id.items()}

    for batch in dev_loader:
        wavs  = batch["wavs"].to(device)
        feats = get_features(wavs, training=False)  # no augmentation

        logits         = model(feats)
        logit_lengths  = wav_lengths_to_logit_lengths(batch["wav_lengths"]).to(device)
        targets        = batch["letters"].to(device)
        target_lengths = (targets != CTC_TARGET_PAD_ID).sum(dim=1).to(device)

        loss = ctc_loss_from_logits(
            logits_btn=logits,
            targets_bk=targets,
            input_lengths_b=logit_lengths,
            target_lengths_b=target_lengths,
            blank_id=CTC_BLANK_ID,
            target_pad_id=CTC_TARGET_PAD_ID,
            reduction="mean",
            zero_infinity=True,
        )
        total_loss += loss.item()

        # ---- greedy CER ----
        try:
            greedy_seqs = decode_batch_ctc_greedy(
                logits, logit_lengths, dev_dataset, blank_id=CTC_BLANK_ID
            )
        except Exception:
            greedy_seqs = [""] * targets.size(0)

        for b in range(targets.size(0)):
            ref_ids = targets[b, :target_lengths[b]].tolist()
            ref_seq = "".join(id2letter.get(int(i), "") for i in ref_ids)
            pred    = greedy_seqs[b] if b < len(greedy_seqs) else ""
            e, c    = _cer(pred, ref_seq)
            greedy_edits += e
            greedy_chars += c

        # ---- beam CER ----
        try:
            beam_seqs = decode_batch_ctc_beam(
                logits, logit_lengths, dev_dataset,
                beam=BEAM_SIZE, blank_id=CTC_BLANK_ID
            )
        except Exception:
            beam_seqs = [""] * targets.size(0)

        for b in range(targets.size(0)):
            ref_ids = targets[b, :target_lengths[b]].tolist()
            ref_seq = "".join(id2letter.get(int(i), "") for i in ref_ids)
            pred    = beam_seqs[b] if b < len(beam_seqs) else ""
            e, c    = _cer(pred, ref_seq)
            beam_edits += e
            beam_chars += c

    avg_loss   = total_loss / len(dev_loader)
    greedy_cer = greedy_edits / greedy_chars if greedy_chars > 0 else 0.0
    beam_cer   = beam_edits   / beam_chars   if beam_chars   > 0 else 0.0
    return avg_loss, greedy_cer, beam_cer


# -------------------------------------------------
# Main Loop with Early Stopping
# -------------------------------------------------
best_greedy_cer = float("inf")
best_dev_loss   = float("inf")
start_time      = time.time()
epoch           = 0
patience_counter = 0

while epoch < MAX_EPOCHS:
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        print("\nTime limit reached")
        break

    epoch += 1
    train_loss = train_epoch()
    dev_loss, greedy_cer, beam_cer = evaluate()

    # Step the scheduler based on dev loss
    scheduler.step(dev_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:04d} | train_loss={train_loss:.4f} | "
          f"lr={current_lr:.2e} | time={elapsed/60:.2f}m")
    print(f"dev_loss={dev_loss:.4f}")
    print(f"dev_CER_greedy={greedy_cer:.4f}")
    print(f"dev_CER_beam={BEAM_SIZE}={beam_cer:.4f}")

    # Check for improvement
    improved = False
    if greedy_cer < best_greedy_cer - MIN_DELTA:
        best_greedy_cer = greedy_cer
        improved = True
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dev_loss": dev_loss,
                "dev_greedy_cer": greedy_cer,
                "epoch": epoch,
                "n_mfcc": n_mfcc,
                **model.config,  # save model config for later inference
            },
            CHECKPOINT_PATH,
        )
        print("Saved new best model (by greedy CER)")
    
    # Early stopping check
    if improved:
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {PATIENCE} epochs without improvement")
            break

    # Also stop if training loss is very low but dev loss keeps increasing
    if train_loss < 0.01 and dev_loss > best_dev_loss * 1.5:
        print("\nStopping due to severe overfitting detected")
        break
    
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss

print("\nTraining finished.")
print("Best dev greedy CER:", best_greedy_cer)
