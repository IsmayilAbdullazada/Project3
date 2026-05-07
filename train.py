import time
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.dataset import CLSPDataset, clsp_collate
from modules.model import SequenceClassifier
from modules.loss import ctc_loss_from_logits
from utils.features import get_mfcc_transform, wav_lengths_to_logit_lengths

from utils.decode import (
    decode_batch_ctc_greedy,
    decode_batch_ctc_beam,
)

# -------------------------------------------------
# Config
# -------------------------------------------------

n_mfcc = 15
BATCH_SIZE = 16
LR = 0.001
MAX_EPOCHS = 100000
TIME_LIMIT = 20 * 60
eps = 1e-8

CHECKPOINT_PATH = Path("best_model.pt")
mfcc_transform = get_mfcc_transform(n_mfcc).to(device)

CTC_BLANK_ID = 0
CTC_TARGET_PAD_ID = CTC_BLANK_ID
BEAM_SIZE = 4

# -------------------------------------------------
# Device
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# Dataset / Loader
# -------------------------------------------------

train_dataset = CLSPDataset(subset="trn")
dev_dataset = CLSPDataset(subset="dev")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=clsp_collate,
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=clsp_collate,
)

# -------------------------------------------------
# Model
# -------------------------------------------------

model = SequenceClassifier(
    num_classes=len(train_dataset.scr_letters),
    feat_dim=n_mfcc,
).to(device)

optimizer = optim.SGD(model.parameters(), lr=LR)

# -------------------------------------------------
# CER helper
# -------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = cur
    return dp[m]

def _cer(pred: str, ref: str) -> tuple[int, int]:
    pred = pred.replace("|", "")
    ref = ref.replace("|", "")
    edits = _levenshtein(pred, ref)
    return edits, max(1, len(ref))

# -------------------------------------------------
# Training
# -------------------------------------------------

def train_epoch():
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(tqdm(train_loader)):
        wavs = batch["wavs"].to(device)
        mfcc = mfcc_transform(wavs).transpose(1, 2)
        
        # DEBUG: Check for NaN in input
        if torch.isnan(mfcc).any():
            print(f"NaN in MFCC at batch {i}")
            continue
            
        mean = mfcc.mean(dim=1, keepdim=True)
        std = mfcc.std(dim=1, keepdim=True)
        mfcc = (mfcc - mean) / (std + eps)
        
        # DEBUG: Check for NaN after normalization  
        if torch.isnan(mfcc).any():
            print(f"NaN after normalization at batch {i}")
            print(f"std min: {std.min()}, std max: {std.max()}")
            continue

        logits = model(mfcc.to(device))
        
        # DEBUG: Check logits
        if torch.isnan(logits).any():
            print(f"NaN in logits at batch {i}")
            continue
        logit_lengths = wav_lengths_to_logit_lengths(batch["wav_lengths"]).to(device)

        targets = batch["letters"].to(device)  # (B, K)
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
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0

    greedy_edits = 0
    greedy_chars = 0
    beam_edits = 0
    beam_chars = 0

    # precompute once
    id2letter = {i: ch for ch, i in dev_dataset.letter2id.items()}

    for batch in dev_loader:
        wavs = batch["wavs"].to(device)
        mfcc = mfcc_transform(wavs).transpose(1, 2)
        mean = mfcc.mean(dim=1, keepdim=True)
        std = mfcc.std(dim=1, keepdim=True)
        mfcc = (mfcc - mean) / (std + eps)

        logits = model(mfcc.to(device))  # (B, T, N)
        logit_lengths = wav_lengths_to_logit_lengths(batch["wav_lengths"]).to(device)

        targets = batch["letters"].to(device)  # (B, K)
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

        # -------------------------
        # CER: greedy
        # -------------------------
        try:
            greedy_seqs = decode_batch_ctc_greedy(
                logits, logit_lengths, dev_dataset, blank_id=CTC_BLANK_ID
            )
        except Exception:
            print("Greedy search not implemented!")
            greedy_seqs = [""] * targets.size(0)

        for b in range(targets.size(0)):
            ref_ids = targets[b, : target_lengths[b]].tolist()
            ref_seq = "".join(id2letter.get(int(i), "") for i in ref_ids)

            pred_seq = greedy_seqs[b] if (b < len(greedy_seqs) and isinstance(greedy_seqs[b], str)) else ""
            e, c = _cer(pred_seq, ref_seq)
            greedy_edits += e
            greedy_chars += c

        # -------------------------
        # CER: beam
        # -------------------------
        try:
            beam_seqs = decode_batch_ctc_beam(
                logits, logit_lengths, dev_dataset, beam=BEAM_SIZE, blank_id=CTC_BLANK_ID
            )
        except Exception:
            print("Beam search not implemented!")
            beam_seqs = [""] * targets.size(0)

        for b in range(targets.size(0)):
            ref_ids = targets[b, : target_lengths[b]].tolist()
            ref_seq = "".join(id2letter.get(int(i), "") for i in ref_ids)

            pred_seq = beam_seqs[b] if (b < len(beam_seqs) and isinstance(beam_seqs[b], str)) else ""
            e, c = _cer(pred_seq, ref_seq)
            beam_edits += e
            beam_chars += c

    avg_loss = total_loss / len(dev_loader)
    greedy_cer = greedy_edits / greedy_chars if greedy_chars > 0 else 0.0
    beam_cer = beam_edits / beam_chars if beam_chars > 0 else 0.0
    return avg_loss, greedy_cer, beam_cer

# -------------------------------------------------
# Main Loop
# -------------------------------------------------

best_greedy_cer = float("inf")
start_time = time.time()
epoch = 0

while epoch < MAX_EPOCHS:
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        print("\nTime limit reached")
        break

    epoch += 1
    train_loss = train_epoch()
    dev_loss, greedy_cer, beam_cer = evaluate()

    print(f"Epoch {epoch:04d} | train_loss={train_loss:.4f} | time={elapsed/60:.2f}m")
    print(f"dev_loss={dev_loss:.4f}")
    print(f"dev_CER_greedy={greedy_cer:.4f}")
    print(f"dev_CER_beam={BEAM_SIZE}={beam_cer:.4f}")

    # checkpoint on *lower* greedy CER
    if greedy_cer < best_greedy_cer:
        best_greedy_cer = greedy_cer
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dev_loss": dev_loss,
                "dev_greedy_cer": greedy_cer,
                "epoch": epoch,
            },
            CHECKPOINT_PATH,
        )
        print("Saved new best model (by greedy CER)")

print("\nTraining finished.")
print("Best dev greedy CER:", best_greedy_cer)
