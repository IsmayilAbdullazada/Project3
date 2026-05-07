#!/usr/bin/env python
# infer.py
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.dataset import CLSPDataset, clsp_collate
from modules.model import SequenceClassifier
from utils.features import (
    get_mfcc_transform,
    get_delta_transform,
    extract_features,
    wav_lengths_to_logit_lengths,
)
from utils.decode import decode_batch_ctc_vocab_minloss, decode_batch_ctc_greedy

try:
    from utils.decode import decode_batch_ctc_beam
    HAS_BEAM = True
except Exception:
    decode_batch_ctc_beam = None
    HAS_BEAM = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset",          type=str,  default="dev")
    parser.add_argument("--checkpoint_path", type=Path, default="best_model.pt")
    parser.add_argument("--output_path",     type=Path, default="output.txt")
    parser.add_argument("--beam",            type=int,  default=4)
    return parser.parse_args()


# --------------------------------------------------
# Default config (overridden by checkpoint metadata if present)
# --------------------------------------------------
DEFAULT_N_MFCC   = 40
DEFAULT_FEAT_DIM = DEFAULT_N_MFCC * 3   # 120
BATCH_SIZE = 32
CTC_BLANK_ID = 0
eps = 1e-8


def main(args):
    CHECKPOINT_PATH = args.checkpoint_path
    OUTPUT_PATH     = args.output_path
    OUTPUT_GREEDY   = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".greedy")
    OUTPUT_BEAM     = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".beam")

    # --------------------------------------------------
    # Load checkpoint and recover training config
    # --------------------------------------------------
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    n_mfcc   = checkpoint.get("n_mfcc",    DEFAULT_N_MFCC)
    feat_dim = checkpoint.get("feat_dim",  DEFAULT_FEAT_DIM)
    print(f"Checkpoint config: n_mfcc={n_mfcc}, feat_dim={feat_dim}")

    mfcc_transform  = get_mfcc_transform(n_mfcc).to(device)
    delta_transform = get_delta_transform().to(device)

    # --------------------------------------------------
    # Dataset / Loader
    # --------------------------------------------------
    test_dataset = CLSPDataset(subset=args.subset)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=clsp_collate,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = SequenceClassifier(
        num_classes=len(test_dataset.scr_letters),
        feat_dim=feat_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --------------------------------------------------
    # Inference loop
    # --------------------------------------------------
    minloss_words = []
    greedy_seqs   = []
    beam_seqs     = [] if HAS_BEAM else None

    vocab_words = [w for w in test_dataset.scr2id.keys() if w != "<unk>"]

    with torch.no_grad():
        for batch in tqdm(test_loader):
            wavs  = batch["wavs"].to(device)
            feats = extract_features(wavs, mfcc_transform, delta_transform, device)
            mean  = feats.mean(dim=1, keepdim=True)
            std   = feats.std(dim=1, keepdim=True)
            feats = (feats - mean) / (std + eps)

            logits        = model(feats)
            logit_lengths = wav_lengths_to_logit_lengths(batch["wav_lengths"]).to(device)

            # 1) min CTC loss over vocabulary
            best_vocab_row = decode_batch_ctc_vocab_minloss(
                logits, logit_lengths, test_dataset
            )
            for i in best_vocab_row.tolist():
                minloss_words.append(vocab_words[i])

            # 2) greedy decoded sequence
            greedy_out = decode_batch_ctc_greedy(
                logits, logit_lengths, test_dataset, blank_id=CTC_BLANK_ID
            )
            greedy_seqs.extend(greedy_out)

            # 3) beam decoded sequence
            if HAS_BEAM:
                beam_out = decode_batch_ctc_beam(
                    logits, logit_lengths, test_dataset,
                    beam=args.beam, blank_id=CTC_BLANK_ID
                )
                beam_seqs.extend(beam_out)

    # --------------------------------------------------
    # Write outputs
    # --------------------------------------------------
    with open(OUTPUT_PATH, "w") as f:
        for w in minloss_words:
            f.write(f"{w}\n")
    print(f"Wrote min-CTC-loss vocab predictions to {OUTPUT_PATH}")

    with open(OUTPUT_GREEDY, "w") as f:
        for s in greedy_seqs:
            f.write(f"{s}\n")
    print(f"Wrote greedy CTC sequences to {OUTPUT_GREEDY}")

    if HAS_BEAM:
        with open(OUTPUT_BEAM, "w") as f:
            for s in beam_seqs:
                f.write(f"{s}\n")
        print(f"Wrote beam CTC sequences to {OUTPUT_BEAM}")


if __name__ == "__main__":
    args = check_argv()
    main(args)