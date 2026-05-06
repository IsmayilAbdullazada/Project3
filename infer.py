#!/usr/bin/env python
# infer.py
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules.dataset import CLSPDataset, clsp_collate
from modules.model import SequenceClassifier
from utils.features import get_mfcc_transform, wav_lengths_to_logit_lengths

# CTC decoders (beam is optional)
from utils.decode import decode_batch_ctc_vocab_minloss, decode_batch_ctc_greedy
try:
    from utils.decode import decode_batch_ctc_beam
    HAS_BEAM = True
except Exception:
    decode_batch_ctc_beam = None
    HAS_BEAM = False


def check_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, help="dev, tst", default="dev")
    parser.add_argument("--checkpoint_path", type=Path, default="best_model.pt")
    parser.add_argument("--output_path", type=Path, default="output.txt")
    parser.add_argument("--beam", type=int, default=4)
    return parser.parse_args()


# --------------------------------------------------
# Config
# --------------------------------------------------
n_mfcc = 15
BATCH_SIZE = 16
mfcc_transform = get_mfcc_transform(n_mfcc)
CTC_BLANK_ID = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def main(args):
    CHECKPOINT_PATH = args.checkpoint_path
    OUTPUT_PATH = args.output_path
    OUTPUT_GREEDY = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".greedy")
    OUTPUT_BEAM = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".beam")

    # --------------------------------------------------
    # Dataset / Loader
    # --------------------------------------------------
    test_dataset = CLSPDataset(subset=args.subset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=clsp_collate
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = SequenceClassifier(
        num_classes=len(test_dataset.scr_letters),
        feat_dim=n_mfcc
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --------------------------------------------------
    # Inference loop
    # --------------------------------------------------
    minloss_words = []
    greedy_seqs = []
    beam_seqs = [] if HAS_BEAM else None

    # for mapping minloss vocab-row -> word string
    vocab_words = [w for w in test_dataset.scr2id.keys() if w != "<unk>"]

    with torch.no_grad():
        for batch in tqdm(test_loader):
            mfcc = mfcc_transform(batch["wavs"]).transpose(1, 2)
            mean = mfcc.mean(dim=1, keepdim=True)
            std = mfcc.std(dim=1, keepdim=True)
            mfcc = (mfcc - mean) / (std + 1e-8)

            logits = model(mfcc.to(device))  # (B,T,V)
            logit_lengths = wav_lengths_to_logit_lengths(batch["wav_lengths"]).to(device)

            # 1) min CTC loss over vocabulary (classification)
            best_vocab_row = decode_batch_ctc_vocab_minloss(logits, logit_lengths, test_dataset)  # (B,)
            for i in best_vocab_row.tolist():
                minloss_words.append(vocab_words[i])

            # 2) greedy decoded sequence
            greedy_out = decode_batch_ctc_greedy(
                logits, logit_lengths, test_dataset, blank_id=CTC_BLANK_ID
            )
            greedy_seqs.extend(greedy_out)

            # 3) beam decoded sequence (optional)
            if HAS_BEAM:
                beam_out = decode_batch_ctc_beam(
                    logits, logit_lengths, test_dataset, beam=args.beam, blank_id=CTC_BLANK_ID
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
    else:
        print("Beam decoder not available; skipping beam output.")


if __name__ == "__main__":
    args = check_argv()
    main(args)