import json
from pathlib import Path
import random

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def alignment_to_indices_fast(alignments, token2id):
    ends = torch.tensor(alignments[1::2])
    starts = torch.cat([torch.tensor([0]), ends[:-1] + 1])

    T = ends[-1].item() + 1
    frames = torch.empty(T, dtype=torch.long)

    for token, s, e in zip(alignments[::2], starts, ends):
        idx = token2id.get(token.lower(), token2id["<unk>"])
        frames[s:e+1] = idx

    return frames

def make_ctc_vocab(base_letters):
    blank = "<ctc_blank>"
    unk = "<unk>"

    letters = list(base_letters)

    if "|" not in letters:
        letters.append("|")

    letters = [t for t in letters if t not in (blank, unk)]

    # blank must be 0
    letters = [blank] + letters + [unk]
    return letters


class CLSPDataset(Dataset):
    def __init__(
        self,
        subset,          # "trn" or "dev"
        seed=0,
    ):
        assert subset in ["trn", "dev", "tst"]
        self.subset = subset

        # -------------------------------------------------
        # Always load TRN files
        # -------------------------------------------------
        lbls_path = Path("data/clsp." + subset + "lbls")
        scr_path = Path("data/clsp." + subset + "scr")
        wav_path = Path("data/clsp." + subset + "wav")

        # ---- vocabularies ----
        with open('data/lbl_vocab.json') as f:
            self.lbl_vocab = json.load(f)

        with open('data/scr_vocab.json') as f:
            self.scr_vocab = json.load(f)

        with open('data/scr_letters.json') as f:
            self.scr_letters = json.load(f)

        if subset != 'tst':
            with open('data/' + subset + '_alignments.json') as f:
                self.alignments_raw = json.load(f)

        self.lbl_vocab.append("<unk>")
        self.scr_vocab.append("<unk>")
        self.scr_letters = make_ctc_vocab(self.scr_letters)

        self.lbl2id = {t: i for i, t in enumerate(self.lbl_vocab)}
        self.scr2id = {t: i for i, t in enumerate(self.scr_vocab)}
        self.letter2id = {t: i for i, t in enumerate(self.scr_letters)}
        self.ctc_blank_id = self.letter2id["<ctc_blank>"]
        self.ctc_unk_id = self.letter2id["<unk>"]

        # -------------------------------------------------
        # Read aligned data
        # -------------------------------------------------
        with open(lbls_path) as f:
            self.inputs = [l.strip() for l in f if "jhucsp" not in l]

        with open(scr_path) as f:
            self.targets = [l.strip() for l in f if "jhucsp" not in l]

        with open(wav_path) as f:
            self.wavs = [
                Path("data/wav") / subset / l.strip()
                for l in f if "jhucsp" not in l
            ]
        if subset != 'tst':
            with open(wav_path) as f:
                self.alignments = [
                    self.alignments_raw[l.strip()]
                    for l in f if "jhucsp" not in l
                ]
                

        assert len(self.inputs) == len(self.targets) == len(self.wavs)

    # -------------------------------------------------

    def encode_labels(self, line):
        tokens = line.split()
        ids = [
            self.lbl2id.get(tok, self.lbl2id["<unk>"])
            for tok in tokens
        ]
        return torch.tensor(ids, dtype=torch.long)

    def encode_transcript(self, line):
        return torch.tensor(
            self.scr2id.get(line, self.scr2id["<unk>"]),
            dtype=torch.long,
        )

    def encode_transcript_letters(self, line):
        letters = [
            self.letter2id.get(letter, self.letter2id["<unk>"])
            for letter in list('|' + line + '|')
        ]
        return torch.tensor(
            letters,
            dtype=torch.long,
        )

    # -------------------------------------------------

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.encode_labels(self.inputs[idx])
        y = self.encode_transcript(self.targets[idx])
        letters = self.encode_transcript_letters(self.targets[idx])

        wav, sr = torchaudio.load(self.wavs[idx])
        if self.subset != 'tst':
            alignments = self.alignments[idx]
            letter_targets = alignment_to_indices_fast(alignments, self.letter2id)
        else:
            alignments = None
            letter_targets = None

        return {
            "input_ids": x,
            "label": y,
            "input_length": len(x),
            "wav": wav,
            "wav_length": wav.shape[1],
            "alignments": alignments,
            "letters": letters,
            "letter_targets": letter_targets
        }

def clsp_collate(batch):

    inputs = [b["input_ids"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])

    wavs = [b["wav"].squeeze(0) for b in batch]

    input_lengths = torch.tensor([b["input_length"] for b in batch])
    wav_lengths = torch.tensor([b["wav_length"] for b in batch])

    if batch[0]["letter_targets"] is not None:
        letter_targets = [b["letter_targets"] for b in batch]
        letter_target_lengths = torch.tensor([b["letter_targets"].shape[0] for b in batch])
        blank_id = batch[0].get("ctc_blank_id", 0)
        letter_targets = pad_sequence(letter_targets, batch_first=True, padding_value=blank_id)
    else:
        letter_targets = None
        letter_target_lengths = None

    letters = [b["letters"] for b in batch]
    letter_lengths = torch.tensor([len(x) for x in letters], dtype=torch.long)
    letters = pad_sequence(letters, batch_first=True, padding_value=0)  # 0 = CTC blank (if you set it so)

    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    wavs = pad_sequence(wavs, batch_first=True, padding_value=0)

    return {
        "input_ids": inputs,
        "labels": labels,
        "input_lengths": input_lengths,
        "wavs": wavs,
        "wav_lengths": wav_lengths,
        "letters": letters,
        "ctc_blank_id": 0,
        "letter_lengths": letter_lengths,
        "letter_targets": letter_targets,
        "letter_target_lengths": letter_target_lengths,
    }

if __name__ == "__main__":
    trn = CLSPDataset(
        subset="trn",
    )

    dev = CLSPDataset(
        subset="dev",
    )

    print("Train size:", len(trn))
    print("Dev size:", len(dev))
    print(trn[0])