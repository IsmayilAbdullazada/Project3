import torch
import torchaudio


def get_mfcc_transform(n_mfcc):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 400,        # 25 ms window
            "win_length": 400,
            "hop_length": 320,   # 20 ms stride
            "center": False,     # IMPORTANT for alignment
            "n_mels": 80,
            "power": 2.0,
        },
    )
    return mfcc_transform


def get_delta_transform():
    """Returns a ComputeDeltas transform."""
    return torchaudio.transforms.ComputeDeltas(win_length=5, mode="replicate")


def extract_features(wavs, mfcc_transform, delta_transform, device):
    """
    Extract MFCC + delta + delta² features.

    Args:
        wavs:             (B, T_wav) waveform tensor
        mfcc_transform:   torchaudio MFCC transform  (on same device)
        delta_transform:  torchaudio ComputeDeltas    (on same device)
        device:           torch.device

    Returns:
        features: (B, T_frames, 3 * n_mfcc)  — static + Δ + ΔΔ
    """
    mfcc   = mfcc_transform(wavs)          # (B, n_mfcc, T)
    delta  = delta_transform(mfcc)         # (B, n_mfcc, T)
    delta2 = delta_transform(delta)        # (B, n_mfcc, T)
    features = torch.cat([mfcc, delta, delta2], dim=1)  # (B, 3*n_mfcc, T)
    return features.transpose(1, 2)        # (B, T, 3*n_mfcc)


def wav_lengths_to_logit_lengths(
    wav_lengths,
    n_fft=400,
    hop_length=320,
):
    """
    Convert waveform sample lengths → MFCC/logit frame lengths
    (center=False)
    """
    logit_lengths = ((wav_lengths - n_fft) // hop_length) + 1
    logit_lengths = torch.clamp(logit_lengths, min=0)
    return logit_lengths