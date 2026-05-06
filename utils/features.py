import torch, torchaudio

def get_mfcc_transform(n_mfcc):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": 400,        # 25 ms window
            "win_length": 400,
            "hop_length": 320,   # 20 ms stride (wav2vec2)
            "center": False,     # IMPORTANT for alignment
            "n_mels": 80,
            "power": 2.0,
        },
    )
    return mfcc_transform

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