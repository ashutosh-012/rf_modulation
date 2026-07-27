import numpy as np
import torch


def iq_to_ap(iqData):
    i_samples = iqData[0]
    q_samples = iqData[1]

    amplitude = np.sqrt(i_samples ** 2 + q_samples ** 2)
    phase = np.arctan2(q_samples, i_samples)

    apData = np.stack([amplitude, phase], axis=0)
    return apData


def iq_to_stft(iqData, nfft=64, hopLen=4, winLen=None):
    i_samples = iqData[0]
    q_samples = iqData[1]
    complexSig = i_samples + 1j * q_samples

    sig_tensor = torch.tensor(complexSig, dtype=torch.complex64)

    if winLen is None:
        winLen = nfft

    window = torch.hann_window(winLen)

    stftOut = torch.stft(
        sig_tensor,
        n_fft=nfft,
        hop_length=hopLen,
        win_length=winLen,
        window=window,
        return_complex=True
    )

    magSpec = torch.abs(stftOut)
    magSpec = magSpec.numpy()

    return magSpec


def iq_to_constellation(iqData, nBins=64, xyRange=None):
    i_samples = iqData[0]
    q_samples = iqData[1]

    if xyRange is None:
        maxVal = max(np.abs(i_samples).max(), np.abs(q_samples).max())
        xyRange = maxVal * 1.1

    counts, _, _ = np.histogram2d(
        i_samples,
        q_samples,
        bins=nBins,
        range=[[-xyRange, xyRange], [-xyRange, xyRange]]
    )

    if counts.max() > 0:
        counts = counts / counts.max()

    return counts
