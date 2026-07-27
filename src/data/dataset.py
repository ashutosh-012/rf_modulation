import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.representations import iq_to_ap, iq_to_stft, iq_to_constellation


MOD_CLASSES = [
    "8PSK", "BPSK", "CPFSK", "GFSK",
    "PAM4", "QAM16", "QAM64", "QPSK"
]

MOD_TO_IDX = {mod: i for i, mod in enumerate(MOD_CLASSES)}


class RadioMLDataset(Dataset):
    def __init__(self, X, labels, representation="iq", stftParams=None, constelParams=None):
        self.X = X
        self.labels = labels
        self.representation = representation

        self.stftParams = stftParams or {}
        self.constelParams = constelParams or {}

        self.modLabels = []
        self.snrLabels = []
        for lbl in labels:
            modStr = str(lbl[0])
            snrVal = int(lbl[1])
            self.modLabels.append(modStr)
            self.snrLabels.append(snrVal)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        iqData = self.X[idx]
        modStr = self.modLabels[idx]
        snr = self.snrLabels[idx]

        if self.representation == "iq":
            features = iqData.astype(np.float32)

        elif self.representation == "ap":
            features = iq_to_ap(iqData).astype(np.float32)

        elif self.representation == "stft":
            spectrogram = iq_to_stft(iqData, **self.stftParams)
            features = spectrogram.astype(np.float32)
            if features.ndim == 2:
                features = np.expand_dims(features, axis=0)

        elif self.representation == "constellation":
            constelImg = iq_to_constellation(iqData, **self.constelParams)
            features = constelImg.astype(np.float32)
            if features.ndim == 2:
                features = np.expand_dims(features, axis=0)

        else:
            raise ValueError(f"unknown representation: {self.representation}")

        labelIdx = MOD_TO_IDX.get(modStr, -1)

        featureTensor = torch.tensor(features, dtype=torch.float32)
        labelTensor = torch.tensor(labelIdx, dtype=torch.long)

        return featureTensor, labelTensor, snr


def load_radioml_data(filepath):
    with open(filepath, "rb") as f:
        rawData = pickle.load(f, encoding="bytes")

    modFilter = [b"WBFM", b"AM-DSB", b"AM-SSB"]

    keyMap = {
        b"QAM16": "QAM16",
        b"QAM64": "QAM64",
        b"QPSK": "QPSK",
        b"8PSK": "8PSK",
        b"CPFSK": "CPFSK",
        b"GFSK": "GFSK",
        b"BPSK": "BPSK",
        b"PAM4": "PAM4"
    }

    allX = []
    allLabels = []

    for (modType, snr), samples in rawData.items():
        if modType in modFilter:
            continue

        modName = keyMap.get(modType, modType.decode("utf-8") if isinstance(modType, bytes) else modType)

        for i in range(samples.shape[0]):
            allX.append(samples[i])
            allLabels.append((modName, snr))

    X = np.array(allX)
    labels = np.array(allLabels, dtype=object)

    return X, labels
