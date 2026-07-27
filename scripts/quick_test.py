import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

print("=== testing representations ===")
from src.data.representations import iq_to_ap, iq_to_stft, iq_to_constellation

fakeIQ = np.random.randn(2, 128).astype(np.float32)

ap = iq_to_ap(fakeIQ)
print(f"iq_to_ap: input {fakeIQ.shape} -> output {ap.shape}")
assert ap.shape == (2, 128), f"expected (2,128) got {ap.shape}"

spec = iq_to_stft(fakeIQ, nfft=64, hopLen=4)
print(f"iq_to_stft: input {fakeIQ.shape} -> output {spec.shape}")
assert spec.ndim == 2, f"expected 2d got {spec.ndim}d"

constel = iq_to_constellation(fakeIQ, nBins=64)
print(f"iq_to_constellation: input {fakeIQ.shape} -> output {constel.shape}")
assert constel.shape == (64, 64), f"expected (64,64) got {constel.shape}"

print("\n=== testing splitter ===")
from src.data.splitter import snr_stratified_split

fakeX = np.random.randn(800, 2, 128)
fakeLabels = []
mods = ["BPSK", "QPSK", "8PSK", "QAM16"]
snrs = [-10, -5, 0, 5, 10]
for m in mods:
    for s in snrs:
        for _ in range(40):
            fakeLabels.append((m, s))
fakeLabels = np.array(fakeLabels, dtype=object)

splits = snr_stratified_split(fakeX, fakeLabels)
print(f"train: {len(splits['train']['X'])}, val: {len(splits['val']['X'])}, test: {len(splits['test']['X'])}")
totalAfterSplit = len(splits['train']['X']) + len(splits['val']['X']) + len(splits['test']['X'])
assert totalAfterSplit == 800, f"lost samples: {totalAfterSplit} vs 800"

print("\n=== testing models forward pass ===")
from src.models.basic_cnn import BasicCNN
from src.models.resnet1d import ResNet1D
from src.models.cldnn import CLDNN
from src.models.cldnn_se import CLDNN_SE
from src.models.lstm_model import LSTMClassifier

batchInput = torch.randn(4, 2, 128)

models1d = {
    "BasicCNN": BasicCNN(numClasses=8),
    "ResNet1D": ResNet1D(numClasses=8),
    "CLDNN": CLDNN(numClasses=8),
    "CLDNN_SE": CLDNN_SE(numClasses=8),
    "LSTM": LSTMClassifier(numClasses=8),
}

for name, model in models1d.items():
    model.eval()
    with torch.no_grad():
        out = model(batchInput)
    print(f"{name}: input {batchInput.shape} -> output {out.shape}")
    assert out.shape == (4, 8), f"{name} wrong output shape: {out.shape}"

print("\n=== testing dataset class ===")
from src.data.dataset import RadioMLDataset

ds = RadioMLDataset(fakeX, fakeLabels, representation="iq")
feat, lbl, snr = ds[0]
print(f"dataset[0]: feat {feat.shape}, label {lbl.item()}, snr {snr}")

dsAP = RadioMLDataset(fakeX, fakeLabels, representation="ap")
feat2, lbl2, snr2 = dsAP[0]
print(f"dataset ap[0]: feat {feat2.shape}, label {lbl2.item()}, snr {snr2}")

dsSTFT = RadioMLDataset(fakeX, fakeLabels, representation="stft")
feat3, lbl3, snr3 = dsSTFT[0]
print(f"dataset stft[0]: feat {feat3.shape}, label {lbl3.item()}, snr {snr3}")

print("\n=== all tests passed ===")
