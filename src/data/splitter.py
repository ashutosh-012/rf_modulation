import numpy as np
from collections import defaultdict


def snr_stratified_split(X, labels, trainRatio=0.67, valRatio=0.13, testRatio=0.20, seed=42):
    rng = np.random.RandomState(seed)

    snrGroups = defaultdict(list)
    for idx in range(len(labels)):
        modType = labels[idx][0]
        snr = labels[idx][1]
        key = (modType, snr)
        snrGroups[key].append(idx)

    trainIdx = []
    valIdx = []
    testIdx = []

    for key in sorted(snrGroups.keys()):
        indices = np.array(snrGroups[key])
        rng.shuffle(indices)

        n = len(indices)
        nTrain = int(n * trainRatio)
        nVal = int(n * valRatio)

        trainIdx.extend(indices[:nTrain])
        valIdx.extend(indices[nTrain:nTrain + nVal])
        testIdx.extend(indices[nTrain + nVal:])

    trainIdx = np.array(trainIdx)
    valIdx = np.array(valIdx)
    testIdx = np.array(testIdx)

    rng.shuffle(trainIdx)
    rng.shuffle(valIdx)
    rng.shuffle(testIdx)

    splits = {
        "train": {
            "X": X[trainIdx],
            "labels": labels[trainIdx]
        },
        "val": {
            "X": X[valIdx],
            "labels": labels[valIdx]
        },
        "test": {
            "X": X[testIdx],
            "labels": labels[testIdx]
        }
    }

    return splits


def verify_no_leakage(splits):
    trainLabels = set()
    for lbl in splits["train"]["labels"]:
        trainLabels.add((str(lbl[0]), str(lbl[1])))

    valLabels = set()
    for lbl in splits["val"]["labels"]:
        valLabels.add((str(lbl[0]), str(lbl[1])))

    testLabels = set()
    for lbl in splits["test"]["labels"]:
        testLabels.add((str(lbl[0]), str(lbl[1])))

    allCovered = trainLabels == valLabels == testLabels
    return allCovered
