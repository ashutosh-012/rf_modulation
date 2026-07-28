import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

def compute_per_snr_accuracy(allPreds, allLabels, allSnrs):
    snrAccuracy = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, label, snr in zip(allPreds, allLabels, allSnrs):
        snrAccuracy[snr]["total"] += 1
        if pred == label:
            snrAccuracy[snr]["correct"] += 1

    results = {}
    for snr in sorted(snrAccuracy.keys()):
        total = snrAccuracy[snr]["total"]
        correct = snrAccuracy[snr]["correct"]
        acc = correct / total if total > 0 else 0.0
        results[snr] = acc

    return results

def get_confusion_matrix(allPreds, allLabels, classNames):
    cm = confusion_matrix(allLabels, allPreds)
    return cm

def get_classification_report(allPreds, allLabels, classNames):
    report = classification_report(
        allLabels,
        allPreds,
        target_names=classNames,
        output_dict=True
    )
    return report

def compute_overall_accuracy(allPreds, allLabels):
    correct = sum(1 for p, l in zip(allPreds, allLabels) if p == l)
    total = len(allPreds)
    return correct / total if total > 0 else 0.0
