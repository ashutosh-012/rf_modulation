import os
import sys
import pickle
import tarfile
import urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.splitter import snr_stratified_split

DATASET_URL = "https://zenodo.org/records/10603774/files/RML2016.10a_dict.pkl?download=1"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_FILE = os.path.join(DATA_DIR, "RML2016.10a_dict.pkl")
SPLIT_FILE = os.path.join(DATA_DIR, "radioml_splits.pkl")

def download_dataset():
    if os.path.exists(RAW_FILE):
        print(f"dataset already exists at {RAW_FILE}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print("downloading RadioML 2016.10A from Kaggle...")
    
    try:
        import subprocess
        
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "python-dotenv", "--quiet"], check=True)
        
        from dotenv import load_dotenv
        envPath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        load_dotenv(envPath)
        
        import kaggle
        kaggle.api.authenticate()
        
        print("running kaggle api dataset download...")
        kaggle.api.dataset_download_cli("nolasthitnotomorrow/radioml2016-deepsigcom", path=DATA_DIR, unzip=True)
        
        if not os.path.exists(RAW_FILE):
            print(f"Warning: Kaggle download succeeded but {RAW_FILE} was not found.")
            
            for file in os.listdir(DATA_DIR):
                if file.endswith(".pkl"):
                    os.rename(os.path.join(DATA_DIR, file), RAW_FILE)
                    print(f"Renamed downloaded file to {RAW_FILE}")
                    break
        else:
            print("\ndownload complete")
            
    except Exception as e:
        print(f"\nKaggle download failed: {e}")
        print("Please ensure your Kaggle credentials are correct in the .env file")
        sys.exit(1)

def _progress(blockNum, blockSize, totalSize):
    downloaded = blockNum * blockSize
    if totalSize > 0:
        pct = min(100, downloaded * 100 / totalSize)
        sys.stdout.write(f"\r  {pct:.1f}% ({downloaded // (1024*1024)} MB / {totalSize // (1024*1024)} MB)")
        sys.stdout.flush()

def prepare_splits():
    if os.path.exists(SPLIT_FILE):
        print(f"splits already exist at {SPLIT_FILE}")
        return

    print("loading raw dataset...")
    with open(RAW_FILE, "rb") as f:
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
            
        if snr < -4:
            continue

        modName = keyMap.get(modType, modType.decode("utf-8"))

        for i in range(samples.shape[0]):
            allX.append(samples[i])
            allLabels.append((modName, snr))

    X = np.array(allX)
    labels = np.array(allLabels, dtype=object)

    print(f"total samples: {len(X)}")
    print(f"modulation types: {sorted(set(l[0] for l in allLabels))}")
    print(f"snr range: {sorted(set(l[1] for l in allLabels))}")

    print("splitting with snr stratification...")
    splits = snr_stratified_split(X, labels)

    print(f"  train: {len(splits['train']['X'])}")
    print(f"  val:   {len(splits['val']['X'])}")
    print(f"  test:  {len(splits['test']['X'])}")

    with open(SPLIT_FILE, "wb") as f:
        pickle.dump(splits, f, protocol=4)

    print(f"saved splits to {SPLIT_FILE}")

if __name__ == "__main__":
    download_dataset()
    prepare_splits()
