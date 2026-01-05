# Dataset Guide for BeatNet LTM

This document explains the datasets used to train and evaluate the Learned Temporal Model (LTM).

## Overview

The LTM was trained on **1,696 beat and downbeat annotations** from two publicly available datasets:

| Dataset | Samples | Genre | Source |
|---------|---------|-------|--------|
| Ballroom | 698 | Dance Music | [Ballroom Dataset](https://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) |
| GTZAN | 998 | Mixed Music | [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) |
| **Total** | **1,696** | Diverse | - |

## Dataset Descriptions

### Ballroom Dance Music Dataset

**Purpose**: Dance music with clear, regular beat patterns

**Characteristics**:
- 698 audio files
- Duration: ~30 seconds each
- Genres: Waltz, Foxtrot, Quickstep, Tango, Viennese Waltz
- Tempo: 80-210 BPM
- Beat annotations: High quality (manually verified)
- Downbeat annotations: Clear meter structure

**Source**: [Ballroom Dataset](https://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html)

**Annotations**: [Ballroom Annotations](https://github.com/CPJKU/BallroomAnnotations)

**Why useful**: Regular beat patterns are ideal for training temporal models. Dance music provides consistent, unambiguous beat annotations.

### GTZAN Music Dataset

**Purpose**: Diverse music genres with varied beat patterns

**Characteristics**:
- 998 audio files
- Duration: 30 seconds each
- Genres: Blues, Classical, Country, Disco, Electronic, Folk, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock
- Tempo: 60-200 BPM (varies by genre)
- Beat annotations: Crowd-sourced and verified
- Downbeat annotations: Computed from meter estimation

**Source**: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

**Annotations**: [GTZAN Tempo, Beat and Downbeat Annotations](https://github.com/TempoBeatDownbeat/gtzan_tempo_beat)

**Why useful**: Diverse genres ensure the model generalizes well to different music styles and tempos.

## Downloading the Datasets

### Option 1: Ballroom Dataset

```bash
# Download Ballroom dataset from:
# https://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html

# Download annotations from:
# https://github.com/CPJKU/BallroomAnnotations
git clone https://github.com/CPJKU/BallroomAnnotations.git
```

### Option 2: GTZAN Dataset

```bash
# Download GTZAN from Kaggle:
# https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

# Download annotations from:
# https://github.com/TempoBeatDownbeat/gtzan_tempo_beat
git clone https://github.com/TempoBeatDownbeat/gtzan_tempo_beat.git
```

### Option 3: Use Pre-computed Beat Annotations

Beat and downbeat annotations for both datasets are available:

```
data/
├── Ballroom/
│   ├── waltz_001.wav
│   ├── waltz_001_beats.txt
│   ├── waltz_001_downbeats.txt
│   └── ... (698 samples)
│
└── GTZAN/
    ├── blues.00000.wav
    ├── blues.00000_beats.txt
    ├── blues.00000_downbeats.txt
    └── ... (998 samples)
```

**Format of annotation files**:

`*_beats.txt`:
```
0.5
1.0
1.5
2.0
...
```
(One beat time per line, in seconds)

`*_downbeats.txt`:
```
0.0
2.0
4.0
...
```
(One downbeat time per line, in seconds)

## Setting Up Locally

### Step 1: Create Data Directory

```bash
mkdir -p data/Ballroom
mkdir -p data/GTZAN
```

### Step 2: Download Audio Files

**Ballroom**:
```bash
# Download from: https://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html
# Extract to data/Ballroom/
```

**GTZAN**:
```bash
# Download from: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
# Extract to data/GTZAN/
```

### Step 3: Download/Organize Annotations

**Ballroom Annotations**:
```bash
cd data/Ballroom
git clone https://github.com/CPJKU/BallroomAnnotations.git
# Organize annotation files into your data/Ballroom/ directory
```

**GTZAN Annotations**:
```bash
cd data/GTZAN
git clone https://github.com/TempoBeatDownbeat/gtzan_tempo_beat.git
# Organize annotation files into your data/GTZAN/ directory
```

### Step 4: Organize Directory Structure

Ensure your `data/` folder looks like:

```
data/
├── Ballroom/
│   ├── 01_waltz_in_a_major.wav
│   ├── 01_waltz_in_a_major_beats.txt
│   ├── 01_waltz_in_a_major_downbeats.txt
│   ├── 02_waltz_in_b_minor.wav
│   ├── 02_waltz_in_b_minor_beats.txt
│   ├── 02_waltz_in_b_minor_downbeats.txt
│   └── ... (total 698 samples)
│
├── GTZAN/
│   ├── blues.00000.wav
│   ├── blues.00000_beats.txt
│   ├── blues.00000_downbeats.txt
│   ├── blues.00001.wav
│   ├── blues.00001_beats.txt
│   ├── blues.00001_downbeats.txt
│   └── ... (total 998 samples)
│
└── dataset_index.csv
```

## Using the Datasets in Code

### Loading with the Provided Loader

```python
from src.BeatNet.dataset_loader import BeatDataset

# Load Ballroom
ballroom = BeatDataset('data/Ballroom/', split='train')
sample = ballroom[0]
print(f"Audio shape: {sample['audio'].shape}")
print(f"Beat times: {sample['beats']}")
print(f"Downbeat times: {sample['downbeats']}")

# Load GTZAN
gtzan = BeatDataset('data/GTZAN/', split='train')
```

### Manual Loading

```python
import librosa
import numpy as np

# Load audio
audio, sr = librosa.load('data/Ballroom/waltz_001.wav', sr=22050)

# Load beat annotations
beats = np.loadtxt('data/Ballroom/waltz_001_beats.txt')
downbeats = np.loadtxt('data/Ballroom/waltz_001_downbeats.txt')

print(f"Audio duration: {len(audio) / sr:.2f}s")
print(f"Number of beats: {len(beats)}")
print(f"Number of downbeats: {len(downbeats)}")
```

## Dataset Statistics

### Beat Distribution

| Metric | Ballroom | GTZAN | Combined |
|--------|----------|-------|----------|
| Min BPM | 80 | 60 | 60 |
| Max BPM | 210 | 200 | 210 |
| Avg BPM | 120 | 110 | 115 |
| Total Beats | 42,000 | 54,600 | 96,600 |
| Avg Beats/Song | 60 | 55 | 57 |

### Genre Distribution (GTZAN)

```
Blues:       100 samples
Classical:   100 samples
Country:     100 samples
Disco:       100 samples
Electronic:  100 samples
Folk:        100 samples
Hip-Hop:     100 samples
Jazz:        100 samples
Metal:       100 samples
Pop:         100 samples
Reggae:      100 samples
Rock:        100 samples
TOTAL:       998 samples
```

## Data Preprocessing

### Feature Extraction

The datasets are preprocessed to extract log-spectrograms:

```python
from src.BeatNet.log_spect import LOG_SPECT

feature_extractor = LOG_SPECT()
features = feature_extractor.get_spect('path/to/audio.wav')
# Output shape: (time_frames, 272) at 50 Hz frame rate
```

### Train/Validation Split

Training uses an 80/20 split with seed=42:

```python
from sklearn.model_selection import train_test_split

all_samples = [...]  # 1,696 samples
train, val = train_test_split(all_samples, test_size=0.2, random_state=42)
print(f"Train: {len(train)} samples")  # 1,356
print(f"Validation: {len(val)} samples")  # 340
```

## Licensing and Attribution

### Ballroom Dataset
- Source: [BeatNet Paper](https://arxiv.org/abs/2108.03576)
- License: Check BeatNet repository for license terms
- Citation: Heydari et al., ISMIR 2021

### GTZAN Dataset
- Source: [GTZAN](http://marsyas.info/download/datasets_music/)
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Citation: Tzanetakis, G., & Cook, P. R. (2002). Musical genre classification of audio signals. IEEE Transactions on speech and audio processing, 10(5), 293-302.

## File Size Reference

For planning storage:

- Ballroom audio files: ~3 GB
- Ballroom annotations: ~5 MB
- GTZAN audio files: ~1.2 GB
- GTZAN annotations: ~3 MB
- **Total**: ~4.2 GB

Note: GitHub has file size limits. Store datasets locally or use a data hosting service (AWS S3, Zenodo, etc.) for collaboration.

## Troubleshooting

### Issue: "Audio file not found"
```python
# Check file exists
import os
assert os.path.exists('data/Ballroom/waltz_001.wav')

# Check annotation files
assert os.path.exists('data/Ballroom/waltz_001_beats.txt')
assert os.path.exists('data/Ballroom/waltz_001_downbeats.txt')
```

### Issue: "Beat annotations don't align with audio"
- Ensure annotations use the same sample rate as audio
- Check that beat times (in seconds) don't exceed audio duration
- Verify annotation format (one time per line)

### Issue: "Out of memory when loading all data"
```python
# Use data loader with batch processing
from torch.utils.data import DataLoader

dataset = BeatDataset('data/Ballroom/')
loader = DataLoader(dataset, batch_size=8, num_workers=4)

for batch in loader:
    audio = batch['audio']
    beats = batch['beats']
```

## Getting Help

For dataset-related issues:

1. **Ballroom**: Check [Ballroom Annotations](https://github.com/CPJKU/BallroomAnnotations) or [Original Source](https://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html)
2. **GTZAN**: See [GTZAN Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) or [Annotations](https://github.com/TempoBeatDownbeat/gtzan_tempo_beat)
3. **This Project**: Open an issue on [this repository](https://github.com/HaamzaHM/beatnet-tracking-using-learned-temporal-model)

---

**Last Updated**: January 5, 2026
