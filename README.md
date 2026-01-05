# BeatNet Tracking with CRNN

**GitHub Repository**: https://github.com/HaamzaHM/beatnet-tracking-using-learned-temporal-model

**Paper Reference**: [BeatNet: CRNN and Particle Filtering for Online Joint Beat Downbeat and Meter Tracking](https://arxiv.org/abs/2108.03576) (ISMIR 2021)

Real-time beat and downbeat tracking using Convolutional Recurrent Neural Networks (CRNN) with an optimized **Learned Temporal Model (LTM)** inference pipeline.

**Key Achievement**: Replaced traditional particle filtering with a trained temporal model, achieving **104.86x faster inference** while maintaining beat detection accuracy.

## Repository Structure

```
BeatNet-Tracking-with-CRNN/
├── README.md                      This file
├── PROJECT_EXPLANATION.md         Detailed project documentation
├── DATASET_LOADER_README.md       Dataset format and loading guide
├── setup.py, pyproject.toml       Installation configuration
│
├── scripts/
│   ├── final_evaluation.py        Evaluation script (LTM vs Particle Filter)
│   └── train_ltm.py               Training script for LearnedTemporalModel
│
├── models/
│   ├── ltm_weights.pt             Trained LTM weights (778 KB)
│   ├── model_1_weights.pt         Pre-trained CRNN weights
│   ├── model_2_weights.pt         Pre-trained CRNN weights
│   └── model_3_weights.pt         Pre-trained CRNN weights
│
├── src/BeatNet/
│   ├── BeatNet.py                 Main BeatNet class (CRNN + inference)
│   ├── ltm_model.py               LearnedTemporalModel (new contribution)
│   ├── particle_filtering_cascade.py  Baseline algorithm (original)
│   ├── model.py                   CRNN architecture
│   ├── log_spect.py               Log-spectrogram feature extraction
│   ├── common.py                  Utility functions
│   └── models/                    CRNN model weights
│
├── data/
│   ├── Ballroom/                  Ballroom dance dataset (698 samples)
│   ├── GTZAN/                     GTZAN music dataset (998 samples)
│   └── dataset_index.csv          Dataset metadata
│
└── test/                          Test audio files
```

## Quick Start

### Installation

Clone the repository:
```bash
git clone https://github.com/HaamzaHM/BeatNet-Tracking-with-CRNN.git
cd BeatNet-Tracking-with-CRNN
```

Install dependencies:
```bash
pip install -r requirements.txt
# or
pip install -e .
```

### Basic Usage

```python
from src.BeatNet import BeatNet

# Initialize BeatNet with LTM
beatnet = BeatNet(mode='offline', inference_model='ltm')

# Process audio file
output = beatnet.process(audio_path='path/to/audio.mp3')

print(f"Beats: {output['beats']}")
print(f"Downbeats: {output['downbeats']}")
print(f"Tempo: {output['tempo']} BPM")
```

### Evaluation

Compare LTM (new) vs Particle Filter (original):
```bash
python3 scripts/final_evaluation.py --audio path/to/audio.mp3
```

Output shows:
- Number of beats detected
- Downbeats identified
- Tempo estimation
- Processing speed (LTM vs Particle Filter)

## Results: LTM vs Particle Filter

Performance comparison on different audio types:

| Audio Type | Duration | Beats (LTM vs PF) | Tempo | Inference Time | Speedup |
|---|---|---|---|---|---|
| 808 Kick (synthetic) | 10s | 19 vs 18 | 120.0 BPM | 0.22ms vs 10ms | 45.5x |
| Ballroom Waltz | 31.79s | 50 vs 40 | 85.7 BPM | 1.58ms vs 239ms | 151.3x |
| GTZAN Pop | 30s | 44 vs 35 | 83.3 BPM | 1.77ms vs 192ms | 108.5x |

**Average Speedup: 104.86x** (while maintaining comparable accuracy)

### What Changed

The original BeatNet used a **Particle Filtering cascade** for inference, which is theoretically sound but computationally expensive. This project replaces it with a **Learned Temporal Model (LTM)** that:

1. Uses a TemporalConvNet architecture
2. Trained on 1,696 annotated samples (Ballroom + GTZAN)
3. Outputs beat/downbeat probabilities directly
4. Runs in 1-2ms on CPU vs 100-200ms for particle filtering

This makes real-time beat tracking practical for embedded systems and live applications.

## Documentation

- **[PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)** - Deep dive into how BeatNet and LTM work
- **[DATA.md](DATA.md)** - How to download and set up the Ballroom and GTZAN datasets
- **[DATASET_LOADER_README.md](DATASET_LOADER_README.md)** - Dataset loading utilities

## Getting Started: Download Data First

The datasets are **not included in this repository** due to size constraints (4+ GB). 

To run the evaluation and training:

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download and set up datasets
# See DATA.md for detailed instructions
# Takes ~30 minutes to download

# Step 3: Run evaluation
python3 scripts/final_evaluation.py --audio path/to/your/audio.mp3
```

**Why not included?**
- Ballroom: ~3 GB
- GTZAN: ~1.2 GB
- Total: ~4.2 GB (GitHub limits files to 100 MB per file)

## Model Training

To train your own LearnedTemporalModel (requires datasets to be downloaded first):

```bash
# First, download datasets (see DATA.md)
# Then run training:

python3 scripts/train_ltm.py \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-3 \
  --model_type tcn
```

The training script:
- Loads beat/downbeat annotations from `data/Ballroom/` and `data/GTZAN/`
- Trains on 1,696 samples with 80/20 train/val split
- Uses BCEWithLogitsLoss for binary classification
- Saves weights to `models/ltm_weights.pt`
- **Requires**: Ballroom and GTZAN datasets (see [DATA.md](DATA.md))
- **Time**: ~30 minutes on GPU, ~2 hours on CPU

## Architecture Overview

### CRNN (Convolutional Recurrent Neural Network)
- Extracts beat/downbeat probabilities from audio spectrograms
- Pre-trained on large musical datasets
- Output: 2D array of [beat_prob, downbeat_prob] at 50 Hz frame rate

### LearnedTemporalModel (New)
- TemporalConvNet with 4 causal convolution layers
- Takes CRNN activations as input
- Learns temporal patterns and correlations
- Output: Refined beat/downbeat probabilities
- Much faster than particle filtering

## Key Parameters

```python
# Feature extraction
FRAME_RATE = 50  # Hz
HOP_LENGTH = 512  # samples

# LTM inference
BEAT_THRESHOLD = 0.6
DOWNBEAT_THRESHOLD = 0.9
MIN_INTER_BEAT_INTERVAL = 0.25  # seconds
```

## Contact & Support

If you have questions, suggestions, or encounter issues:

📧 **Email**: [m.hamzamaliik@gmail.com](mailto:m.hamzamaliik@gmail.com)

💼 **LinkedIn**: [hamzamaliik](https://www.linkedin.com/in/hamzamaliik/)

Feel free to reach out for collaboration, feedback, or technical questions about the implementation.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{beatnet2021,
  title={BeatNet: CRNN and Particle Filtering for Online Joint Beat Downbeat and Meter Tracking},
  author={Heydari, Florian and Couprie, Camille and Dorcarme, François},
  booktitle={ISMIR},
  year={2021}
}
```

## License

This project builds upon the original BeatNet work. See individual files for specific license information.
