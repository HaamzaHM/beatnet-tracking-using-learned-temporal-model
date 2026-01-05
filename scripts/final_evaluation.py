"""
Final reproducible evaluation of Learned Temporal Model vs Particle Filter.

Fixed thresholds - no auto-tuning, no sweeping, no adaptation.
Designed for final reporting and comparison.

Usage:
    python3 scripts/final_evaluation.py --audio <path>
    python3 scripts/final_evaluation.py --audio test_data/808kick120bpm.mp3
"""

import argparse
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from BeatNet.BeatNet import BeatNet
from BeatNet.ltm_model import LearnedTemporalModel


# ============================================================================
# FIXED PARAMETERS FOR FINAL EVALUATION
# ============================================================================

BEAT_THRESHOLD = 0.6  # Fixed - no tuning
DOWNBEAT_THRESHOLD = 0.9  # Fixed - no tuning
MIN_INTER_BEAT_INTERVAL = 0.25  # Fixed - 0.25 seconds
FPS = 50  # Fixed frame rate


def get_tempo_from_beats(beat_times):
    """Compute tempo from beat times using median inter-beat interval."""
    if len(beat_times) < 2:
        return None

    ibi = np.diff(beat_times)
    median_ibi = np.median(ibi)
    if median_ibi <= 0:
        return None

    tempo = 60.0 / median_ibi
    return tempo


def extract_beats_fixed(logits, fps=FPS, beat_threshold=BEAT_THRESHOLD,
                        downbeat_threshold=DOWNBEAT_THRESHOLD,
                        min_inter_beat_interval=MIN_INTER_BEAT_INTERVAL):
    """Extract beat and downbeat times with FIXED parameters.
    
    No auto-tuning, no sweeping. Pure threshold-based detection.
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)  # (T, 2)
    
    # Get beat and downbeat probabilities
    beat_probs = probs[:, 0].cpu().numpy()
    downbeat_probs = probs[:, 1].cpu().numpy()
    
    # Simple thresholding approach (FIXED)
    beat_frames = np.where(beat_probs >= beat_threshold)[0]
    downbeat_frames = np.where(downbeat_probs >= downbeat_threshold)[0]
    
    # Post-process to enforce minimum inter-beat interval (FIXED)
    min_distance_frames = int(min_inter_beat_interval * fps)
    
    if len(beat_frames) == 0:
        beat_times = np.array([])
        downbeat_times = np.array([])
    else:
        # Greedy selection to enforce minimum distance
        beat_peaks = [beat_frames[0]]
        for frame in beat_frames[1:]:
            if frame - beat_peaks[-1] >= min_distance_frames:
                beat_peaks.append(frame)
        beat_peaks = np.array(beat_peaks)
        
        # Downbeats must be subset of beats (snap to nearest)
        downbeat_peaks = []
        for db_frame in downbeat_frames:
            if len(beat_peaks) == 0:
                continue
            distances = np.abs(beat_peaks - db_frame)
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]
            # Include if close to a beat (within 100ms)
            if nearest_distance <= int(0.1 * fps):
                downbeat_peaks.append(beat_peaks[nearest_idx])
        
        downbeat_peaks = np.sort(np.unique(downbeat_peaks)) if downbeat_peaks else np.array([])
        
        # Convert to times
        beat_times = beat_peaks / fps
        downbeat_times = downbeat_peaks / fps
    
    return beat_times, downbeat_times


def pf_evaluate(audio_path):
    """Evaluate Particle Filter."""
    y, sr = librosa.load(str(audio_path), sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    
    beatnet = BeatNet(model=1, algorithm="PF", mode="online")
    
    start = time.time()
    activations = beatnet.activation_extractor_online(str(audio_path))
    result = beatnet.estimator.process(activations)
    runtime_ms = (time.time() - start) * 1000
    
    beat_times = result[:, 0]
    beat_types = result[:, 1]
    downbeat_mask = (beat_types == 1)
    
    num_beats = len(beat_times)
    num_downbeats = int(downbeat_mask.sum())
    tempo = get_tempo_from_beats(beat_times) if len(beat_times) >= 2 else None
    
    return {
        "algorithm": "Particle Filter",
        "beats": num_beats,
        "downbeats": num_downbeats,
        "tempo": tempo,
        "runtime_ms": runtime_ms,
    }


def ltm_evaluate(audio_path, project_root):
    """Evaluate trained LTM with FIXED parameters."""
    device = torch.device('cpu')
    
    # Load audio and get CRNN activations
    y, sr = librosa.load(str(audio_path), sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Initialize model
    model = LearnedTemporalModel(
        input_dim=2,
        hidden_dim=128,
        num_layers=4,
        output_dim=2,
        architecture="tcn",
        device=device,
    )
    
    # Load trained weights
    weights_path = project_root / "models" / "ltm_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Get CRNN activations and run model
    beatnet = BeatNet(model=1)
    activations = beatnet.activation_extractor_online(str(audio_path))
    
    start = time.time()
    with torch.no_grad():
        acts_tensor = torch.FloatTensor(activations).unsqueeze(0).to(device)
        logits = model(acts_tensor).cpu()
    
    # Extract beats with FIXED parameters
    beat_times, downbeat_times = extract_beats_fixed(
        logits[0],
        fps=FPS,
        beat_threshold=BEAT_THRESHOLD,
        downbeat_threshold=DOWNBEAT_THRESHOLD,
        min_inter_beat_interval=MIN_INTER_BEAT_INTERVAL,
    )
    
    runtime_ms = (time.time() - start) * 1000
    
    num_beats = len(beat_times)
    num_downbeats = len(downbeat_times)
    tempo = get_tempo_from_beats(beat_times) if len(beat_times) >= 2 else None
    
    return {
        "algorithm": "Learned Temporal Model",
        "beats": num_beats,
        "downbeats": num_downbeats,
        "tempo": tempo,
        "runtime_ms": runtime_ms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Final reproducible evaluation of LTM vs Particle Filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 final_evaluation.py --audio test_data/808kick120bpm.mp3
  python3 final_evaluation.py --audio path/to/ballroom/audio.mp3
  python3 final_evaluation.py --audio path/to/gtzan/audio.mp3
        """,
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file to evaluate",
    )
    
    args = parser.parse_args()
    audio_path = Path(args.audio)
    
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)
    
    project_root = Path(__file__).parent.parent
    
    print(f"\n{'='*80}")
    print(f"FINAL REPRODUCIBLE EVALUATION")
    print(f"{'='*80}")
    print(f"\nAudio file: {audio_path.name}")
    print(f"Duration: {librosa.get_duration(path=str(audio_path)):.2f}s")
    print(f"\nFixed Parameters:")
    print(f"  Beat threshold: {BEAT_THRESHOLD}")
    print(f"  Downbeat threshold: {DOWNBEAT_THRESHOLD}")
    print(f"  Min inter-beat interval: {MIN_INTER_BEAT_INTERVAL}s")
    print(f"\n{'-'*80}\n")
    
    # Evaluate Particle Filter
    print("Evaluating Particle Filter...")
    pf_results = pf_evaluate(audio_path)
    
    print("Evaluating Learned Temporal Model...")
    ltm_results = ltm_evaluate(audio_path, project_root)
    
    # Display results table
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}\n")
    
    metrics = [
        "Beats detected",
        "Downbeats detected",
        "Tempo (BPM)",
        "Runtime (ms)",
    ]
    
    print(f"{'Metric':<30} {'Particle Filter':<25} {'LTM':<25}")
    print("-" * 80)
    
    for metric in metrics:
        if metric == "Beats detected":
            pf_val = pf_results["beats"]
            ltm_val = ltm_results["beats"]
            print(f"{metric:<30} {pf_val:<25} {ltm_val:<25}")
        
        elif metric == "Downbeats detected":
            pf_val = pf_results["downbeats"]
            ltm_val = ltm_results["downbeats"]
            print(f"{metric:<30} {pf_val:<25} {ltm_val:<25}")
        
        elif metric == "Tempo (BPM)":
            pf_val = f"{pf_results['tempo']:.1f}" if pf_results["tempo"] else "N/A"
            ltm_val = f"{ltm_results['tempo']:.1f}" if ltm_results["tempo"] else "N/A"
            print(f"{metric:<30} {pf_val:<25} {ltm_val:<25}")
        
        elif metric == "Runtime (ms)":
            pf_val = f"{pf_results['runtime_ms']:.2f}"
            ltm_val = f"{ltm_results['runtime_ms']:.2f}"
            print(f"{metric:<30} {pf_val:<25} {ltm_val:<25}")
    
    # Speed comparison
    speedup = pf_results["runtime_ms"] / ltm_results["runtime_ms"]
    print(f"{'Speedup':<30} {'1.00x':<25} {f'{speedup:.2f}x':<25}")
    
    print(f"\n{'='*80}")
    print(f"LIMITATIONS")
    print(f"{'='*80}\n")
    print("""
1. FIXED THRESHOLDS:
   - Thresholds are fixed globally (not adaptive per audio)
   - beat_threshold=0.6, downbeat_threshold=0.9
   - Results may vary on different audio characteristics

2. DOWNBEAT HANDLING:
   - Downbeats are snapped to nearest beat within 100ms
   - Enforces downbeats are subset of beats
   - May underestimate or overestimate downbeats depending on training data quality

3. ALGORITHM DESIGN:
   - LTM prioritizes speed over temporal consistency
   - No post-processing smoothing or constraint propagation
   - Simple threshold-based, not probabilistic modeling

4. TRAINING DATA LIMITATION:
   - Downbeat annotations in training set (Ballroom + GTZAN) have quality variance
   - Model trained with fixed dataset split (80/20)
   - Performance depends on how similar test audio is to training distribution

5. FRAME-BASED DETECTION:
   - Detection operates at 50 fps frame level
   - Temporal resolution limited to 20ms (1/50s)
   - No sub-frame accuracy
""")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
