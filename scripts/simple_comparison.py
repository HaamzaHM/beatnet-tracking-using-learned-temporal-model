"""
Simplified Comparison: DBN vs LearnedTemporalModel
Generates publication-quality comparison plots

Usage:
    python3 scripts/simple_comparison.py --audio <path>
"""

import argparse
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from BeatNet.BeatNet import BeatNet
from BeatNet.ltm_model import LearnedTemporalModel
from BeatNet.log_spect import LOG_SPECT


BEAT_THRESHOLD = 0.6
DOWNBEAT_THRESHOLD = 0.9
MIN_INTER_BEAT_INTERVAL = 0.25
FPS = 50


def get_tempo_from_beats(beat_times):
    """Compute tempo from beat times."""
    if len(beat_times) < 2:
        return 0
    ibi = np.diff(beat_times)
    median_ibi = np.median(ibi)
    if median_ibi <= 0:
        return 0
    tempo = 60.0 / median_ibi
    return tempo


def process_with_dbn(audio_path):
    """Process audio with DBN inference."""
    print("\n🎵 Processing with Dynamic Bayesian Network (DBN)...")
    start_time = time.time()
    
    beatnet_dbn = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
    output_dbn = beatnet_dbn.process(str(audio_path))
    
    dbn_time = time.time() - start_time
    
    beats_dbn = output_dbn[:, 0]
    downbeats_dbn = output_dbn[:, 1]
    
    tempo_dbn = get_tempo_from_beats(beats_dbn)
    
    return {
        'beats': beats_dbn,
        'downbeats': downbeats_dbn,
        'tempo': tempo_dbn,
        'time': dbn_time,
        'num_beats': len(beats_dbn),
        'num_downbeats': len(downbeats_dbn)
    }


def process_with_ltm(audio_path):
    """Process audio with LearnedTemporalModel."""
    print("\n⚡ Processing with LearnedTemporalModel (LTM)...")
    start_time = time.time()
    
    # Load audio and compute spectrogram
    y, sr = librosa.load(str(audio_path), sr=22050)
    
    # Use CRNN to get activations
    beatnet_crnn = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
    activations = beatnet_crnn.crnn.forward(y)  # Get CRNN activations
    
    # Process with LTM
    ltm = LearnedTemporalModel(input_dim=128, hidden_dim=128, num_layers=4)
    ltm_weights = Path(__file__).parent.parent / 'models' / 'ltm_weights.pt'
    
    if ltm_weights.exists():
        ltm.load_state_dict(torch.load(ltm_weights, map_location='cpu'))
        ltm.eval()
    
    # Get LTM outputs
    with torch.no_grad():
        ltm_output = ltm(torch.tensor(activations, dtype=torch.float32).unsqueeze(0))
    
    # Extract beats and downbeats from LTM output
    beat_probs = ltm_output[0, :, 0].cpu().numpy()
    downbeat_probs = ltm_output[0, :, 1].cpu().numpy()
    
    # Threshold
    beat_frames = np.where(beat_probs > BEAT_THRESHOLD)[0]
    downbeat_frames = np.where(downbeat_probs > DOWNBEAT_THRESHOLD)[0]
    
    # Convert to time
    beats_ltm = beat_frames / FPS
    downbeats_ltm = downbeat_frames / FPS
    
    # Remove duplicates within min interval
    beats_ltm_filtered = []
    for beat in beats_ltm:
        if not beats_ltm_filtered or beat - beats_ltm_filtered[-1] >= MIN_INTER_BEAT_INTERVAL:
            beats_ltm_filtered.append(beat)
    beats_ltm = np.array(beats_ltm_filtered)
    
    ltm_time = time.time() - start_time
    tempo_ltm = get_tempo_from_beats(beats_ltm)
    
    return {
        'beats': beats_ltm,
        'downbeats': downbeats_ltm,
        'tempo': tempo_ltm,
        'time': ltm_time,
        'num_beats': len(beats_ltm),
        'num_downbeats': len(downbeats_ltm)
    }


def create_comparison_plot(audio_path, output_dbn, output_ltm, save_path='comparison.png'):
    """Create comparison visualization."""
    
    print("\n📊 Generating comparison visualization...")
    
    y, sr = librosa.load(str(audio_path), sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    frames = S_db.shape[1]
    time_axis = librosa.frames_to_time(np.arange(frames), sr=sr)
    time_audio = np.arange(len(y)) / sr
    
    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(18, 12))
    fig.suptitle('BeatNet: DBN vs LearnedTemporalModel - Comprehensive Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Row 0: Waveform
    ax = axes[0, 0]
    ax.plot(time_audio, y, color='steelblue', alpha=0.7, linewidth=0.5)
    for beat in output_dbn['beats']:
        ax.axvline(beat, color='red', alpha=0.5, linewidth=1.5)
    for downbeat in output_dbn['downbeats']:
        ax.axvline(downbeat, color='lime', alpha=0.8, linewidth=2)
    ax.set_title('DBN: Audio Waveform with Beats', fontsize=11, fontweight='bold')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(time_audio, y, color='steelblue', alpha=0.7, linewidth=0.5)
    for beat in output_ltm['beats']:
        ax.axvline(beat, color='red', alpha=0.5, linewidth=1.5)
    for downbeat in output_ltm['downbeats']:
        ax.axvline(downbeat, color='lime', alpha=0.8, linewidth=2)
    ax.set_title('LTM: Audio Waveform with Beats', fontsize=11, fontweight='bold')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    # Row 1: Mel spectrogram
    ax = axes[1, 0]
    im = ax.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
    for beat in output_dbn['beats']:
        ax.axvline(librosa.time_to_frames(beat, sr=sr), color='red', alpha=0.5, linewidth=1)
    for downbeat in output_dbn['downbeats']:
        ax.axvline(librosa.time_to_frames(downbeat, sr=sr), color='lime', alpha=0.7, linewidth=1.5)
    ax.set_title('DBN: Mel Spectrogram', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mel Bin')
    plt.colorbar(im, ax=ax, label='dB')
    
    ax = axes[1, 1]
    im = ax.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
    for beat in output_ltm['beats']:
        ax.axvline(librosa.time_to_frames(beat, sr=sr), color='red', alpha=0.5, linewidth=1)
    for downbeat in output_ltm['downbeats']:
        ax.axvline(librosa.time_to_frames(downbeat, sr=sr), color='lime', alpha=0.7, linewidth=1.5)
    ax.set_title('LTM: Mel Spectrogram', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mel Bin')
    plt.colorbar(im, ax=ax, label='dB')
    
    # Row 2: Beat timeline
    ax = axes[2, 0]
    ax.set_xlim(0, time_axis[-1])
    ax.set_ylim(0, 1)
    for beat in output_dbn['beats']:
        color = 'lime' if beat in output_dbn['downbeats'] else 'red'
        size = 150 if beat in output_dbn['downbeats'] else 80
        ax.scatter(beat, 0.5, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=1)
    ax.set_title(f"DBN: Beat Timeline\n(Beats: {output_dbn['num_beats']}, Downbeats: {output_dbn['num_downbeats']})", 
                 fontsize=11, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    ax = axes[2, 1]
    ax.set_xlim(0, time_axis[-1])
    ax.set_ylim(0, 1)
    for beat in output_ltm['beats']:
        color = 'lime' if beat in output_ltm['downbeats'] else 'red'
        size = 150 if beat in output_ltm['downbeats'] else 80
        ax.scatter(beat, 0.5, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=1)
    ax.set_title(f"LTM: Beat Timeline\n(Beats: {output_ltm['num_beats']}, Downbeats: {output_ltm['num_downbeats']})", 
                 fontsize=11, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Row 3: Metrics
    ax = axes[3, 0]
    ax.axis('off')
    metrics_dbn = f"""
DBN (Dynamic Bayesian Network)

Beats Detected: {output_dbn['num_beats']}
Downbeats: {output_dbn['num_downbeats']}
Estimated Tempo: {output_dbn['tempo']:.1f} BPM

Inference Time: {output_dbn['time']*1000:.2f} ms
    """
    ax.text(0.5, 0.5, metrics_dbn, transform=ax.transAxes, fontfamily='monospace',
            fontsize=10, ha='center', va='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax = axes[3, 1]
    ax.axis('off')
    speedup = output_dbn['time'] / max(output_ltm['time'], 0.001)
    metrics_ltm = f"""
LearnedTemporalModel (LTM)

Beats Detected: {output_ltm['num_beats']}
Downbeats: {output_ltm['num_downbeats']}
Estimated Tempo: {output_ltm['tempo']:.1f} BPM

Inference Time: {output_ltm['time']*1000:.2f} ms
SPEEDUP: {speedup:.2f}x faster ⚡
    """
    ax.text(0.5, 0.5, metrics_ltm, transform=ax.transAxes, fontfamily='monospace',
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparison plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare DBN vs LTM inference')
    parser.add_argument('--audio', type=str, default='test/test_data/808kick120bpm.mp3',
                       help='Path to audio file')
    parser.add_argument('--output', type=str, default='comparison.png',
                       help='Output image path')
    
    args = parser.parse_args()
    audio_path = Path(args.audio)
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"\n📁 Processing: {audio_path.name}")
    duration = librosa.get_duration(filename=str(audio_path))
    print(f"   Duration: {duration:.2f}s")
    
    # Process with both methods
    output_dbn = process_with_dbn(audio_path)
    output_ltm = process_with_ltm(audio_path)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\n📊 DBN (Dynamic Bayesian Network):")
    print(f"   • Beats detected: {output_dbn['num_beats']}")
    print(f"   • Downbeats: {output_dbn['num_downbeats']}")
    print(f"   • Tempo: {output_dbn['tempo']:.1f} BPM")
    print(f"   • Time: {output_dbn['time']*1000:.2f} ms")
    
    print(f"\n⚡ LTM (LearnedTemporalModel):")
    print(f"   • Beats detected: {output_ltm['num_beats']}")
    print(f"   • Downbeats: {output_ltm['num_downbeats']}")
    print(f"   • Tempo: {output_ltm['tempo']:.1f} BPM")
    print(f"   • Time: {output_ltm['time']*1000:.2f} ms")
    
    speedup = output_dbn['time'] / max(output_ltm['time'], 0.001)
    print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster")
    print("="*70 + "\n")
    
    # Create visualization
    create_comparison_plot(audio_path, output_dbn, output_ltm, args.output)
    print(f"✅ Complete! Check '{args.output}'")


if __name__ == '__main__':
    main()
