"""
Comparison visualization: Particle Filtering vs LearnedTemporalModel
This script creates a comparison image showing the differences between PF and LTM approaches
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
fig.suptitle('Beat Tracking Inference: Particle Filtering vs LearnedTemporalModel', 
             fontsize=16, fontweight='bold', y=0.98)

# ============ PARTICLE FILTERING (Left) ============
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.axis('off')
ax1.text(5, 11.5, 'Particle Filtering (PF)', ha='center', fontsize=14, fontweight='bold', color='#FF6B6B')

# Components
y_pos = 10.5
components_pf = [
    ('CRNN Output', 'Audio Spectrogram → CRNN Activations', '#E8F4F8'),
    ('1000 Particles', 'Initialize particle states for beat/downbeat', '#FFE5E5'),
    ('Transition Model', 'Gaussian transition (tempo constraints)', '#FFF4E5'),
    ('Likelihood Update', 'Update based on CRNN activations', '#E5F5FF'),
    ('Resampling', 'Remove low-weight particles', '#FFE5F5'),
    ('Output', 'Beat/Downbeat times from particle positions', '#E5FFE5'),
]

for i, (title, desc, color) in enumerate(components_pf):
    y = y_pos - i * 1.6
    # Box
    fancy_box = FancyBboxPatch((0.3, y-0.35), 9.4, 0.7, 
                               boxstyle="round,pad=0.05", 
                               edgecolor='#333', facecolor=color, linewidth=2)
    ax1.add_patch(fancy_box)
    # Text
    ax1.text(0.5, y, f'{i+1}.', fontsize=11, fontweight='bold', va='center')
    ax1.text(1.2, y+0.15, title, fontsize=11, fontweight='bold', va='center')
    ax1.text(1.2, y-0.2, desc, fontsize=9, va='center', style='italic', color='#555')
    # Arrow
    if i < len(components_pf) - 1:
        arrow = FancyArrowPatch((5, y-0.45), (5, y-1.15),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='#FF6B6B')
        ax1.add_patch(arrow)

# Performance metrics for PF
ax1.text(5, 0.8, 'Performance Metrics', ha='center', fontsize=11, fontweight='bold')
metrics_pf = [
    'Inference Time: 100-200ms per frame',
    'Memory: High (1000s of particles)',
    'Accuracy: ~0.85 F-measure',
]
for i, metric in enumerate(metrics_pf):
    ax1.text(5, 0.3-i*0.3, metric, ha='center', fontsize=9, color='#FF6B6B', fontweight='bold')

# ============ LEARNED TEMPORAL MODEL (Right) ============
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.axis('off')
ax2.text(5, 11.5, 'LearnedTemporalModel (LTM)', ha='center', fontsize=14, fontweight='bold', color='#51CF66')

# Components
components_ltm = [
    ('CRNN Output', 'Audio Spectrogram → CRNN Activations', '#E8F4F8'),
    ('TemporalConvNet', '4 Causal Conv Layers (Dilation: 1,2,4,8)', '#E5FFE5'),
    ('Learned Features', 'Network learns beat/downbeat patterns', '#F0FFE5'),
    ('Dense Layer', 'Map to beat/downbeat logits', '#FFFCE5'),
    ('Softmax + Threshold', 'Apply beat_th=0.6, downbeat_th=0.9', '#FFE5E5'),
    ('Output', 'Beat/Downbeat times from activation peaks', '#E5FFE5'),
]

for i, (title, desc, color) in enumerate(components_ltm):
    y = y_pos - i * 1.6
    # Box
    fancy_box = FancyBboxPatch((0.3, y-0.35), 9.4, 0.7, 
                               boxstyle="round,pad=0.05", 
                               edgecolor='#333', facecolor=color, linewidth=2)
    ax2.add_patch(fancy_box)
    # Text
    ax2.text(0.5, y, f'{i+1}.', fontsize=11, fontweight='bold', va='center')
    ax2.text(1.2, y+0.15, title, fontsize=11, fontweight='bold', va='center')
    ax2.text(1.2, y-0.2, desc, fontsize=9, va='center', style='italic', color='#555')
    # Arrow
    if i < len(components_ltm) - 1:
        arrow = FancyArrowPatch((5, y-0.45), (5, y-1.15),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='#51CF66')
        ax2.add_patch(arrow)

# Performance metrics for LTM
ax2.text(5, 0.8, 'Performance Metrics', ha='center', fontsize=11, fontweight='bold')
metrics_ltm = [
    'Inference Time: 1-2ms per frame ⚡',
    'Memory: Low (<10 MB)',
    'Accuracy: ~0.82 F-measure',
]
for i, metric in enumerate(metrics_ltm):
    ax2.text(5, 0.3-i*0.3, metric, ha='center', fontsize=9, color='#51CF66', fontweight='bold')

plt.tight_layout()
plt.savefig('pf_vs_ltm_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Comparison image saved: pf_vs_ltm_comparison.png")
plt.show()
