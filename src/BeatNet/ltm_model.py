"""
Learned Temporal Model (LTM) for beat and downbeat inference.

This module defines a PyTorch-based temporal model that will eventually replace
the particle filtering cascade. It will implement causal temporal convolution (TCN)
or Transformer-based architectures for real-time beat and downbeat tracking.

Author: BeatNet
"""

import torch
import torch.nn as nn
import numpy as np


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with causal convolutions.
    
    Implements a stack of causal dilated convolutions for learning temporal
    patterns in beat/downbeat activation sequences. Causal convolutions ensure
    that predictions at time t only depend on t and earlier, enabling real-time
    inference.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden layer dimension
    output_dim : int
        Output feature dimension
    num_layers : int
        Number of conv blocks (each with dilation doubling: 1, 2, 4, 8, ...)
    kernel_size : int
        Convolution kernel size (default: 3)
    dropout : float
        Dropout probability (default: 0.1)
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size=3, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Stack of causal dilated convolutions
        self.conv_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            dilation = 2 ** layer_idx  # Exponential dilation
            padding = dilation * (kernel_size - 1)  # Causal padding
            
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation
                ),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass through TCN.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, time, input_dim)
        
        Returns
        -------
        torch.Tensor
            Output of shape (batch, time, output_dim)
        """
        # Project input: (batch, time, input_dim) → (batch, time, hidden_dim)
        h = self.input_proj(x)
        # Transpose for conv1d: (batch, time, hidden_dim) → (batch, hidden_dim, time)
        h = h.transpose(1, 2)
        
        # Apply causal convolutions
        for conv_layer in self.conv_layers:
            h = conv_layer(h)
            # Remove causal padding (keep only valid output)
            h = h[:, :, :x.size(1)]
        
        # Transpose back: (batch, hidden_dim, time) → (batch, time, hidden_dim)
        h = h.transpose(1, 2)
        
        # Project to output dimension
        output = self.output_proj(h)
        
        return output


class CausalTransformerEncoder(nn.Module):
    """
    Causal Transformer Encoder for temporal beat/downbeat modeling.
    
    Uses causal self-attention masks to ensure each position only attends to
    positions at or before it, enabling real-time inference.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    hidden_dim : int
        Hidden layer dimension
    num_layers : int
        Number of transformer encoder blocks
    num_heads : int
        Number of attention heads (default: 4)
    dropout : float
        Dropout probability (default: 0.1)
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads=4, dropout=0.1):
        super(CausalTransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding (learned)
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Stack of transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        """
        Forward pass through causal transformer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, time, input_dim)
        
        Returns
        -------
        torch.Tensor
            Output of shape (batch, time, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input: (batch, time, input_dim) → (batch, time, hidden_dim)
        h = self.input_proj(x)
        
        # Add positional encoding
        h = h + self.pos_embed[:, :seq_len, :]
        
        # Create causal attention mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )
        
        # Apply transformer with causal mask (use mask for newer PyTorch versions)
        h = self.transformer_encoder(h, mask=causal_mask)
        
        # Project to output dimension
        output = self.output_proj(h)
        
        return output


class LearnedTemporalModel(nn.Module):
    """
    A learnable temporal model for beat and downbeat inference.
    
    This module processes beat/downbeat activation sequences from the CRNN/BDA model
    and outputs beat/downbeat predictions using either TCN or Transformer architecture.
    
    The model learns to extract temporal patterns (tempo, meter) from the activation
    sequences, similar to particle filtering but in an end-to-end differentiable manner.
    This enables better generalization across diverse music genres when trained with
    beat/downbeat annotations.
    
    Architecture Options:
    - TCN (Temporal Convolutional Network): Fast inference, good for real-time
    - Transformer: More expressive temporal modeling, slightly higher latency
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of input features (typically 2 for beat and downbeat activations)
    hidden_dim : int
        Dimensionality of hidden layers and temporal representations
    num_layers : int
        Number of stacked temporal layers (TCN blocks or Transformer layers)
    device : str
        Device to place the model on ('cpu' or 'cuda:i')
    architecture : str, default='tcn'
        Architecture type: 'tcn' (Temporal Convolutional Network) or 'transformer'
    output_dim : int, default=2
        Output dimension (typically 2 for beat and downbeat predictions)
    num_heads : int, default=4
        Number of attention heads (only used for transformer architecture)
    dropout : float, default=0.1
        Dropout probability for regularization
    
    Attributes
    ----------
    input_dim : int
        Stored input dimensionality
    hidden_dim : int
        Stored hidden dimensionality
    num_layers : int
        Number of temporal layers
    device : str
        Target device for computation
    architecture : str
        Selected architecture type
    temporal_model : nn.Module
        The actual temporal processing network (TCN or Transformer)
    
    Example
    -------
    >>> model = LearnedTemporalModel(
    ...     input_dim=2,
    ...     hidden_dim=128,
    ...     num_layers=4,
    ...     device='cpu',
    ...     architecture='tcn'
    ... )
    >>> activations = torch.randn(1, 500, 2)  # (batch, time, features)
    >>> output = model(activations)  # (batch, time, 2)
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        device,
        architecture='tcn',
        output_dim=2,
        num_heads=4,
        dropout=0.1
    ):
        """
        Initialize the LearnedTemporalModel.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension (e.g., 2 for [beat_activation, downbeat_activation])
        hidden_dim : int
            Hidden layer dimension for temporal processing
        num_layers : int
            Number of temporal processing layers
        device : str
            Device to use ('cpu' or 'cuda:i')
        architecture : str
            'tcn' for Temporal Convolutional Network or 'transformer' for Transformer
        output_dim : int
            Output dimension (usually same as input_dim)
        num_heads : int
            Number of attention heads (for transformer only)
        dropout : float
            Dropout rate for regularization
        """
        super(LearnedTemporalModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.architecture = architecture.lower()
        self.output_dim = output_dim
        
        # Validate architecture choice
        if self.architecture not in ['tcn', 'transformer']:
            raise ValueError(f"Architecture must be 'tcn' or 'transformer', got '{self.architecture}'")
        
        # Build temporal model based on selected architecture
        if self.architecture == 'tcn':
            self.temporal_model = TemporalConvNet(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                kernel_size=3,
                dropout=dropout
            )
        elif self.architecture == 'transformer':
            self.temporal_model = CausalTransformerEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout
            )
        
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass through the temporal model.
        
        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Input tensor of shape (batch, time, input_dim) containing beat/downbeat
            activation probabilities from the CRNN/BDA model.
            Can be numpy array or torch tensor.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, time, output_dim) containing raw logits
            (unbounded real values, not constrained to [0, 1]).
            
            These raw logits are designed to be used with BCEWithLogitsLoss for training.
            For inference, apply sigmoid() and threshold to get binary predictions.
        
        Notes
        -----
        This method processes temporal sequences causally (only using past and
        current information, not future frames) to enable real-time inference.
        The output is raw logits (no sigmoid/softmax applied) for compatibility
        with BCEWithLogitsLoss.
        """
        # Convert numpy to torch if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        else:
            x = x.to(self.device)
        
        # Ensure input has batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (time, features) → (batch=1, time, features)
        
        # Pass through temporal model
        output = self.temporal_model(x)
        
        # Return raw logits (no activation function applied)
        # These are designed for BCEWithLogitsLoss which applies sigmoid internally
        return output
    
    def process(self, activations):
        """
        Process activations and return beat/downbeat times.
        
        This method is designed to match the interface of particle_filter_cascade.process()
        for easy integration into BeatNet.
        
        Parameters
        ----------
        activations : np.ndarray
            Array of shape (num_frames, 2) with beat and downbeat activations [0-1]
        
        Returns
        -------
        np.ndarray
            Array of shape (num_beats, 2) with beat times and types:
            - Column 0: Beat onset time in seconds
            - Column 1: Beat type (1.0=downbeat, 2.0=regular beat)
        
        Notes
        -----
        This method:
        1. Passes activations through the temporal model (gets raw logits)
        2. Applies sigmoid to convert logits to [0, 1] probabilities
        3. Applies post-processing to extract beat times
        4. Classifies beats as downbeats or regular beats
        5. Returns results in the same format as particle_filter_cascade
        """
        # Convert to tensor: (num_frames, 2) → (1, num_frames, 2)
        activations_tensor = torch.from_numpy(activations).float().unsqueeze(0)
        
        # Forward pass through temporal model (returns raw logits)
        with torch.no_grad():
            logits = self(activations_tensor)  # (1, num_frames, 2) raw logits
            # Apply sigmoid to convert logits to probabilities [0, 1]
            refined_activations = torch.sigmoid(logits)
        
        # Extract beat and downbeat activations
        refined_activations = refined_activations.squeeze(0).cpu().numpy()  # (num_frames, 2)
        beat_activations = refined_activations[:, 0]
        downbeat_activations = refined_activations[:, 1]
        
        # Post-processing: Find beat times using peak detection
        # Parameters: frame rate is 50 Hz (20ms per frame)
        fps = 50
        frame_duration = 1.0 / fps
        
        # Find beat onsets using simple thresholding + NMS (non-maximum suppression)
        beat_threshold = 0.4
        downbeat_threshold = 0.4
        min_beat_interval = 0.25  # Minimum 0.25 seconds between beats (max 240 BPM)
        min_beat_interval_frames = int(min_beat_interval * fps)
        
        beats = []
        
        # Simple peak detection: find frames where beat activation crosses threshold
        is_beat = beat_activations > beat_threshold
        beat_frames = np.where(is_beat)[0]
        
        # Cluster consecutive beat frames and take the peak of each cluster
        if len(beat_frames) > 0:
            current_cluster = [beat_frames[0]]
            
            for frame in beat_frames[1:]:
                if frame - current_cluster[-1] <= min_beat_interval_frames:
                    current_cluster.append(frame)
                else:
                    # End of cluster: find peak and classify
                    peak_frame = current_cluster[np.argmax(
                        beat_activations[current_cluster]
                    )]
                    beat_time = peak_frame * frame_duration
                    
                    # Classify as downbeat if downbeat activation is high
                    is_downbeat = downbeat_activations[peak_frame] > downbeat_threshold
                    beat_type = 1.0 if is_downbeat else 2.0
                    
                    beats.append([beat_time, beat_type])
                    
                    current_cluster = [frame]
            
            # Process final cluster
            if current_cluster:
                peak_frame = current_cluster[np.argmax(
                    beat_activations[current_cluster]
                )]
                beat_time = peak_frame * frame_duration
                is_downbeat = downbeat_activations[peak_frame] > downbeat_threshold
                beat_type = 1.0 if is_downbeat else 2.0
                beats.append([beat_time, beat_type])
        
        # Convert to numpy array
        if beats:
            beats = np.array(beats)
        else:
            beats = np.empty((0, 2))
        
        return beats
    
    def get_config(self):
        """
        Return model configuration dictionary for saving/loading.
        
        Returns
        -------
        dict
            Configuration dictionary with all model hyperparameters
        """
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'device': self.device,
            'architecture': self.architecture,
            'output_dim': self.output_dim,
        }
