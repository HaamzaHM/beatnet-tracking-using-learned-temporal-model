# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>


# This is the script handler of the BeatNet. First, it extracts the input embeddings of the current frame or the whole song, depending on the working mode. 
# Then by feeding them into the selected pre-trained model, it calculates the beat/downbeat activation probabilities.
# Finally, it infers beats and downbeats of the current frame/song based on one of the four performance modes and selected inference method.

import os
import torch
import numpy as np
from madmom.features import DBNDownBeatTrackingProcessor
from BeatNet.particle_filtering_cascade import particle_filter_cascade
from BeatNet.ltm_model import LearnedTemporalModel
from BeatNet.log_spect import LOG_SPECT
import librosa
import sys
from BeatNet.model import BDA
try:
    import pyaudio
except ImportError:
    pyaudio = None
import matplotlib.pyplot as plt
import time
import threading


class BeatNet:

    '''
    The main BeatNet handler class including different trained models, different modes for extracting the activation and causal and non-causal inferences

        Parameters
        ----------
        Inputs: 
            model: An scalar in the range [1,3] to select which pre-trained CRNN models to utilize. 
            mode: An string to determine the working mode. i.e. 'stream', 'realtime', 'online' and ''offline.
                'stream' mode: Uses the system microphone to capture sound and does the process in real-time. Due to training the model on standard mastered songs, it is highly recommended to make sure the microphone sound is as loud as possible. Less reverbrations leads to the better results.  
                'Realtime' mode: Reads an audio file chunk by chunk, and processes each chunck at the time.
                'Online' mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then infers the parameters on interest using particle filtering.
                'offline' mode: Reads the whole audio and feeds it into the BeatNet CRNN at the same time and then inferes the parameters on interest using madmom dynamic Bayesian network. This method is quicker that madmom beat/downbeat tracking.
            inference model: A string to choose the inference approach. i.e. 'PF' standing for Particle Filtering for causal inferences and 'DBN' standing for Dynamic Bayesian Network for non-causal usages.
            algorithm: A string to choose the temporal inference algorithm. i.e. 'PF' standing for Particle Filtering (default) and 'LTM' standing for Learned Temporal Model. (Default: 'PF')
            plot: A list of strings to plot. 
                'activations': Plots the neural network activations for beats and downbeats of each time frame. 
                'beat_particles': Plots beat/tempo tracking state space and current particle states at each time frame.
                'downbeat_particles': Plots the downbeat/meter tracking state space and current particle states at each time frame.
                Note that to speedup plotting the figures, rather than new plots per frame, the previous plots get updated. However, to secure realtime results, it is recommended to not plot or have as less number of plots as possible at the time.   
            threading: To decide whether accomplish the inference at the main thread or another thread. 
            device: type of dvice. cpu or cuda:i

        Outputs:
            A vector including beat times and downbeat identifier columns, respectively with the following shape: numpy_array(num_beats, 2).
    '''
    
    
    def __init__(self, model, mode='online', inference_model='PF', algorithm='PF', plot=[], thread=False, device='cpu'):
        self.model = model
        self.mode = mode
        self.inference_model = inference_model
        self.algorithm = algorithm
        self.plot = plot
        self.thread = thread
        self.device = device
        if plot and thread:
            raise RuntimeError('Plotting cannot be accomplished in the threading mode')
        # Validate algorithm parameter
        if self.algorithm not in ['PF', 'LTM']:
            raise RuntimeError('algorithm can be either "PF" (Particle Filter) or "LTM" (Learned Temporal Model)')
        self.sample_rate = 22050
        self.log_spec_sample_rate = self.sample_rate
        self.log_spec_hop_length = int(20 * 0.001 * self.log_spec_sample_rate)
        self.log_spec_win_length = int(64 * 0.001 * self.log_spec_sample_rate)
        self.proc = LOG_SPECT(sample_rate=self.log_spec_sample_rate, win_length=self.log_spec_win_length,
                             hop_size=self.log_spec_hop_length, n_bands=[24], mode = self.mode)
        if self.algorithm == "PF":                 # instantiating a Particle Filter decoder - Is Chosen for online inference
            self.estimator = particle_filter_cascade(beats_per_bar=[], fps=50, plot=self.plot, mode=self.mode)
        elif self.algorithm == "LTM":                # instantiating a Learned Temporal Model decoder
            # Initialize LearnedTemporalModel with TCN architecture
            # Input: beat and downbeat activations from CRNN/BDA
            # Hidden: 128-dim representations for temporal modeling
            # Layers: 4 stacked TCN/Transformer blocks for causal processing
            self.estimator = LearnedTemporalModel(
                input_dim=2,           # beat, downbeat activations
                hidden_dim=128,        # internal representation size
                num_layers=4,          # number of temporal layers
                device=self.device,
                architecture='tcn',    # use Temporal Convolutional Network
                output_dim=2,          # output: beat, downbeat predictions
                dropout=0.1
            )
        
        # Note: inference_model parameter is kept for backward compatibility with DBN (madmom)
        if self.inference_model == "DBN":                # instantiating an HMM decoder - Is chosen for offline inference
            self.estimator = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        elif self.inference_model not in ["PF", "DBN"]:
            raise RuntimeError('inference_model can be either "PF" or "DBN"')
        
        script_dir = os.path.dirname(__file__)
        #assiging a BeatNet CRNN instance to extract joint beat and downbeat activations
        self.model = BDA(272, 150, 2, self.device)   #Beat Downbeat Activation detector
        #loading the pre-trained BeatNet CRNN weigths
        if model == 1:  # GTZAN out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_1_weights.pt')), strict=False)
        elif model == 2:  # Ballroom out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_2_weights.pt')), strict=False)
        elif model == 3:  # Rock_corpus out trained model
            self.model.load_state_dict(torch.load(os.path.join(script_dir, 'models/model_3_weights.pt')), strict=False)
        else:
            raise RuntimeError(f'Failed to open the trained model: {model}')
        self.model.eval()
        if self.mode == 'stream':
            if pyaudio is None:
                raise RuntimeError('pyaudio is required for streaming mode. Please install it with: pip install pyaudio')
            self.stream_window = np.zeros(self.log_spec_win_length + 2 * self.log_spec_hop_length, dtype=np.float32)                                          
            self.stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32,
                                             channels=1,
                                             rate=self.sample_rate,
                                             input=True,
                                             frames_per_buffer=self.log_spec_hop_length,)
                                             
    def process(self, audio_path=None):
        # FULL PIPELINE: Audio → Feature Extraction → CRNN/BDA Activations → Temporal Inference → Beat/Downbeat Output
        # Orchestrates the complete beat and downbeat tracking pipeline across different modes (stream, realtime, online, offline)
        # Temporal inference can use either Particle Filtering (PF) or Learned Temporal Model (LTM) via the algorithm parameter
        if self.mode == "stream":
            if self.inference_model != "PF":
                    raise RuntimeError('The infernece model should be set to "PF" for the streaming mode!')
            if self.algorithm == "LTM":
                raise NotImplementedError("LTM inference not implemented yet")
            self.counter = 0
            while self.stream.is_active():
                self.activation_extractor_stream()  # Using BeatNet causal Neural network streaming mode to extract activations
                if self.thread:
                    x = threading.Thread(target=self.estimator.process, args=(self.pred), daemon=True)   # Processing the inference in another thread 
                    x.start()
                    x.join()    
                else:
                    # Particle filtering inference: activations → beat/downbeat tracking
                    output = self.estimator.process(self.pred)
                self.counter += 1

                
        elif self.mode == "realtime":
            self.counter = 0
            self.completed = 0
            if self.inference_model != "PF":
                raise RuntimeError('The infernece model for the streaming mode should be set to "PF".')
            if self.algorithm == "LTM":
                raise NotImplementedError("LTM inference not implemented yet")
            if isinstance(audio_path, str) or audio_path.all()!=None:
                while self.completed == 0:
                    self.activation_extractor_realtime(audio_path) # Using BeatNet causal Neural network realtime mode to extract activations
                    if self.thread:
                        x = threading.Thread(target=self.estimator.process, args=(self.pred), daemon=True)   # Processing the inference in another thread 
                        x.start()
                        x.join()    
                    else:
                        # Particle filtering inference: activations → beat/downbeat tracking
                        output = self.estimator.process(self.pred)  # Using particle filtering online inference to infer beat/downbeats
                    self.counter += 1
                return output
            else:
                raise RuntimeError('An audio object or file directory is required for the realtime usage!')
        
        
        elif self.mode == "online":
            if isinstance(audio_path, str) or audio_path.all()!=None:
                preds = self.activation_extractor_online(audio_path)    # Using BeatNet causal Neural network to extract activations
            else:
                raise RuntimeError('An audio object or file directory is required for the online usage!')
            if self.algorithm == "PF":   # Particle filtering inference (causal)
                # Particle filtering inference: activations → beat/downbeat tracking
                output = self.estimator.process(preds)  # Using particle filtering online inference to infer beat/downbeats
                return output
            elif self.algorithm == "LTM":   # Learned Temporal Model inference (causal)
                # Learned Temporal Model inference: activations → beat/downbeat tracking
                # The LTM processes activations through TCN/Transformer to refine predictions
                # and extract beat/downbeat times using post-processing
                output = self.estimator.process(preds)  # Using LTM causal inference to infer beat/downbeats
                return output
            elif self.inference_model == "DBN":    # Dynamic bayesian Network Inference (non-causal)
                output = self.estimator(preds)  # Using DBN offline inference to infer beat/downbeats
                return output
        
        
        elif self.mode == "offline":
                if self.inference_model != "DBN":
                    raise RuntimeError('The infernece model should be set to "DBN" for the offline mode!')
                if isinstance(audio_path, str) or audio_path.all()!=None:
                    preds = self.activation_extractor_online(audio_path)    # Using BeatNet causal Neural network to extract activations
                    # Dynamic Bayesian Network inference: activations → beat/downbeat tracking (non-causal)
                    output = self.estimator(preds)  # Using DBN offline inference to infer beat/downbeats
                    return output
        
                else:
                    raise RuntimeError('An audio object or file directory is required for the offline usage!')
                

    def activation_extractor_stream(self):
        # TODO: 
        ''' Streaming window
        Given the training input window's origin set to center, this streaming data formation causes 0.084 (s) delay compared to the trained model that needs to be fixed. 
        '''
        # PIPELINE: Streaming audio (microphone) → Log-mel spectrogram → BDA (CRNN) → Beat/Downbeat activations
        # Input: Real-time audio stream from microphone (hop_length samples per call)
        # Output: Beat and downbeat activation probabilities for the current frame (1x2 array)
        with torch.no_grad():
            hop = self.stream.read(self.log_spec_hop_length)
            hop = np.frombuffer(hop, dtype=np.float32)
            self.stream_window = np.append(self.stream_window[self.log_spec_hop_length:], hop)
            if self.counter < 5:
                self.pred = np.zeros([1,2])
            else:
                feats = self.proc.process_audio(self.stream_window).T[-1]
                feats = torch.from_numpy(feats)
                feats = feats.unsqueeze(0).unsqueeze(0).to(self.device)
                pred = self.model(feats)[0]
                pred = self.model.final_pred(pred)
                
                # Accumulate activations for LTM if needed (streaming mode)
                if self.algorithm == "LTM":
                    if not hasattr(self, 'ltm_activations'):
                        self.ltm_activations = []
                    # pred shape: (3, 1) for current frame
                    self.ltm_activations.append(pred.cpu().detach().numpy().squeeze())
                
                pred_numpy = pred.cpu().detach().numpy()
                self.pred = np.transpose(pred_numpy[:2, :])


    def activation_extractor_realtime(self, audio_path):
        # PIPELINE: Audio file chunks → Log-mel spectrogram → BDA (CRNN) → Beat/Downbeat activations
        # Input: Audio file path or numpy array; processes one chunk per call
        # Output: Beat and downbeat activation probabilities for the current chunk (1x2 array)
        with torch.no_grad():
            if self.counter==0: #loading the audio
                if isinstance(audio_path, str):
                    self.audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading the data
                elif len(np.shape(audio_path))>1:
                    self.audio = np.mean(audio_path ,axis=1)
                else:
                    self.audio = audio_path
                if self.algorithm == "LTM":
                    # Initialize buffer to accumulate all activations for LTM
                    self.ltm_activations = []
            if self.counter<(round(len(self.audio)/self.log_spec_hop_length)):
                if self.counter<2:
                    self.pred = np.zeros([1,2])
                else:
                    feats = self.proc.process_audio(self.audio[self.log_spec_hop_length * (self.counter-2):self.log_spec_hop_length * (self.counter) + self.log_spec_win_length]).T[-1]
                    feats = torch.from_numpy(feats)
                    feats = feats.unsqueeze(0).unsqueeze(0).to(self.device)
                    pred = self.model(feats)[0]
                    pred = self.model.final_pred(pred)
                    
                    # Accumulate activations for LTM if needed
                    if self.algorithm == "LTM":
                        # pred shape: (3, 1) for current frame - store all 3 activations
                        self.ltm_activations.append(pred.cpu().detach().numpy().squeeze())
                    
                    pred_numpy = pred.cpu().detach().numpy()
                    self.pred = np.transpose(pred_numpy[:2, :])
            else:
                self.completed = 1


    def activation_extractor_online(self, audio_path):
        # PIPELINE: Full audio → Log-mel spectrogram → BDA (CRNN) → Beat/Downbeat activations
        # Input: Audio file path or numpy array (entire audio)
        # Output: Beat and downbeat activation probabilities for all frames (Nx2 array, where N=num_frames)
        with torch.no_grad():
            if isinstance(audio_path, str):
                audio, _ = librosa.load(audio_path, sr=self.sample_rate)  # reading the data
            elif len(np.shape(audio_path))>1:
                audio = np.mean(audio_path ,axis=1)
            else:
                audio = audio_path
            # Feature extraction: audio → log-mel spectrogram
            feats = self.proc.process_audio(audio).T
            feats = torch.from_numpy(feats)
            feats = feats.unsqueeze(0).to(self.device)
            # CRNN/BDA activation computation: spectrogram → beat/downbeat activations
            preds = self.model(feats)[0]  # extracting the activations by passing the feature through the NN
            preds = self.model.final_pred(preds)
            preds_numpy = preds.cpu().detach().numpy()
            
            # Construct ltm_input tensor for LearnedTemporalModel if needed
            if self.algorithm == "LTM":
                # Stack beat, downbeat, and nonbeat activations into (1, T, 3) tensor
                # preds shape after softmax: (3, T) where T is number of frames
                # We want: (1, T, 3) where dim 0 is batch, dim 1 is time, dim 2 is [beat, downbeat, nonbeat]
                ltm_input = preds.unsqueeze(0).transpose(1, 2)  # (1, T, 3)
                # TODO: pass ltm_input through LearnedTemporalModel and use its outputs instead of particle filter
                self.ltm_input = ltm_input
            
            preds_numpy = np.transpose(preds_numpy[:2, :])
        return preds_numpy

