# Author: Mojtaba Heydari <mheydari@ur.rochester.edu>


import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from madmom.features.beats_hmm import BarStateSpace, BarTransitionModel     # importing the bar pointer state space implemented in Madmom
from madmom.ml.hmm import TransitionModel, ObservationModel

class BDObservationModel(ObservationModel):
    """
    Observation model for beat and downbeat tracking with particle filtering.

    Parameters
    ----------
    state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
    observation_lambda : str
        Based on the first character of this parameter, each (down-)beat period gets split into (down-)beat states
        "B" stands for border model which classifies 1/(observation lambda) fraction of states as downbeat states and
        the rest as the beat states (if it is used for downbeat tracking state space) or the same fraction of states
        as beat states and the rest as the none beat states (if it is used for beat tracking state space).
        "N" model assigns a constant number of the beginning states as downbeat states and the rest as beat states
         or beginning states as beat and the rest as none-beat states
        "G" model is a smooth Gaussian transition (soft border) between downbeat/beat or beat/none-beat states

    """

    def __init__(self, state_space, observation_lambda):

        if observation_lambda[0] == 'B':
            observation_lambda = int(observation_lambda[1:])
            # compute observation pointers
            # always point to the non-beat densities
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            # unless they are in the beat range of the state space
            border = 1. / observation_lambda
            pointers[state_space.state_positions % 1 < border] = 1
            # the downbeat (i.e. the first beat range) points to density column 2
            pointers[state_space.state_positions < border] = 2
            # instantiate a ObservationModel with the pointers
            super(BDObservationModel, self).__init__(pointers)

        elif observation_lambda[0] == 'N':
            observation_lambda = int(observation_lambda[1:])
            # compute observation pointers
            # always point to the non-beat densities
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            # unless they are in the beat range of the state space
            for i in range(observation_lambda):
                border = np.asarray(state_space.first_states) + i
                pointers[border[1:]] = 1
                # the downbeat (i.e. the first beat range) points to density column 2
                pointers[border[0]] = 2
                # instantiate a ObservationModel with the pointers
            super(BDObservationModel, self).__init__(pointers)

        elif observation_lambda[0] == 'G':
            observation_lambda = float(observation_lambda[1:])
            pointers = np.zeros((state_space.num_beats + 1, state_space.num_states))
            for i in range(state_space.num_beats + 1):
                pointers[i] = gaussian(state_space.state_positions, i, observation_lambda)
            pointers[0] = pointers[0] + pointers[-1]
            pointers[1] = np.sum(pointers[1:-1], axis=0)
            pointers = pointers[:2]
            super(BDObservationModel, self).__init__(pointers)


def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.) / 2)  


def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.) / 2)  


def beat_densities(observations, observation_model, state_model):
    """
    Convert beat activation probability to particle weights for the beat filter.
    
    This function maps a single beat activation value (0-1) to a weight array
    with one weight per state in the beat state space. States corresponding to
    "beat" parts of the beat interval get the activation value, while "non-beat"
    states get a small baseline weight (0.03).
    
    Purpose: Allows particle resampling to favor particles at beat onsets.
    
    Parameters
    ----------
    observations : float or array
        Beat activation probability from CRNN/BDA model (0-1).
    observation_model : BDObservationModel
        Maps state space positions to beat/non-beat designations.
    state_model : BarStateSpace
        The beat state space (tempo x phase).
    
    Returns
    -------
    new_obs : numpy.ndarray, shape (num_states,)
        Weight for each state. High at beat onset, low elsewhere.
        Sum ≠ 1 (not normalized, done later by resampler).
    """
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # B or N
        new_obs[np.argwhere(observation_model.pointers == 2)] = observations
        new_obs[np.argwhere(observation_model.pointers == 0)] = 0.03
    elif len(np.shape(observation_model.pointers)) == 2:  # G
        new_obs = observation_model.pointers[0] * observations
        new_obs[new_obs < 0.005] = 0.03
    return new_obs

def down_densities(observations, observation_model, state_model):
    """
    Convert beat/downbeat activations to particle weights for the downbeat filter.
    
    This function maps beat and downbeat activation probabilities (from CRNN/BDA)
    to a weight array for the downbeat state space. States at meter position 1
    (downbeat/bar start) are weighted by the downbeat activation, while other
    positions are weighted by the beat activation.
    
    Purpose: Allows particle resampling to favor particles at downbeats vs regular beats.
    
    Parameters
    ----------
    observations : array, shape (2,)
        [beat_activation, downbeat_activation] from CRNN/BDA model (0-1 each).
    observation_model : BDObservationModel
        Maps meter positions to downbeat/beat designations.
    state_model : BarStateSpace
        The downbeat state space (beats per bar positions).
    
    Returns
    -------
    new_obs : numpy.ndarray, shape (num_states,)
        Weight for each downbeat state. High at downbeats (pos=1) or beats (other pos).
        Sum ≠ 1 (not normalized, done later by resampler).
    """
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # B or N
        new_obs[np.argwhere(
            observation_model.pointers == 2)] = observations[1]
        new_obs[np.argwhere(
            observation_model.pointers == 0)] = observations[0]
    elif len(np.shape(observation_model.pointers)) == 2:  # G
        new_obs = observation_model.pointers[0] * observations
        new_obs[new_obs < 0.005] = 0.03
    return new_obs

#   assigning downbeat vs beat weights - second model
def down_densities2(observations, beats_per_bar):
    """
    Alternative downbeat weighting function (simpler than down_densities).
    
    This is a simplified version that directly assigns activation values to
    meter positions without going through an observation model. Position 0
    (downbeat) gets downbeat activation, all other positions get beat activation.
    
    Parameters
    ----------
    observations : array, shape (2,)
        [beat_activation, downbeat_activation] from CRNN/BDA model (0-1 each).
    beats_per_bar : int
        Number of beats per bar (e.g., 4 for 4/4 time).
    
    Returns
    -------
    new_obs : numpy.ndarray, shape (beats_per_bar,)
        Weight for each meter position.
        - new_obs[0] = downbeat_activation (position 1 of bar)
        - new_obs[1:] = beat_activation (positions 2-4 of bar)
    """
    new_obs = np.zeros(beats_per_bar, float)
    new_obs[0] = observations[1]  # downbeat activation
    new_obs[1:] = observations[0]  # beat activation
    return new_obs

#   Inference initialization
class particle_filter_cascade:
    np.random.seed(1)
    PARTICLE_SIZE = 1500   #  1500
    DOWN_PARTICLE_SIZE = 250  # 250
    MIN_BPM = 55.
    MAX_BPM = 215.  
    NUM_TEMPI = 300
    LAMBDA_B = 60  # beat transition lambda
    LAMBDA_D = 0.1  # downbeat transition lambda
    OBSERVATION_LAMBDA_B = "B56"  # beat observation lambda
    OBSERVATION_LAMBDA_D = "B56"  # downbeat observation lambda
    fps = 50
    T = 1 / fps
    MIN_BEAT_PER_BAR = 2
    MAX_BEAT_PER_BAR = 4
    OFFSET = 0 # The point of time after which the inference model starts to work. Can be zero!
    IG_THRESHOLD = 0.4  # Information Gate threshold

    def __init__(self, beats_per_bar=[], particle_size=PARTICLE_SIZE, down_particle_size=DOWN_PARTICLE_SIZE,
                 min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI, min_beats_per_bar=MIN_BEAT_PER_BAR,
                 max_beats_per_bar=MAX_BEAT_PER_BAR, offset=OFFSET, ig_threshold=IG_THRESHOLD, lambda_b=LAMBDA_B,
                 lambda_d=LAMBDA_D, observation_lambda_b=OBSERVATION_LAMBDA_B, observation_lambda_d=OBSERVATION_LAMBDA_D,
                 fps=None, plot=False, mode=None, **kwargs):
        """
        Initialize the two-stage particle filter cascade for beat and downbeat tracking.
        
        This constructor sets up:
        1. Beat tracking state space (tempo tracking via particles)
        2. Downbeat tracking state space (meter tracking via particles)
        3. Observation models (how to weight particles based on CRNN/BDA activations)
        4. Transition models (how particles move through states over time)
        5. Particle populations for both stages
        6. Optional visualization infrastructure
        
        Parameters
        ----------
        beats_per_bar : list, optional
            Manually set beats per bar (e.g., [3, 4] for 3/4 or 4/4 time signatures).
            If empty, will automatically detect between min_beats_per_bar and max_beats_per_bar.
        particle_size : int, default=1500
            Number of particles for beat tracking. More = more accurate but slower.
        down_particle_size : int, default=250
            Number of particles for downbeat tracking. More = more accurate but slower.
        min_bpm : float, default=55
            Minimum tempo to track (beats per minute).
        max_bpm : float, default=215
            Maximum tempo to track (beats per minute).
        num_tempi : int, default=300
            Number of tempo bins between min_bpm and max_bpm (state space resolution).
        min_beats_per_bar : int, default=2
            Minimum beats per bar for meter tracking.
        max_beats_per_bar : int, default=4
            Maximum beats per bar for meter tracking.
        offset : float, default=0
            Time offset (seconds) before inference starts.
        ig_threshold : float, default=0.4
            Information gate threshold - activations below this are treated as silence.
        lambda_b : float, default=60
            Beat transition lambda - higher = more flexible tempo changes.
        lambda_d : float, default=0.1
            Downbeat transition lambda - higher = more flexible meter changes.
        observation_lambda_b : str, default="B56"
            Beat observation model type (B=border, N=numeric, G=gaussian).
        observation_lambda_d : str, default="B56"
            Downbeat observation model type (B=border, N=numeric, G=gaussian).
        fps : int, optional
            Frame rate of input activations (50 Hz is standard).
        plot : bool or str, optional
            Enable visualization. Can be True or contain 'activations', 'beat_particles', 'downbeat_particles'.
        mode : str, optional
            Inference mode: 'stream', 'realtime', 'online', or 'offline'. Affects plotting behavior.
        """
        # ===== STEP 1: Store configuration parameters =====
        # These control the inference behavior and state space resolution
        self.particle_size = particle_size
        self.down_particle_size = down_particle_size
        self.particle_filter = []
        self.beats_per_bar = beats_per_bar
        self.fps = fps
        self.Lambda_b = lambda_b
        self.Lambda_d = lambda_d
        self.observation_lambda_b = observation_lambda_b
        self.observation_lambda_d = observation_lambda_d
        self.plot = plot
        self.min_beats_per_bar = min_beats_per_bar
        self.max_beats_per_bar = max_beats_per_bar
        self.offset = offset
        self.ig_threshold = ig_threshold
        self.mode = mode
        
        # ===== STEP 2: Create Beat State Space =====
        # Convert BPM to frames: min/max beat intervals define the state space dimensions
        # 1 beat interval = time between beat onsets (inversely proportional to BPM)
        min_interval = 60. * fps / max_bpm  # faster tempo = shorter interval
        max_interval = 60. * fps / min_bpm  # slower tempo = longer interval
        self.st = BarStateSpace(1, min_interval, max_interval, num_tempi)    # beat tracking state space with 300 tempo bins
        if beats_per_bar:   # if the number of beats per bar is given
            self.st2 = BarStateSpace(1, min(self.beats_per_bar ), max(self.beats_per_bar),
                                max(self.beats_per_bar ) - min(self.beats_per_bar) + 1)   # downbeat tracking state space
        else:   # if the number of beats per bar is not given
            self.st2 = BarStateSpace(1, self.min_beats_per_bar, self.max_beats_per_bar, self.max_beats_per_bar - self.min_beats_per_bar + 1)  # downbeat tracking state space
        
        # ===== STEP 3: Create Observation Models =====
        # Observation models map activation values to particle weights
        # They define which parts of the state space are "beat" vs "non-beat"
        tm = BarTransitionModel(self.st, self.Lambda_b)
        self.tm = list(TransitionModel.make_dense(tm.states, tm.pointers, tm.probabilities))   # beat transition model (allows tempo changes)
        self.om = BDObservationModel(self.st, self.observation_lambda_b)   # beat observation model
        self.st.last_states = list(np.concatenate(self.st.last_states).flat)    # beat last states
        self.om2 = BDObservationModel(self.st2, self.observation_lambda_d)  # downbeat observation model
        
        # ===== STEP 4: Create Downbeat Transition Model =====
        # This is a simple transition matrix for cycling through meter positions
        # Particles either stay in same position (prob=1-lambda_d) or jump to another position (prob=lambda_d)
        # This allows the meter to occasionally change (e.g., switch between 3/4 and 4/4)
        self.tm2 = np.zeros((len(self.st2.first_states[0]), len(self.st2.first_states[0])))  # downbeat transition model
        for i in range(len(self.st2.first_states[0])):
            for j in range(len(self.st2.first_states[0])):
                if i == j:
                    self.tm2[i, j] = 1 - self.Lambda_d  # stay in current meter position
                else:
                    self.tm2[i, j] = self.Lambda_d / (len(self.st2.first_states[0]) - 1)  # switch to different meter
        pass
        self.T = 1 / self.fps
        self.counter = -1
        self.path = np.zeros((1, 2), dtype=float)
        
        # ===== STEP 5: Initialize Particles =====
        # Particles are initialized randomly in the state space
        # They will be iteratively moved and reweighted during processing
        self.particles = np.sort(np.random.choice(np.arange(0, self.st.num_states - 1), self.particle_size, replace=True))
        self.down_particles = np.sort(np.random.choice(np.arange(0, self.st2.num_states - 1), self.down_particle_size, replace=True))  
        self.beat = np.squeeze(self.st.first_states)
        
        # ===== STEP 6: Initialize Visualization (Optional) =====
        # If plotting is enabled, set up matplotlib figures for real-time visualization
        if 'activations' in self.plot and (self.mode == 'stream' or self.mode == 'realtime'):
            f1, self.subplot1 = plt.subplots(figsize=(100, 40), dpi=5)
            self.subplot1.set_ylim([0,1])
            self.beats_activation_show = np.zeros(100)
            self.downbeats_activation_show = np.zeros(100)
            self.subplot1.plot(self.beats_activation_show, color='black', label='Beat Activations', linewidth=15) # Make a new line for the beat activations
            self.subplot1.set_xlabel("t", size=200)
            self.subplot1.set_ylabel("activations likelihood", size=200)
            self.subplot1.set_title("Activations plot", size=200)
            self.subplot1.plot(self.downbeats_activation_show , color='purple', label='Downbeat Activations', linewidth=15) # Make a new line for the downbeat activations
            self.activation_lines = self.subplot1.get_lines() # Obtain a list of lines on the plot
            
        if 'beat_particles' in self.plot:
            f2, self.subplot2 = plt.subplots(figsize=(30, 10), dpi=50)
            self.beat_particles_show = self.subplot2.collections # setting up beat particle collections to display
            position_beats = np.median(self.st.state_positions[self.particles])  
            m = np.r_[True, self.particles[:-1] != self.particles[1:], True]
            counts = np.diff(np.flatnonzero(m))
            unq = self.particles[m[:-1]]
            part = np.c_[unq, counts]
            self.subplot2.scatter(self.st.state_positions, np.max(self.st.state_intervals)-self.st.state_intervals, marker='o', color='grey', alpha=0.12)
            self.subplot2.scatter(self.st.state_positions[part[:, 0]], np.max(self.st.state_intervals) - self.st.state_intervals[part[:, 0]], marker='o', s=part[:, 1] * 2, color="red")
            self.subplot2.scatter(self.st.state_positions[self.om.pointers == 2], np.max(self.st.state_intervals) - self.st.state_intervals[self.om.pointers == 2], marker='o', s=50, color='yellow', alpha=0)
            self.subplot2.set_xlabel("ϕ_b: Phase of the frame within the beat interval", size=20)
            self.subplot2.set_ylabel("ϕ'_b: Tempo", size=20)
            self.subplot2.set_title("Beat particle states", size=20)
            self.beat_particles_swarm = self.subplot2.axvline(x=position_beats) # setting up beat particle average to display
            
        if 'downbeat_particles' in self.plot:    
            f3, self.subplot3 = plt.subplots(figsize=(30, 10), dpi=50)
            self.downbeat_particles_show = self.subplot3.collections
            m = np.bincount(self.down_particles)
            self.down_max = np.argmax(m)  # calculating downbeat particles clutter
            position_downs = np.max(self.st2.state_positions[self.down_max])
            m1 = np.r_[True, self.down_particles[:-1] != self.down_particles[1:], True]
            counts1 = np.diff(np.flatnonzero(m1))
            unq1 = self.down_particles[m1[:-1]]
            part1 = np.c_[unq1, counts1]
            self.subplot3.scatter(self.st2.state_positions, np.max(self.st2.state_intervals) - self.st2.state_intervals, marker='o', color='grey', alpha=0.12, s=50)
            self.subplot3.scatter(self.st2.state_positions[part1[:, 0]], np.max(self.st2.state_intervals) - self.st2.state_intervals[part1[:, 0]], marker='o', s=part1[:, 1] * 4, color="red")
            self.subplot3.scatter(self.st2.state_positions[self.om2.pointers == 2], np.max(self.st2.state_intervals) - self.st2.state_intervals[self.om2.pointers == 2], marker='o',s=100, color='green', alpha=0)
            self.subplot3.set_xlabel("ϕ_d: Phase of the beat within the bar interval", size=20)
            self.subplot3.set_ylabel("ϕ'_d: Meter", size=20)
            self.subplot3.set_title("Downbeat particle states", size=20)
            self.down_particles_swarm = self.subplot3.axvline(x=position_downs) # setting up downbeat particle average to display
        
            #plt.show(block=False)

    def process(self, activations):

        """
        Run the two-stage particle filter cascade on CRNN/BDA activations to infer beat/downbeat times.
        
        This is the main inference method. It processes activation probabilities frame-by-frame:
        
        PIPELINE:
        1. Apply information gate threshold (ignore weak activations)
        2. For each frame:
           a. BEAT STAGE: Advance beat particles, weight by beat activation, resample
           b. If beat is detected: trigger downbeat stage
           c. DOWNBEAT STAGE: Advance downbeat particles, weight by downbeat activation, resample
           d. Determine: Is this a downbeat (type=1.0) or regular beat (type=2.0)?
        3. Return array of detected beat times and types
        
        INPUT/OUTPUT SHAPES AND MEANINGS:
        Input activations have shape (num_frames, 2) where:
          - Column 0: Beat activation probability (0-1) from CRNN/BDA model
          - Column 1: Downbeat activation probability (0-1) from CRNN/BDA model
        
        Output (self.path) has shape (num_beats, 2) where:
          - Column 0: Beat onset time in seconds
          - Column 1: Beat type (1.0 = downbeat/bar start, 2.0 = regular beat)
        
        EXAMPLE WORKFLOW FOR 3 FRAMES:
          Frame 0: activations=[0.1, 0.05] → particles scattered, no beat detected
          Frame 1: activations=[0.9, 0.05] → beat particles clustered → BEAT DETECTED (type=2.0)
          Frame 2: activations=[0.85, 0.8] → beat+downbeat → BEAT DETECTED (type=1.0, downbeat)
        
        PARTICLE FILTER MECHANICS:
          - Motion: Particles advance through state space (tempo for beat, meter for downbeat)
          - Correction: Particles reweighted by observation likelihood (from activations)
          - Resampling: Low-weight particles eliminated, high-weight particles duplicated
          - Detection: When particles cluster at transition points, beats are detected

        Parameters
        ----------
        activations : numpy.ndarray, shape (num_frames, 2)
            Frame-by-frame CRNN/BDA model outputs:
            - activations[:, 0]: Beat activation probability for each frame [0-1]
            - activations[:, 1]: Downbeat activation probability for each frame [0-1]
            
            Typically at 50 Hz, so each frame = 0.02 seconds.

        Returns
        -------
        beats : numpy.ndarray, shape (num_beats, 2)
            Detected beats and downbeats, excluding the first dummy entry.
            - beats[:, 0]: Beat onset time in seconds
            - beats[:, 1]: Beat type (1.0=downbeat, 2.0=regular beat)
        """
        # ===== Pre-processing: Apply Information Gate =====
        # Skip the initial offset period (for models that need warmup)
        activations = activations[int(self.offset / self.T):]
        if np.shape(activations)==(2,):
            activations = np.reshape(activations, (-1, 2))
        both_activations = activations.copy()
        
        # Take max of beat/downbeat and apply threshold (ignore weak activations as silence)
        activations = np.max(activations, axis=1)
        activations[activations < self.ig_threshold] = 0.03
        self.activations = activations
        self.both_activations = both_activations
        
        if 'activations' in self.plot and (self.mode == 'online' or self.mode == 'offline'):
            self.activations_plot()

        # ===== Main Processing Loop: Frame-by-Frame Inference =====
        for i in range(len(activations)):  # loop through the provided frame/s to infer beats/downbeats
            self.counter += 1
            if 'activations' in self.plot and (self.mode == 'stream' or self.mode == 'realtime'):  # Ploting the activations
                self.activations_plot() 
            gathering = int(np.median(self.particles))   # calculating beat particles clutter
            # checking if the clutter is within the beat interval
            if ((gathering - self.beat[self.st.state_intervals[self.beat] == self.st.state_intervals[gathering]]) < (
                    int(.07 / self.T)) + 1).any() and (self.offset + self.counter * self.T) - self.path[-1][0] > .4 * self.T * \
                    self.st.state_intervals[gathering]:
                
                # ===== DOWNBEAT FILTER STAGE =====
                # This stage only runs when a beat is detected above
                # It tracks which position in the bar (1st beat, 2nd beat, 3rd beat, etc.)
                
                # MOTION: Downbeat particles advance through meter positions
                last1 = self.down_particles[np.in1d(self.down_particles, self.st2.last_states)]   
                state1 = self.down_particles[~np.in1d(self.down_particles, self.st2.last_states)] + 1
                for j in range(len(last1)):
                    arg1 = np.argwhere(self.st2.last_states[0] == last1[j])[0][0]
                    # Transition: particles at end of bar wrap to beginning (or jump via tm2)
                    nn = np.random.choice(self.st2.first_states[0], 1, p=(np.squeeze(self.tm2[arg1])))
                    state1 = np.append(state1, nn)
                self.down_particles = state1
                
                # CORRECTION: Weight downbeat particles by downbeat activation
                if both_activations[i][1]>0.7:
                    # High downbeat activation: boost particle population
                    self.down_particles = np.append(self.down_particles,np.array([self.st2.first_states]))
                obs2 = down_densities(both_activations[i], self.om2, self.st2)
                # Resample: eliminate low-weight particles, duplicate high-weight ones
                self.down_particles = universal_resample(self.down_particles, obs2[self.down_particles])  
                if both_activations[i][1]>0.7:
                    # Remove boosted particles that didn't survive resampling
                    self.down_particles = np.delete(self.down_particles, np.random.choice(self.down_particle_size, len(self.st2.first_states), replace=False))
                
                # DETECTION: Determine beat type (downbeat vs regular beat)
                m = np.bincount(self.down_particles)
                self.down_max = np.argmax(m)  # calculating downbeat particles clutter (most common position)
                
                # Output the detected beat with appropriate type
                if self.down_max in self.st2.first_states[0] and self.path[-1][1] !=1 and both_activations[i][1]>0.4:
                    # Downbeat detected: particles clustered at position 1 (bar start) with strong downbeat activation
                    self.path = np.append(self.path, [[self.offset + self.counter * self.T, 1]], axis=0)
                    if self.mode == 'stream' or self.mode == 'realtime':
                        print("*beat!")
                elif (activations[i]>0.4) :
                    # Regular beat detected: particles clustered but not at downbeat position
                    self.path = np.append(self.path, [[self.offset + self.counter * self.T, 2]], axis=0)
                    if self.mode == 'stream' or self.mode == 'realtime':
                        print("beat!")
                    #librosa.clicks(times=None, frames=None, sr=22050, hop_length=512, click_freq=440.0, click_duration=0.1, click=None, length=None)
                if 'downbeat_particles' in self.plot:
                    self.downbeat_particles_plot()

            # ===== BEAT FILTER STAGE =====
            # This stage runs every frame
            # It tracks the current tempo and detects beat onsets
            
            # MOTION: Beat particles advance through beat interval states (phase/tempo)
            last = self.particles[np.in1d(self.particles, self.st.last_states)]   
            state = self.particles[~np.in1d(self.particles, self.st.last_states)] + 1
            for j in range(len(last)):
                args = np.argwhere(self.tm[1] == last[j])
                probs = self.tm[2][args]
                # Transition: particles at end of beat wrap to beginning of next beat (allowing tempo change)
                nn = np.random.choice(np.squeeze(self.tm[0][args]), 1, p=(np.squeeze(probs)))
                state = np.append(state, nn)
            self.particles = state

            # CORRECTION: Weight beat particles by beat activation
            obs = beat_densities(activations[i], self.om, self.st)
            if activations[i] > 0.1:  # resampling is done only when there is a meaningful activation
                if activations[i] > 0.8:
                    # Very strong beat activation: boost particle population at beat onset
                    self.particles = np.append(self.particles,np.array([self.st.first_states[0][np.arange(np.random.randint(4),len(self.st.first_states[0]),6)]]))
                # Resample: eliminate low-weight particles, duplicate high-weight ones
                self.particles = universal_resample(self.particles, obs[self.particles], )  # beat correction
                if activations[i] > 0.8:
                    # Remove boosted particles that didn't survive resampling
                    np.delete(self.particles, np.random.choice(self.particle_size, len(self.st.first_states), replace=False))
            
            # Update visualization of beat particles if enabled
            if 'beat_particles' in self.plot:
                if self.counter % 1 == 0:  # choosing how often to plot
                    self.beat_particles_plot()
        
        # Return detected beats, excluding the first dummy initialization entry
        return self.path[1:]
        
        
    def activations_plot(self):
        """
        Visualize CRNN/BDA activation probabilities over time.
        
        Behavior depends on inference mode:
        - 'online'/'offline': Creates a static plot of all activations at once
        - 'stream'/'realtime': Updates a rolling window plot in real-time (last 100 frames)
        
        Shows two lines:
        - Black: Beat activation probability
        - Purple: Downbeat activation probability
        """
        if self.mode == 'online' or self.mode == 'offline':
            f, subplot1 = plt.subplots(figsize=(30,5), dpi=100)
            subplot1.set_ylim([0,1])
            subplot1.plot(self.both_activations[:,0], color='black', label='Beat Activations') # plotting beat activations
            subplot1.plot(self.both_activations[:,1] , color='purple', label='Downbeat Activations') # plotting downbeat activations
            plt.show()
        elif self.mode == 'stream' or self.mode == 'realtime':
            self.beats_activation_show = np.roll(self.beats_activation_show,1)
            self.beats_activation_show[-1] = self.both_activations[0,0]
            self.downbeats_activation_show = np.roll(self.downbeats_activation_show,1)
            self.downbeats_activation_show[-1] = self.both_activations[0,1] 
            # Update the existing beat activations
            self.activation_lines[0].set_ydata(self.beats_activation_show)
            # Update the existing downbeat activations
            self.activation_lines[1].set_ydata(self.downbeats_activation_show)
            plt.pause(0.0000000001)

            
    def beat_particles_plot(self):
        """
        Visualize beat particle distribution in tempo/phase state space.
        
        Creates a 2D scatter plot where:
        - X-axis: Phase within beat (ϕ_b, 0-1 representing position in beat interval)
        - Y-axis: Tempo (ϕ'_b, BPM converted to frames)
        - Red dots: Individual particles (sized by count)
        - Vertical line: Median particle position (current beat phase estimate)
        - Yellow shading: Beat observation region (where particles should be weighted high)
        
        Useful for debugging: If particles cluster at beat phase ≈ 0, beat detection is working.
        """
        position_beats = np.median(self.st.state_positions[self.particles])    # calculating beat clutter position    
        m = np.r_[True, self.particles[:-1] != self.particles[1:], True]
        counts = np.diff(np.flatnonzero(m))
        unq = self.particles[m[:-1]]
        part = np.c_[unq, counts]
        if self.mode == 'stream' or self.mode == 'realtime':
            current_activations = self.both_activations[0]
            current_activation = self.activations[0]
        else:
            current_activations = self.both_activations[self.counter]
            current_activation = self.activations[self.counter]
        if current_activations [0] > current_activations[1]:
            beat_color_show='yellow'
        else:
            beat_color_show='green'
        self.beat_particles_show[1].set_offsets(np.c_[self.st.state_positions[part[:, 0]], np.max(
            self.st.state_intervals) - self.st.state_intervals[part[:, 0]]])
        self.beat_particles_show[2].set_alpha(current_activation)
        self.beat_particles_show[2].set_color(beat_color_show)
        self.beat_particles_swarm.set_xdata(x=position_beats)
        plt.pause(0.000000001)
    
    def downbeat_particles_plot(self):
        """
        Visualize downbeat particle distribution in meter/phase state space.
        
        Creates a 2D scatter plot where:
        - X-axis: Phase within bar (ϕ_d, representing position in beat interval)
        - Y-axis: Meter position (which beat in bar: 1, 2, 3, 4)
        - Red dots: Individual particles (sized by count)
        - Vertical line: Most probable meter position (current downbeat estimate)
        - Green shading: Downbeat observation region (position 1 of bar)
        
        Useful for debugging: If particles cluster at position 1, downbeat detection is working.
        Particles at other positions indicate regular beats in the bar.
        """
        m1 = np.r_[True, self.down_particles[:-1] != self.down_particles[1:], True]
        counts1 = np.diff(np.flatnonzero(m1))
        unq1 = self.down_particles[m1[:-1]]
        part1 = np.c_[unq1, counts1]
        position_downs = np.max(self.st2.state_positions[self.down_max])
        if self.mode == 'stream' or self.mode == 'realtime':
            current_activations = self.both_activations[0]
        else:
            current_activations = self.both_activations[self.counter]     
        self.downbeat_particles_show[1].set_offsets(np.c_[self.st2.state_positions[part1[:, 0]], np.max(self.st2.state_intervals) - self.st2.state_intervals[part1[:, 0]]])
        self.downbeat_particles_show[2].set_alpha(current_activations[1])
        self.down_particles_swarm.set_xdata(x=position_downs)
        plt.pause(0.0000000001)



def universal_resample_original(particles, weights):  # state_space
    new_particles = []
    J = len(particles)
    weights = weights / sum(weights)
    r = np.random.uniform(0, 1 / J)
    i = 0
    c = weights[0]
    for j in range(J):
        U = r + j * (1 / J)
        while U > c:
            i += 1
            c += weights[i]
        new_particles = np.append(new_particles, particles[i])
    new_particles = new_particles.astype(int)
    return new_particles

# The following resampling method is optimized and is faster than the original BeatNet resampling 
def universal_resample(particles, weights):
    J = len(particles)
    weights = weights / sum(weights)
    cumsum_weights = np.cumsum(weights)
    r = np.random.uniform(0, 1 / J, J)
    U = r + np.arange(J) * (1 / J)
    new_particles = particles[np.searchsorted(cumsum_weights, U)]
    return new_particles


def systematic_resample(particles, weights):
    N = len(weights)
    # make N subdivisions, choose positions
    # with a consistent random offset
    positions = (np.random.randint(0, N) + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N & j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes]


def stratified_resample(particles, weights):
    N = len(weights)
    # make N subdivisions, chose a random position within each one
    positions = (np.random.randint(0, N) + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N & j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes]
