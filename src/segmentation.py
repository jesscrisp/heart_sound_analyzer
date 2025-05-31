"""
Heart sound segmentation module.

This module provides functions for segmenting heart sounds using various methods,
including peak detection and LR-HSMM models.
"""
import os
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Try to import pyPCG
try:
    import pyPCG
    from pyPCG import normalize, plot
    HAS_PYPCG = True
except ImportError:
    HAS_PYPCG = False
    logger.warning("pyPCG module not found. Some functionality may be limited.")

# Try to import scipy for fallback methods
try:
    from scipy.signal import find_peaks, convolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not found. Some functionality may be limited.")

# Type aliases
PCGSignal = Any  # Type alias for PCG signal objects
ArrayLike = Union[np.ndarray, List[float]]

def _extract_envelope(signal: Union[PCGSignal, ArrayLike], fs: float = 1000.0) -> np.ndarray:
    """Extract envelope from PCG signal.
    
    Args:
        signal: Input PCG signal (PCGSignal object or numpy array)
        fs: Sampling frequency for envelope extraction
        
    Returns:
        Extracted and normalized envelope as numpy array
    """
    try:
        # Try pyPCG's envelope extraction if available
        if HAS_PYPCG and hasattr(signal, 'get_envelope'):
            try:
                env = signal.get_envelope()
                if env is not None:
                    return env
            except Exception as e:
                logger.debug(f"pyPCG envelope extraction failed, falling back to custom method: {e}")
        
        # Extract signal data if it's a PCGSignal object
        signal_data = signal.signal if hasattr(signal, 'signal') else np.asarray(signal)
        
        # Simple envelope extraction using absolute value + moving average
        env = np.abs(signal_data)
        
        # Apply moving average for smoothing using scipy if available
        if HAS_SCIPY:
            window_size = int(0.02 * fs)  # 20ms window
            if window_size > 1:
                window = np.ones(window_size) / window_size
                env = convolve(env, window, mode='same')
        
        # Normalize to [0, 1]
        if np.max(env) > 0:
            env = env / np.max(env)
            
        return env
        
    except Exception as e:
        logger.error(f"Error extracting envelope: {e}")
        # Fallback to absolute value
        signal_data = signal.signal if hasattr(signal, 'signal') else signal
        return np.abs(np.asarray(signal_data))


def _segment_with_peak_detection(
    signal: Union[PCGSignal, np.ndarray],
    fs: float = 1000.0,
    min_peak_distance: float = 0.1,
    **kwargs
) -> Dict[str, Any]:
    """Segment heart sounds using peak detection on the envelope.
    
    Args:
        signal: Input PCG signal (PCG object or numpy array)
        fs: Sampling frequency in Hz
        min_peak_distance: Minimum distance between peaks in seconds
        **kwargs: Additional arguments for peak detection:
            - peak_prominence: Minimum prominence of peaks (default: 0.1)
            
    Returns:
        Dictionary containing segmentation results with keys:
        - 's1_starts', 's1_ends': Arrays of S1 sound boundaries (in samples)
        - 's2_starts', 's2_ends': Arrays of S2 sound boundaries (in samples)
        - 'method': The segmentation method used
        - 'envelope': Extracted envelope signal (for visualization)
    """
    try:
        # Extract and normalize envelope
        env = _extract_envelope(signal, fs=fs)
        
        # Convert minimum peak distance to samples
        min_peak_samples = int(min_peak_distance * fs)
        
        # Find peaks using scipy's find_peaks or a simple fallback
        if HAS_PYPCG:
            try:
                from pyPCG.segment import adv_peak, peak_sort_diff, segment_peaks
                
                # Use pyPCG's advanced peak detection
                peak_prominence = kwargs.get('peak_prominence', 0.5)
                _, peak_indices = adv_peak(env, percent_th=peak_prominence)
                
                if len(peak_indices) >= 2:
                    # Sort peaks into S1 and S2 using pyPCG's method
                    s1_peaks, s2_peaks = peak_sort_diff(peak_indices)
                    
                    # Get segment boundaries using pyPCG's method
                    s1_starts, s1_ends = segment_peaks(s1_peaks, env, 
                                                     start_drop=0.6, end_drop=0.6)
                    s2_starts, s2_ends = segment_peaks(s2_peaks, env,
                                                     start_drop=0.6, end_drop=0.6)
                    
                    return {
                        's1_starts': s1_starts.astype(int),
                        's1_ends': s1_ends.astype(int),
                        's2_starts': s2_starts.astype(int),
                        's2_ends': s2_ends.astype(int),
                        'method': 'peak_detection',
                        'envelope': env.tolist()
                    }
                    
            except Exception as e:
                logger.debug(f"pyPCG peak detection failed, falling back to scipy: {e}")
        
        # Get peak prominence parameter
        peak_prominence = kwargs.get('peak_prominence', 0.1) * np.max(env)  # Scale once
        
        # Fall back to scipy's find_peaks if available
        if HAS_SCIPY:
            try:
                peaks, _ = find_peaks(
                    env,
                    distance=min_peak_samples,
                    prominence=peak_prominence
                )
                peaks = np.array(peaks) if len(peaks) > 0 else np.array([])
            except Exception as e:
                logger.warning(f"Error in find_peaks: {e}")
                peaks = np.array([])
        else:
            # Simple peak detection fallback
            logger.warning("scipy not available, using simple peak detection")
            peaks = []
            for i in range(1, len(env)-1):
                if env[i] > env[i-1] and env[i] > env[i+1] and env[i] > peak_prominence:
                    peaks.append(i)
            peaks = np.array(peaks) if len(peaks) > 0 else np.array([])
        
        if len(peaks) < 2:
            logger.warning("Insufficient peaks detected for segmentation")
            return {
                's1_starts': np.array([], dtype=int),
                's1_ends': np.array([], dtype=int),
                's2_starts': np.array([], dtype=int),
                's2_ends': np.array([], dtype=int),
                'method': 'peak_detection',
                'envelope': env.tolist()
            }
        
        # Simple peak classification based on alternating pattern
        if len(peaks) > 1:
            # Sort peaks by amplitude
            peak_amplitudes = env[peaks]
            sorted_idx = np.argsort(peak_amplitudes)[::-1]  # Sort descending
            
            # Classify as S1 and S2 based on position and amplitude
            s1_peaks = []
            s2_peaks = []
            for i, idx in enumerate(peaks[sorted_idx]):
                if i % 2 == 0:
                    s1_peaks.append(idx)
                else:
                    s2_peaks.append(idx)
                    
            s1_peaks = np.sort(s1_peaks)
            s2_peaks = np.sort(s2_peaks)
        else:
            s1_peaks = np.array(peaks) if len(peaks) > 0 else np.array([])
            s2_peaks = np.array([])
        
        # Estimate segment boundaries based on peak locations
        def estimate_segments(peak_indices, signal_length):
            """Estimate segment starts and ends based on peak locations."""
            if len(peak_indices) == 0:
                return np.array([]), np.array([])
                
            # Simple boundary estimation
            mid_points = (peak_indices[1:] + peak_indices[:-1]) // 2
            starts = np.concatenate(([0], mid_points))
            ends = np.concatenate((mid_points, [signal_length - 1]))
            
            # Apply start/end drop to refine boundaries
            start_drop = 0.8  # Start of S1/S2 as fraction of peak height
            end_drop = 0.7    # End of S1/S2 as fraction of peak height
            
            for i, peak in enumerate(peak_indices):
                if peak >= len(env):
                    continue
                    
                peak_val = env[peak]
                
                # Find start of peak (first point below start_drop * peak_val)
                start_idx = peak
                while start_idx > 0 and env[start_idx] > start_drop * peak_val:
                    start_idx -= 1
                starts[i] = max(0, start_idx)
                
                # Find end of peak (first point below end_drop * peak_val)
                end_idx = peak
                while end_idx < len(env) - 1 and env[end_idx] > end_drop * peak_val:
                    end_idx += 1
                ends[i] = min(len(env) - 1, end_idx)
                
            return starts, ends
        
        # Get segments for S1 and S2
        s1_starts, s1_ends = estimate_segments(s1_peaks, len(env))
        s2_starts, s2_ends = estimate_segments(s2_peaks, len(env))
        
        return {
            's1_starts': s1_starts.astype(int),
            's1_ends': s1_ends.astype(int),
            's2_starts': s2_starts.astype(int),
            's2_ends': s2_ends.astype(int),
            'method': 'peak_detection',
            'envelope': env.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error in peak detection segmentation: {str(e)}", exc_info=True)
        # Return consistent format with all required keys
        return {
            's1_starts': np.array([], dtype=int),
            's1_ends': np.array([], dtype=int),
            's2_starts': np.array([], dtype=int),
            's2_ends': np.array([], dtype=int),
            'method': 'peak_detection',
            'envelope': []
        }

def create_heart_cycle_segments(segments: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Convert segment boundaries to a list of heart cycles.
    
    Args:
        segments: Dictionary containing 's1_starts', 's1_ends', 's2_starts', 's2_ends'
        
    Returns:
        List of dictionaries, each representing a heart cycle with its segments,
        including timing information like durations and intervals.
    """
    cycles = []
    
    try:
        # Ensure all required keys are present
        required_keys = ['s1_starts', 's1_ends', 's2_starts', 's2_ends']
        if not all(key in segments for key in required_keys):
            logger.warning("Missing required segment keys in create_heart_cycle_segments")
            return []
            
        # Get all S1 and S2 timings
        s1_starts = np.asarray(segments['s1_starts'])
        s1_ends = np.asarray(segments['s1_ends'])
        s2_starts = np.asarray(segments['s2_starts'])
        s2_ends = np.asarray(segments['s2_ends'])
        
        # Calculate durations and intervals
        s1_durations = s1_ends - s1_starts
        s2_durations = s2_ends - s2_starts
        s1s2_intervals = s2_starts - s1_starts
        
        # Create cycles by pairing S1 and S2 segments
        for i in range(min(len(s1_starts), len(s2_starts))):
            # Ensure valid segment ordering
            if s1_starts[i] >= s2_starts[i] or s1_ends[i] >= s2_ends[i]:
                continue
                
            cycle = {
                # Segment boundaries
                's1_start': int(s1_starts[i]),
                's1_end': int(s1_ends[i]),
                's2_start': int(s2_starts[i]),
                's2_end': int(s2_ends[i]),
                
                # Durations and intervals
                's1_duration': int(s1_durations[i]),
                's2_duration': int(s2_durations[i]),
                's1s2_interval': int(s1s2_intervals[i]),
                'cycle_duration': int(s2_ends[i] - s1_starts[i]) if i < len(s2_ends) - 1 else None
            }
            cycles.append(cycle)
            
    except Exception as e:
        logger.error(f"Error in create_heart_cycle_segments: {e}", exc_info=True)
    
    return cycles

def segment_heart_sounds(signal: Union[PCGSignal, np.ndarray], method: str = 'peak_detection', **kwargs) -> Dict[str, Any]:
    """Segment heart sounds using the specified method.
    
    Args:
        signal: Input PCG signal (PCGSignal object or numpy array)
        method: Segmentation method ('peak_detection' or 'lr_hsmm')
        **kwargs: Additional parameters for the segmentation method
            - fs: Sampling frequency in Hz (required if signal is numpy array)
            - For peak_detection:
                - min_peak_distance: Minimum distance between peaks in seconds (default: 0.1)
                - min_peak_height: Minimum peak height (0-1, default: 0.5)
            - For lr_hsmm:
                - model_path: Path to the LR-HSMM model file (required)
            
    Returns:
        Dictionary containing segmentation results with keys:
        - 's1_starts', 's1_ends': Arrays of S1 sound boundaries (in samples)
        - 's2_starts', 's2_ends': Arrays of S2 sound boundaries (in samples)
        - 'method': The segmentation method used
        - 'envelope': Extracted envelope signal (for visualization)
        - 'cycles': List of heart cycle segments (if include_cycles=True)
        
    Raises:
        ValueError: If an unsupported segmentation method is provided
    """
    if method not in ['peak_detection', 'lr_hsmm']:
        raise ValueError(f"Unsupported segmentation method: {method}")
        
    if method == 'lr_hsmm' and not HAS_PYPCG:
        raise ImportError("pyPCG is required for lr_hsmm segmentation")
        
    try:
        # Extract fs from signal object or kwargs
        fs = signal.fs if hasattr(signal, 'fs') else kwargs.get('fs')
        
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided either in the signal object or as a parameter")
        
        # Call the appropriate segmentation method
        if method == 'peak_detection':
            result = _segment_with_peak_detection(signal, fs=fs, **kwargs)
        elif method == 'lr_hsmm':
            result = _segment_with_lr_hsmm(signal, fs=fs, **kwargs)
            
        # Add cycles if requested
        if kwargs.get('include_cycles', True):
            result['cycles'] = create_heart_cycle_segments(result)
            
        return result
            
    except Exception as e:
        logger.error(f"Error in {method} segmentation: {e}")
        # Return empty results on error
        empty_result = {
            's1_starts': np.array([], dtype=int),
            's1_ends': np.array([], dtype=int),
            's2_starts': np.array([], dtype=int),
            's2_ends': np.array([], dtype=int),
            'method': method,
            'envelope': np.array([])
        }
        if kwargs.get('include_cycles', True):
            empty_result['cycles'] = []
        return empty_result

def _segment_with_lr_hsmm(
    signal: Union[PCGSignal, np.ndarray],
    fs: float = 1000.0,
    model_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Segment heart sounds using LR-HSMM model from pyPCG.
    
    Args:
        signal: Input PCG signal (PCG object or numpy array)
        fs: Sampling frequency in Hz (required if signal is numpy array)
        model_path: Path to the LR-HSMM model file (required)
        **kwargs: Additional parameters for the segmentation method:
            - expected_hr_range: Tuple of (min_hr, max_hr) in BPM (default: (40, 180))
            - bandpass_frq: Tuple of (low_cut, high_cut) in Hz for bandpass filter (default: (25, 400))
            - recalc_timing: Whether to recalculate timing parameters (default: False)
            
    Returns:
        Dictionary containing segmentation results with keys:
        - 's1_starts', 's1_ends': Arrays of S1 sound boundaries (in samples)
        - 's2_starts', 's2_ends': Arrays of S2 sound boundaries (in samples)
        - 'method': The segmentation method used ('lr_hsmm')
        - 'envelope': Extracted envelope signal (for visualization)
    """
    if not HAS_PYPCG:
        raise ImportError("pyPCG is required for LR-HSMM segmentation")
        
    if not model_path or not os.path.isfile(model_path):
        raise ValueError(f"Valid model_path is required for LR-HSMM segmentation. Got: {model_path}")
    
    try:
        # Import required pyPCG modules
        from pyPCG.segment import load_hsmm, heart_state
        from pyPCG.pcg_signal import PCG
        from pyPCG.preprocessing import envelope
        
        # Create PCG signal object if needed
        if hasattr(signal, 'data') and hasattr(signal, 'fs'):
            pcg_signal = signal
            fs = pcg_signal.fs  # Use the signal's fs if it's a PCG object
        else:
            # Create PCG signal object from numpy array
            pcg_signal = PCG(data=np.asarray(signal, dtype=np.float64), fs=fs)
        
        # Extract envelope for visualization
        env_signal = envelope(pcg_signal)
        env = env_signal.data if hasattr(env_signal, 'data') else env_signal
        
        # Set model parameters from kwargs or use defaults
        expected_hr_range = kwargs.get('expected_hr_range', (40, 180))
        bandpass_frq = kwargs.get('bandpass_frq', (25, 400))
        recalc_timing = kwargs.get('recalc_timing', False)
        
        # Load the HSMM model
        hsmm_model = load_hsmm(model_path)
        
        # Run HSMM segmentation
        states = hsmm_model.segment_single(
            pcg_signal.data,
            recalc_timing=recalc_timing
        )
        
        # Convert states to segment boundaries
        from pyPCG.segment import convert_hsmm_states
        
        # Get S1 segments
        s1_starts, s1_ends = convert_hsmm_states(states, heart_state.S1)
        
        # Get S2 segments
        s2_starts, s2_ends = convert_hsmm_states(states, heart_state.S2)
        
        # Convert to numpy arrays and ensure they're 1D
        s1_starts = np.asarray(s1_starts).flatten().astype(int)
        s1_ends = np.asarray(s1_ends).flatten().astype(int)
        s2_starts = np.asarray(s2_starts).flatten().astype(int)
        s2_ends = np.asarray(s2_ends).flatten().astype(int)
        
        # Create result dictionary
        result = {
            's1_starts': s1_starts,
            's1_ends': s1_ends,
            's2_starts': s2_starts,
            's2_ends': s2_ends,
            'method': 'lr_hsmm',
            'envelope': env.tolist() if hasattr(env, 'tolist') else env
        }
        
        # Add states if available and requested
        if kwargs.get('include_states', False):
            result['states'] = states
            
        return result
        
    except Exception as e:
        logger.error(f"Error in LR-HSMM segmentation: {e}", exc_info=True)
        logger.warning("Falling back to peak detection")
        
        # Fall back to peak detection if HSMM fails
        return _segment_with_peak_detection(signal, fs=fs, **kwargs)
