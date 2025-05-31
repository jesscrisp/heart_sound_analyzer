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
    print("[IMPORT DEBUG] Attempting: import pyPCG")
    import pyPCG
    print("[IMPORT DEBUG] Success: import pyPCG")

    print("[IMPORT DEBUG] Attempting: from pyPCG import normalize, plot")
    from pyPCG import normalize, plot
    print("[IMPORT DEBUG] Success: from pyPCG import normalize, plot")

    print("[IMPORT DEBUG] Attempting: from pyPCG.preprocessing import envelope as pypcg_envelope_func")
    from pyPCG.preprocessing import envelope as pypcg_envelope_func
    print("[IMPORT DEBUG] Success: from pyPCG.preprocessing import envelope")


    logger.info("All pyPCG modules for segmentation.py appear to be imported.")
    HAS_PYPCG = True
    print("[IMPORT DEBUG] HAS_PYPCG set to True")
    logger.info("pyPCG successfully confirmed in segmentation.py. Using pyPCG for segmentation.")
except ImportError as e_import:
    print(f"[IMPORT DEBUG] ImportError occurred: {e_import}")
    HAS_PYPCG = False
    print("[IMPORT DEBUG] HAS_PYPCG set to False due to ImportError")
    logger.warning(f"pyPCG module not found or import error: {e_import}. Some functionality may be limited.")

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
    """Extract envelope from PCG signal, prioritizing pyPCG's method.
    
    Args:
        signal: Input PCG signal (pyPCG PCGSignal object or numpy array)
        fs: Sampling frequency
        
    Returns:
        Extracted and normalized envelope as numpy array
    """
    print("[DEBUG PRINT] _extract_envelope called.") # New print
    try:
        print(f"[DEBUG PRINT] _extract_envelope: HAS_PYPCG = {HAS_PYPCG}") # New print
        if HAS_PYPCG:
            try:
                pcg_signal_obj = None
                # Ensure signal is a pyPCG PCGSignal object
                if isinstance(signal, pyPCG.pcg_signal):
                    pcg_signal_obj = signal
                    # Update fs if the object has its own fs, to ensure consistency
                    if hasattr(pcg_signal_obj, 'fs') and pcg_signal_obj.fs is not None:
                        fs = pcg_signal_obj.fs 
                elif hasattr(signal, 'data') and hasattr(signal, 'fs'): # Check for generic PCG-like structure
                    # Attempt to create pyPCG.PCGSignal object, assuming signal.data is array-like
                    pcg_signal_obj = pyPCG.pcg_signal(data=np.asarray(signal.data), fs=signal.fs)
                    fs = signal.fs # Use fs from the object
                else: # Assuming it's a numpy array or list
                    signal_data_arr = np.asarray(signal)
                    print("[DEBUG PRINT] _extract_envelope: Creating pyPCG.PCGSignal object from np.array.") # New print
                    pcg_signal_obj = pyPCG.pcg_signal(data=signal_data_arr, fs=fs)
                    print("[DEBUG PRINT] _extract_envelope: pyPCG.PCGSignal object created.") # New print

                # Use pyPCG's envelope function
                print(f"[DEBUG PRINT] _extract_envelope: Calling pypcg_envelope_func with object type: {type(pcg_signal_obj)}") # New print
                envelope_pcg_signal = pypcg_envelope_func(pcg_signal_obj)
                print("[DEBUG PRINT] _extract_envelope: pypcg_envelope_func returned.") # New print
                env = np.asarray(envelope_pcg_signal.data)

                # Normalize to [0, 1]
                if env.size > 0 and np.max(env) > 0:
                    env = env / np.max(env)
                elif env.size > 0: # All zeros or negative (though envelope should be non-negative)
                    env = np.zeros_like(env)
                # If env is empty, it will be returned as such
                
                print("[DEBUG PRINT] Attempting to log pyPCG envelope success...") # Temporary print for debugging
                logger.debug("Successfully extracted envelope using pyPCG.")
                return env
            except Exception as e:
                print(f"[DEBUG PRINT] _extract_envelope: EXCEPTION in pyPCG path: {e}") # New print
                logger.warning(f"pyPCG envelope extraction failed: {e}. Falling back to custom method.")
        
        # Fallback: Custom envelope extraction (absolute value + moving average)
        logger.debug("Using custom envelope extraction method.")
        
        current_signal_data = None
        if HAS_PYPCG and isinstance(signal, pyPCG.pcg_signal):
            current_signal_data = np.asarray(signal.data)
        elif hasattr(signal, 'data') and hasattr(signal, 'fs'): # Check for generic PCG-like structure
            current_signal_data = np.asarray(signal.data)
        else: # Assume it's a numpy array or list
            current_signal_data = np.asarray(signal)

        if current_signal_data.size == 0:
            logger.warning("Input signal for custom envelope extraction is empty.")
            return np.array([])
            
        env = np.abs(current_signal_data)
        
        if HAS_SCIPY:
            window_size = int(0.02 * fs)  # 20ms window for smoothing
            if window_size > 1 and env.size >= window_size:
                window = np.ones(window_size) / window_size
                env = convolve(env, window, mode='same')
            elif window_size > 1 and env.size < window_size:
                logger.debug(f"Signal length ({env.size}) too short for moving average window ({window_size}). Skipping.")

        if env.size > 0 and np.max(env) > 0:
            env = env / np.max(env)
        elif env.size > 0:
            env = np.zeros_like(env)
            
        return env
        
    except Exception as e:
        logger.error(f"Critical error in _extract_envelope: {e}", exc_info=True)
        # Ultimate fallback: simple normalized absolute value if all else fails
        ultimate_fallback_data = None
        if HAS_PYPCG and isinstance(signal, pyPCG.pcg_signal):
            ultimate_fallback_data = np.asarray(signal.data)
        elif hasattr(signal, 'data') and hasattr(signal, 'fs'):
            ultimate_fallback_data = np.asarray(signal.data)
        else:
            ultimate_fallback_data = np.asarray(signal)
        
        if ultimate_fallback_data.size == 0:
            return np.array([])

        abs_env = np.abs(ultimate_fallback_data)
        if abs_env.size > 0 and np.max(abs_env) > 0:
            return abs_env / np.max(abs_env)
        elif abs_env.size > 0:
            return np.zeros_like(abs_env)
        return np.array([]) # Should not be reached if ultimate_fallback_data.size > 0


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
        
        # Classify S1 and S2 peaks based on inter-peak intervals
        # This aims to emulate the principle of pyPCG's peak_sort_diff:
        # S1-S2 intervals are typically shorter than S2-S1 intervals.
        s1_peaks = []
        s2_peaks = []

        if len(peaks) > 0:
            if len(peaks) == 1:
                s1_peaks.append(peaks[0]) # Assume a single detected peak is S1
            else:
                # Determine if the sequence likely starts with S1 or S2
                # This is based on comparing the first two inter-peak intervals.
                starts_with_s1 = True # Default assumption: sequence starts with S1
                if len(peaks) >= 3: # Need at least 3 peaks to have two intervals
                    interval_A = peaks[1] - peaks[0] # Duration between peak 0 and peak 1
                    interval_B = peaks[2] - peaks[1] # Duration between peak 1 and peak 2
                    
                    # Heuristic: If interval_A is significantly longer than interval_B,
                    # it suggests interval_A is diastolic (S2-S1) and interval_B is systolic (S1-S2).
                    # This implies the sequence starts with S2 (peaks[0] is S2).
                    # The ratio threshold can be tuned via kwargs if needed.
                    ratio_threshold = kwargs.get('diastolic_systolic_ratio_threshold', 1.2)
                    if interval_A > interval_B * ratio_threshold:
                        starts_with_s1 = False
                
                # Assign peaks by alternating, starting with the determined label for the first peak
                current_label_is_s1 = starts_with_s1
                for peak_location in peaks:
                    if current_label_is_s1:
                        s1_peaks.append(peak_location)
                    else:
                        s2_peaks.append(peak_location)
                    current_label_is_s1 = not current_label_is_s1 # Alternate for the next peak
        
        # Ensure outputs are sorted numpy arrays (though `peaks` was already sorted)
        s1_peaks = np.sort(np.array(s1_peaks, dtype=int))
        s2_peaks = np.sort(np.array(s2_peaks, dtype=int))
        # End of new S1/S2 classification logic
        
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
        # Determine the number of S1-S2 pairs we can form
        _common_len_s1s2 = min(len(s1_starts), len(s2_starts))
        
        # Calculate s1s2_intervals only for these common pairs.
        # Note: This specific interval (S1_start to S2_start) might not be the
        # standard definition of a systolic interval (which is typically S1_end to S2_start),
        # but we are preserving the original variable's intent for now to fix the crash.
        if _common_len_s1s2 > 0:
            s1s2_intervals = s2_starts[:_common_len_s1s2] - s1_starts[:_common_len_s1s2]
        else:
            s1s2_intervals = np.array([]) # No pairs to calculate intervals for
        
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
            other_kwargs = {k: v for k, v in kwargs.items() if k != 'fs'}
            result = _segment_with_peak_detection(signal, fs=fs, **other_kwargs)
        elif method == 'lr_hsmm':
            other_kwargs = {k: v for k, v in kwargs.items() if k != 'fs'}
            result = _segment_with_lr_hsmm(signal, fs=fs, **other_kwargs)
            
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
        
    try:
        if not model_path or not os.path.isfile(model_path):
            raise ValueError(f"Valid model_path is required for LR-HSMM segmentation. Got: {model_path}")
        
        # Import required pyPCG modules
        from pyPCG.segment import load_hsmm, heart_state
        from pyPCG.pcg_signal import PCG
        from pyPCG.preprocessing import envelope
        
        # Create PCG signal object if needed
        if hasattr(signal, 'data') and hasattr(signal, 'fs'):
            pcg_signal = signal
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
