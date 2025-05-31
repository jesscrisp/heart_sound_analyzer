"""
Heart sound segmentation module.

This module provides functions for segmenting heart sounds using various methods,
including peak detection and LR-HSMM models.
"""
import os
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import inspect

# Set up logging
logger = logging.getLogger(__name__)

# pyPCG components and status
HAS_PYPCG = False
# Core components
pcg_signal = None
adv_peak = None
peak_sort_diff = None
segment_peaks = None
# LR-HSMM specific components
load_hsmm = None
heart_state = None
convert_hsmm_states = None
PCGSignalType = Any  # Default type hint for pyPCG signal objects

try:
    from pyPCG import pcg_signal as _imported_pcg_signal_class
    from pyPCG.segment import adv_peak as _imported_adv_peak_func
    from pyPCG.segment import peak_sort_diff as _imported_peak_sort_diff_func
    from pyPCG.segment import segment_peaks as _imported_segment_peaks_func
    # LR-HSMM specific imports
    from pyPCG.segment import load_hsmm as _imported_load_hsmm
    from pyPCG.segment import heart_state as _imported_heart_state
    from pyPCG.segment import convert_hsmm_states as _imported_convert_hsmm_states

    # Assign to module-level variables upon successful import
    pcg_signal = _imported_pcg_signal_class
    adv_peak = _imported_adv_peak_func
    peak_sort_diff = _imported_peak_sort_diff_func
    segment_peaks = _imported_segment_peaks_func
    load_hsmm = _imported_load_hsmm
    heart_state = _imported_heart_state
    convert_hsmm_states = _imported_convert_hsmm_states
    
    HAS_PYPCG = True
    PCGSignalType = pcg_signal  # Update type hint
    logger.info("Successfully imported all required pyPCG components for segmentation.")

except ImportError as e:
    # HAS_PYPCG remains False. All component variables remain None.
    # PCGSignalType remains Any.
    logger.warning(
        f"Could not import all required pyPCG components. Error: {e}. "
        "Both peak detection and LR-HSMM segmentation methods will likely be unavailable or fail."
    )

# Definition for _pyPCG_SignalClass used for isinstance checks.
_pyPCG_SignalClass = pcg_signal if HAS_PYPCG and pcg_signal is not None else Any

# Try to import scipy for fallback methods
try:
    from scipy.signal import find_peaks, convolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not found. Some functionality may be limited.")

# Type aliases
ArrayLike = Union[np.ndarray, List[float]]

# _extract_envelope function removed as it's redundant. Envelope must be provided by processor.py.


def _segment_with_peak_detection(signal_data: np.ndarray, fs: float, envelope_data: Optional[np.ndarray] = None, min_peak_distance: float = 0.15, **kwargs) -> Dict[str, Any]:
    """Segment heart sounds using a peak detection-based approach.

    This method prioritizes a pre-computed envelope. If not provided or invalid,
    it falls back to extracting an envelope using the Scipy/Numpy-based _extract_envelope.
    It then attempts to use pyPCG's peak detection if available and suitable,
    otherwise falls back to a SciPy-based or simple custom peak detection method.
    Finally, it classifies peaks into S1 and S2 and estimates segment boundaries.

    Args:
        signal_data: Input signal as a NumPy array. Required if envelope_data is not provided.
        fs: Sampling frequency of the signal.
        envelope_data: Optional pre-computed envelope (NumPy array).
        min_peak_distance: Minimum distance between peaks in seconds.
        **kwargs: Additional keyword arguments for peak detection and classification:
            pycg_peak_prominence_ratio (float): Prominence ratio for pyPCG adv_peak (default 0.5).
            peak_prominence (float): Relative prominence for SciPy/custom find_peaks (default 0.1 of max env value).
            diastolic_systolic_ratio_threshold (float): Threshold for classifying S1/S2 based on intervals (default 1.2).
            s1_start_drop_ratio (float): Ratio of peak value to find S1 start (default 0.8).
            s1_end_drop_ratio (float): Ratio of peak value to find S1 end (default 0.7).
            s2_start_drop_ratio (float): Ratio of peak value to find S2 start (default 0.8).
            s2_end_drop_ratio (float): Ratio of peak value to find S2 end (default 0.7).

    Returns:
        Dict[str, Any]: Dictionary containing segmentation results:
            's1_starts', 's1_ends', 's2_starts', 's2_ends' (np.ndarray): Segment boundaries.
            'method' (str): Segmentation method used.
            'envelope' (List[float]): The envelope used for segmentation.
    """
    logger.debug(f"_segment_with_peak_detection called. Envelope data shape: {envelope_data.shape if envelope_data is not None else 'None'}, kwargs: {kwargs}")
    env_to_use: Optional[np.ndarray] = None

    try:
        # Validate envelope_data: must be a non-empty NumPy array.
        if envelope_data is None or not isinstance(envelope_data, np.ndarray) or envelope_data.size == 0:
            logger.error("A valid, non-empty NumPy array must be provided for 'envelope_data' in _segment_with_peak_detection.")
            # The calling function (segment_heart_sounds) should handle this exception.
            raise ValueError("A valid, non-empty NumPy array must be provided for 'envelope_data'.")
        
        logger.debug("Using validated pre-computed envelope_data for peak detection.")
        env = envelope_data # Directly use the validated envelope_data.

        # pyPCG parameters from kwargs, with defaults if not provided
        # These names are chosen to be distinct and descriptive for pyPCG usage.
        percent_th_adv_peak = kwargs.get('peak_threshold', 0.5) # Use 'peak_threshold' from config
        hr_win_peak_sort = kwargs.get('hr_win_peak_sort', 1.5) # Keep default or add to config if needed
        dia_length_coeff_peak_sort = kwargs.get('dia_length_coeff_peak_sort', 1.8) # Keep default or add to config if needed
        start_drop_segment_peaks = kwargs.get('start_drop', 0.6) # Use 'start_drop' from config
        end_drop_segment_peaks = kwargs.get('end_drop', 0.6) # Use 'end_drop' from config

        logger.debug("--- pyPCG Peak Detection Debugging ---")
        
        # Log signature of adv_peak
        if HAS_PYPCG and adv_peak: # Check if adv_peak was successfully imported
            try:
                logger.debug(f"Signature of pyPCG.adv_peak: {inspect.signature(adv_peak)}")
            except Exception as e_inspect:
                logger.error(f"Could not get signature of adv_peak: {e_inspect}")
        elif not HAS_PYPCG:
            logger.error("pyPCG not available, cannot inspect adv_peak.")
        else: # HAS_PYPCG is true, but adv_peak is None
            logger.error("adv_peak function is None, cannot inspect.")

        logger.info("Attempting simplified pyPCG peak detection calls...")

        # Ensure pyPCG components are available
        if not HAS_PYPCG:
            logger.error("pyPCG components are not available. This function should not have been called for pyPCG path.")
            raise ImportError("pyPCG components required for _segment_with_peak_detection are not available.")

        # Simplified call to adv_peak
        debug_percent_th = 0.3 # More lenient threshold for debugging
        logger.debug(f"Calling adv_peak(env, fs={fs}, percent_th={debug_percent_th})")
        peak_indices_pycg_debug = np.array([], dtype=int) # Initialize
        try:
            if adv_peak is None:
                raise RuntimeError("adv_peak function is None, cannot proceed with pyPCG peak detection.")
            _, peak_indices_pycg_debug = adv_peak(env, fs=fs, percent_th=debug_percent_th)
            logger.info(f"Simplified adv_peak found {len(peak_indices_pycg_debug)} peaks: {peak_indices_pycg_debug}")
        except Exception as e_adv:
            logger.error(f"Error during simplified adv_peak call: {e_adv}", exc_info=True)
            # Fallback to empty, or could re-raise depending on desired strictness
        
        s1_peaks_pycg_debug = np.array([], dtype=int)
        s2_peaks_pycg_debug = np.array([], dtype=int)
        if len(peak_indices_pycg_debug) >= 2:
            logger.debug(f"Calling peak_sort_diff(peak_indices_pycg_debug, fs={fs}) with {len(peak_indices_pycg_debug)} peaks.")
            try:
                if peak_sort_diff is None:
                    raise RuntimeError("peak_sort_diff function is None, cannot proceed.")
                s1_peaks_pycg_debug, s2_peaks_pycg_debug = peak_sort_diff(peak_indices_pycg_debug, fs=fs)
                logger.info(f"Simplified peak_sort_diff found {len(s1_peaks_pycg_debug)} S1 peaks and {len(s2_peaks_pycg_debug)} S2 peaks.")
                logger.debug(f"S1 peaks (debug): {s1_peaks_pycg_debug}")
                logger.debug(f"S2 peaks (debug): {s2_peaks_pycg_debug}")
            except Exception as e_sort:
                logger.error(f"Error during simplified peak_sort_diff call: {e_sort}", exc_info=True)
        elif HAS_PYPCG and peak_sort_diff is not None : # Only log warning if we could have called it
            logger.warning(f"Skipping peak_sort_diff as fewer than 2 peaks found by adv_peak ({len(peak_indices_pycg_debug)}). Else, peak_sort_diff might be None.")

        # Call segment_peaks (using existing start/end drop parameters from kwargs)
        s1_starts_debug = np.array([], dtype=int)
        s1_ends_debug = np.array([], dtype=int)
        s2_starts_debug = np.array([], dtype=int)
        s2_ends_debug = np.array([], dtype=int)

        if len(s1_peaks_pycg_debug) > 0:
            logger.debug(f"Calling segment_peaks for S1 with {len(s1_peaks_pycg_debug)} peaks. start_drop={start_drop_segment_peaks}, end_drop={end_drop_segment_peaks}")
            try:
                if segment_peaks is None:
                    raise RuntimeError("segment_peaks function is None, cannot proceed for S1.")
                s1_starts_debug, s1_ends_debug = segment_peaks(s1_peaks_pycg_debug, env, start_drop=start_drop_segment_peaks, end_drop=end_drop_segment_peaks)
                logger.info(f"segment_peaks (S1 debug) found {len(s1_starts_debug)} segments.")
            except Exception as e_seg_s1:
                logger.error(f"Error during segment_peaks (S1 debug) call: {e_seg_s1}", exc_info=True)
        
        if len(s2_peaks_pycg_debug) > 0:
            logger.debug(f"Calling segment_peaks for S2 with {len(s2_peaks_pycg_debug)} peaks. start_drop={start_drop_segment_peaks}, end_drop={end_drop_segment_peaks}")
            try:
                if segment_peaks is None:
                    raise RuntimeError("segment_peaks function is None, cannot proceed for S2.")
                s2_starts_debug, s2_ends_debug = segment_peaks(s2_peaks_pycg_debug, env, start_drop=start_drop_segment_peaks, end_drop=end_drop_segment_peaks)
                logger.info(f"segment_peaks (S2 debug) found {len(s2_starts_debug)} segments.")
            except Exception as e_seg_s2:
                logger.error(f"Error during segment_peaks (S2 debug) call: {e_seg_s2}", exc_info=True)

        logger.debug("--- End of pyPCG Peak Detection Debugging ---")
        
        # Use the results from the debug path for the return value
        # This replaces the original pyPCG path's return.
        return {
            's1_starts': s1_starts_debug, 's1_ends': s1_ends_debug,
            's2_starts': s2_starts_debug, 's2_ends': s2_ends_debug,
            'method': 'peak_detection_pyPCG_debug', # Indicate this is from the debug path
            'envelope': env.tolist() if env is not None else []
        }
        
    except Exception as e_outer_peak:
        logger.error(f"Error in peak detection segmentation: {str(e_outer_peak)}", exc_info=True)
        # Ensure env is defined for the return statement in case of early exception
        final_env_list = []
        if 'env' in locals() and env is not None and hasattr(env, 'tolist'):
            final_env_list = env.tolist()
        elif 'env_to_use' in locals() and env_to_use is not None and hasattr(env_to_use, 'tolist'):
             final_env_list = env_to_use.tolist()
        elif envelope_data is not None and hasattr(envelope_data, 'tolist'):
            final_env_list = envelope_data.tolist()

        return {
            's1_starts': np.array([], dtype=int), 's1_ends': np.array([], dtype=int),
            's2_starts': np.array([], dtype=int), 's2_ends': np.array([], dtype=int),
            'method': 'peak_detection_error',
            'envelope': final_env_list
        }

def create_heart_cycle_segments(segments: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Convert segment boundaries to a list of heart cycles.
    
    Args:
        segments: Dictionary containing 's1_starts', 's1_ends', 's2_starts', 's2_ends'
        
    Returns:
        List of dictionaries, each representing a heart cycle with its segments,
        including timing information like durations and intervals.
    """
    logger.debug("create_heart_cycle_segments called.")
    cycles = []
    
    try:
        required_keys = ['s1_starts', 's1_ends', 's2_starts', 's2_ends']
        if not all(key in segments and segments[key] is not None for key in required_keys):
            logger.warning("Missing or None required segment keys in input for create_heart_cycle_segments.")
            return []
            
        s1_starts = np.asarray(segments['s1_starts'])
        s1_ends = np.asarray(segments['s1_ends'])
        s2_starts = np.asarray(segments['s2_starts'])
        s2_ends = np.asarray(segments['s2_ends'])

        logger.debug(f"Input segments: S1 starts ({len(s1_starts)}), S1 ends ({len(s1_ends)}), S2 starts ({len(s2_starts)}), S2 ends ({len(s2_ends)})")

        if not (len(s1_starts) == len(s1_ends) and len(s2_starts) == len(s2_ends)):
            logger.warning("Mismatch in lengths of start/end arrays for S1 or S2 segments.")
            # Attempt to use the minimum length for each pair if they are non-zero
            min_s1_len = min(len(s1_starts), len(s1_ends))
            min_s2_len = min(len(s2_starts), len(s2_ends))
            if min_s1_len == 0 or min_s2_len == 0:
                 logger.warning("Cannot form S1 or S2 segments due to zero length start/end arrays after mismatch.")
                 return []
            s1_starts, s1_ends = s1_starts[:min_s1_len], s1_ends[:min_s1_len]
            s2_starts, s2_ends = s2_starts[:min_s2_len], s2_ends[:min_s2_len]
            logger.debug(f"Adjusted segment lengths: S1 ({min_s1_len}), S2 ({min_s2_len})")

        if len(s1_starts) == 0 or len(s2_starts) == 0:
            logger.info("No S1 or S2 segments available to form cycles.")
            return []

        s1_durations = s1_ends - s1_starts
        s2_durations = s2_ends - s2_starts
        
        _common_len_s1s2 = min(len(s1_starts), len(s2_starts))
        logger.debug(f"Common length for S1-S2 pairing: {_common_len_s1s2}")
        
        s1_start_to_s2_start_interval = np.array([])
        if _common_len_s1s2 > 0:
            s1_start_to_s2_start_interval = s2_starts[:_common_len_s1s2] - s1_starts[:_common_len_s1s2]
        
        for i in range(_common_len_s1s2):
            # Basic validity checks for each segment and their order
            valid_s1 = s1_starts[i] < s1_ends[i]
            valid_s2 = s2_starts[i] < s2_ends[i]
            # S1 should start before S2 for a typical cycle structure S1-Systole-S2
            s1_precedes_s2_start = s1_starts[i] < s2_starts[i]
            # S1 should ideally end before S2 starts (Systolic period)
            s1_ends_before_s2_starts = s1_ends[i] < s2_starts[i]
            # S1 should end before S2 ends
            s1_ends_before_s2_ends = s1_ends[i] < s2_ends[i]

            if not (valid_s1 and valid_s2 and s1_precedes_s2_start and s1_ends_before_s2_starts and s1_ends_before_s2_ends):
                logger.debug(f"Skipping cycle {i}: Invalid segment timing or order. "
                             f"S1: {s1_starts[i]}-{s1_ends[i]} (Valid: {valid_s1}), "
                             f"S2: {s2_starts[i]}-{s2_ends[i]} (Valid: {valid_s2}), "
                             f"S1_starts < S2_starts: {s1_precedes_s2_start}, "
                             f"S1_ends < S2_starts: {s1_ends_before_s2_starts}, "
                             f"S1_ends < S2_ends: {s1_ends_before_s2_ends}")
                continue
                
            cycle = {
                's1_start': int(s1_starts[i]),
                's1_end': int(s1_ends[i]),
                's2_start': int(s2_starts[i]),
                's2_end': int(s2_ends[i]),
                's1_duration': int(s1_durations[i]),
                's2_duration': int(s2_durations[i]),
                's1_start_to_s2_start_interval': int(s1_start_to_s2_start_interval[i]),
                's1_s2_complex_duration': int(s2_ends[i] - s1_starts[i])
            }
            cycles.append(cycle)
            logger.debug(f"Created cycle {i}: {cycle}")
            
        logger.info(f"Created {len(cycles)} heart cycle segments.")
    except Exception as e:
        logger.error(f"Error in create_heart_cycle_segments: {e}", exc_info=True)
    
    return cycles

def segment_heart_sounds(signal_data: np.ndarray, fs: float, envelope_data: Optional[np.ndarray] = None, method: str = 'peak_detection', **kwargs) -> Dict[str, Any]:
    """Segment heart sounds using the specified method.

    Args:
        signal_data: Input PCG signal as a NumPy array.
        fs: Sampling frequency in Hz.
        envelope_data: Optional pre-computed envelope as a NumPy array. If provided,
                       it's passed to compatible segmentation methods (e.g., peak_detection).
        method: Segmentation method ('peak_detection' or 'lr_hsmm').
        **kwargs: Additional parameters for the segmentation method:
            - For peak_detection:
                - min_peak_distance: Minimum distance between peaks in seconds (default: 0.15)
                - peak_prominence: Prominence for SciPy/custom peak detection (default: 0.1 of max envelope)
                - pycg_peak_prominence_ratio: Prominence for pyPCG adv_peak (default: 0.5)
            - For lr_hsmm:
                - model_path: Path to the LR-HSMM model file (required)
            - include_cycles: bool, whether to include 'cycles' in the output (default: True)

    Returns:
        Dictionary containing segmentation results with keys:
        - 's1_starts', 's1_ends': Arrays of S1 sound boundaries (in samples)
        - 's2_starts', 's2_ends': Arrays of S2 sound boundaries (in samples)
        - 'method': The segmentation method used
        - 'envelope': Envelope signal used or generated by the segmentation method
        - 'cycles': List of heart cycle segments

    Raises:
        ValueError: If an unsupported segmentation method is provided, or if signal_data or fs are invalid.
        ImportError: If pyPCG is required for lr_hsmm but not found.
    """
    logger.debug(f"Entering segment_heart_sounds. Method: {method}, Signal data type: {type(signal_data)}, FS: {fs}, Envelope type: {type(envelope_data)}, envelope_data is None: {envelope_data is None}, kwargs: {kwargs}")
    logger.debug(f"kwargs received by segment_heart_sounds: {kwargs}")

    if fs is None or not isinstance(fs, (int, float)) or fs <= 0:
        logger.error(f"Invalid sampling frequency (fs) provided: {fs}. Must be a positive number.")
        raise ValueError("Valid sampling frequency (fs) must be provided as a positive number.")

    if signal_data is None or not isinstance(signal_data, np.ndarray) or signal_data.size == 0:
        logger.error("Invalid signal_data: None, not a NumPy array, or empty.")
        # Return a minimal error structure consistent with other error returns
        return {
            's1_starts': np.array([], dtype=int), 's1_ends': np.array([], dtype=int),
            's2_starts': np.array([], dtype=int), 's2_ends': np.array([], dtype=int),
            'method': f'{method}_error_invalid_input',
            'envelope': envelope_data.tolist() if envelope_data is not None and hasattr(envelope_data, 'tolist') else [],
            'cycles': []
        }

    if np.all(np.isfinite(signal_data)):
        logger.debug(f"  Input signal_data stats - Min: {np.min(signal_data):.4f}, Max: {np.max(signal_data):.4f}, Mean: {np.mean(signal_data):.4f}, Std: {np.std(signal_data):.4f}, Length: {len(signal_data)}")
    else:
        logger.warning("  Input signal_data contains non-finite values. Stats may be unreliable.")

    if method not in ['peak_detection', 'lr_hsmm']:
        logger.error(f"Unsupported segmentation method requested: {method}")
        raise ValueError(f"Unsupported segmentation method: {method}")

    if method == 'lr_hsmm' and not HAS_PYPCG:
        logger.error("Attempted to use 'lr_hsmm' method, but pyPCG is not available.")
        raise ImportError("pyPCG is required for lr_hsmm segmentation")

    try:
        other_kwargs = {k: v for k, v in kwargs.items() if k not in ['fs', 'envelope', 'envelope_data']}
        logger.debug(f"Dispatching to segmentation method '{method}' with fs={fs} and other_kwargs={other_kwargs}")

        results = {}
        if method == 'peak_detection':
            logger.debug("segment_heart_sounds: Routing to _segment_with_peak_detection.")
            logger.debug(f"Calling _segment_with_peak_detection. Provided envelope_data is {'present' if envelope_data is not None else 'absent'}.")
            results = _segment_with_peak_detection(signal_data, fs=fs, envelope_data=envelope_data, **other_kwargs)
        elif method == 'lr_hsmm':
            logger.debug("segment_heart_sounds: Routing to _segment_with_lr_hsmm.")
            model_path_hsmm = other_kwargs.pop('model_path', None) # Get model_path for lr_hsmm
            logger.debug(f"Calling _segment_with_lr_hsmm. model_path: {model_path_hsmm}. Provided envelope_data is {'present' if envelope_data is not None else 'absent'}.")
            results = _segment_with_lr_hsmm(signal_data, fs=fs, envelope_data=envelope_data, model_path=model_path_hsmm, **other_kwargs)
        # No else needed here as method is validated by checks before try block

        # Optionally include heart cycle segments
        if kwargs.get('include_cycles', True) and 's1_starts' in results and results['s1_starts'] is not None and len(results['s1_starts']) > 0:
            logger.debug("Including heart cycle segments in results.")
            results['cycles'] = create_heart_cycle_segments(results)
        else:
            logger.debug("Not including heart cycle segments or segmentation failed to produce S1 starts.")
            results['cycles'] = [] # Ensure 'cycles' key is always present

        logger.info(f"Segmentation with method '{method}' completed. S1 segments: {len(results.get('s1_starts',[]))}, S2 segments: {len(results.get('s2_starts',[]))}")
        return results

    except Exception as e:
        logger.error(f"Error during segmentation with method '{method}': {e}", exc_info=True)
        # Attempt to return the provided envelope if available, otherwise an empty list.
        err_env = []
        if envelope_data is not None and hasattr(envelope_data, 'tolist'):
            err_env = envelope_data.tolist()
        return {
            's1_starts': np.array([], dtype=int),
            's1_ends': np.array([], dtype=int),
            's2_starts': np.array([], dtype=int),
            's2_ends': np.array([], dtype=int),
            'method': f'{method}_error',
            'envelope': err_env,
            'cycles': []
        }




def _segment_with_lr_hsmm(
    signal_data: np.ndarray,
    fs: float,
    envelope_data: Optional[np.ndarray] = None,
    model_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Segment heart sounds using LR-HSMM model from pyPCG."""
    logger.debug(f"_segment_with_lr_hsmm called. fs: {fs}, model_path: {model_path}")
    # ... actual implementation

