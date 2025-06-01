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
PCGSignalType = Any  # Default type hint for pyPCG signal objects

try:
    from pyPCG import pcg_signal as _imported_pcg_signal_class
    from pyPCG.segment import adv_peak as _imported_adv_peak_func
    from pyPCG.segment import peak_sort_diff as _imported_peak_sort_diff_func
    from pyPCG.segment import segment_peaks as _imported_segment_peaks_func

    # Assign to module-level variables upon successful import
    pcg_signal = _imported_pcg_signal_class
    adv_peak = _imported_adv_peak_func
    peak_sort_diff = _imported_peak_sort_diff_func
    segment_peaks = _imported_segment_peaks_func
    
    HAS_PYPCG = True
    PCGSignalType = pcg_signal  # Update type hint
    logger.info("Successfully imported core pyPCG components for segmentation.")

except ImportError as e:
    # HAS_PYPCG remains False. All component variables remain None.
    # PCGSignalType remains Any.
    logger.warning(
        f"Could not import core pyPCG components. Error: {e}. "
        "Peak detection segmentation method will likely be unavailable or fail."
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



# Helper functions for _segment_with_peak_detection pyPCG path
def _pycg_detect_initial_peaks(envelope_pcg_obj: PCGSignalType, **adv_peak_args: Any) -> np.ndarray:
    """Detects initial peaks using pyPCG.segment.adv_peak."""
    if not HAS_PYPCG or adv_peak is None:
        logger.error("adv_peak function is not available. Cannot detect initial peaks.")
        return np.array([], dtype=int)
    try:
        logger.debug(f"Calling adv_peak with args: {adv_peak_args}")
        _, peak_indices = adv_peak(envelope_pcg_obj, **adv_peak_args)
        logger.info(f"adv_peak found {len(peak_indices)} peaks.")
        print(f"🔍 SEGMENTATION (helper): adv_peak found {len(peak_indices)} peaks: {peak_indices[:20]}...")
        return peak_indices
    except Exception as e:
        logger.error(f"Error in _pycg_detect_initial_peaks: {e}", exc_info=True)
        print(f"❌ SEGMENTATION ERROR (helper) in _pycg_detect_initial_peaks: {e}")
        return np.array([], dtype=int)

def _pycg_classify_peaks(peak_indices: np.ndarray, fs: float, **peak_sort_args: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Classifies peaks into S1 and S2 using pyPCG.segment.peak_sort_diff."""
    s1_peaks = np.array([], dtype=int)
    s2_peaks = np.array([], dtype=int)
    if not HAS_PYPCG or peak_sort_diff is None:
        logger.error("peak_sort_diff function is not available. Cannot classify peaks.")
        return s1_peaks, s2_peaks
    
    if len(peak_indices) < 2:
        logger.warning(f"Skipping peak_sort_diff as only {len(peak_indices)} peak(s) found (need at least 2).")
        print(f"🔍 SEGMENTATION_WARNING (helper): Skipping peak_sort_diff, {len(peak_indices)} peaks found.")
        return s1_peaks, s2_peaks
    try:
        logger.debug(f"Calling peak_sort_diff with {len(peak_indices)} peaks and args: {peak_sort_args}")
        s1_peaks, s2_peaks = peak_sort_diff(peak_indices, **peak_sort_args)
        logger.info(f"peak_sort_diff classified {len(s1_peaks)} S1 peaks and {len(s2_peaks)} S2 peaks.")
        print(f"🔍 SEGMENTATION (helper): peak_sort_diff found {len(s1_peaks)} S1 and {len(s2_peaks)} S2 peaks.")
        print(f"🔍 SEGMENTATION_DEBUG (helper): S1 peaks: {s1_peaks[:20]}")
        print(f"🔍 SEGMENTATION_DEBUG (helper): S2 peaks: {s2_peaks[:20]}")
        return s1_peaks, s2_peaks
    except Exception as e:
        logger.error(f"Error in _pycg_classify_peaks: {e}", exc_info=True)
        print(f"❌ SEGMENTATION ERROR (helper) in _pycg_classify_peaks: {e}")
        return np.array([], dtype=int), np.array([], dtype=int)

def _pycg_determine_segment_boundaries(
    classified_peaks: np.ndarray, 
    envelope_pcg_obj: PCGSignalType, 
    sound_type_label: str, 
    **segment_peaks_args: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Determines segment boundaries for classified peaks using pyPCG.segment.segment_peaks."""
    segment_starts = np.array([], dtype=int)
    segment_ends = np.array([], dtype=int)

    if not HAS_PYPCG or segment_peaks is None:
        logger.error(f"segment_peaks function is not available. Cannot determine {sound_type_label} segment boundaries.")
        return segment_starts, segment_ends

    if len(classified_peaks) == 0:
        logger.info(f"No {sound_type_label} peaks provided to _pycg_determine_segment_boundaries.")
        print(f"🔍 SEGMENTATION_INFO (helper): No {sound_type_label} peaks to segment.")
        return segment_starts, segment_ends
    try:
        logger.debug(f"Calling segment_peaks for {sound_type_label} with {len(classified_peaks)} peaks and args: {segment_peaks_args}")
        segment_starts, segment_ends = segment_peaks(classified_peaks, envelope_pcg_obj, **segment_peaks_args)
        logger.info(f"segment_peaks for {sound_type_label} found {len(segment_starts)} segments.")
        print(f"🔍 SEGMENTATION (helper): segment_peaks ({sound_type_label}) found {len(segment_starts)} segments.")
        print(f"🔍 SEGMENTATION_DEBUG (helper): {sound_type_label} starts: {segment_starts[:20]}")
        print(f"🔍 SEGMENTATION_DEBUG (helper): {sound_type_label} ends: {segment_ends[:20]}")
        return segment_starts, segment_ends
    except Exception as e:
        logger.error(f"Error in _pycg_determine_segment_boundaries for {sound_type_label}: {e}", exc_info=True)
        print(f"❌ SEGMENTATION ERROR (helper) in _pycg_determine_segment_boundaries for {sound_type_label}: {e}")
        return np.array([], dtype=int), np.array([], dtype=int)


def _segment_with_peak_detection(envelope_data: np.ndarray, fs: float, min_peak_distance: float = 0.15, **kwargs) -> Dict[str, Any]:
    """Segment heart sounds using a peak detection-based approach from a pre-computed envelope.

    This method uses a pre-computed envelope for peak detection and classification.
    It primarily attempts to use pyPCG's peak detection (adv_peak, peak_sort_diff, segment_peaks).
    If pyPCG components are unavailable or fail, it's intended to fall back to a
    SciPy-based or simple custom peak detection method (currently, this fallback is minimal).

    Args:
        envelope_data: Pre-computed envelope as a NumPy array (required).
        fs: Sampling frequency of the signal.
        min_peak_distance: Minimum distance between peaks in seconds.
                           (Currently primarily used by the SciPy fallback peak detection).
        **kwargs: Additional keyword arguments, typically passed via a 'params':{'peak_detection': {...}}
                  structure from a configuration file. These control pyPCG functions:

            For pyPCG.segment.adv_peak:
                peak_threshold (float): Corresponds to 'percent_th' in adv_peak.
                                        Threshold for peak detection based on envelope amplitude.
                                        If not provided via config, pyPCG's internal default (e.g., 0.05) is used.

            For pyPCG.segment.peak_sort_diff:
                hr_win (float): Corresponds to 'hr_win' in peak_sort_diff.
                                Window size in seconds for heart rate estimation.
                                If not provided via config, pyPCG's internal default (e.g., 1.5) is used.
                dia_length_coeff (float): Corresponds to 'dia_length_coeff' in peak_sort_diff.
                                          Coefficient to estimate diastolic length.
                                          If not provided via config, pyPCG's internal default (e.g., 1.8) is used.

            For pyPCG.segment.segment_peaks:
                start_drop (float): Corresponds to 'start_drop' in segment_peaks.
                                    Ratio of peak value to find segment start.
                                    If not provided via config, pyPCG's internal default (e.g., 0.6) is used.
                end_drop (float): Corresponds to 'end_drop' in segment_peaks.
                                  Ratio of peak value to find segment end.
                                  If not provided via config, pyPCG's internal default (e.g., 0.6) is used.

            For SciPy/custom fallback (less utilized if pyPCG is active):
                peak_prominence (float): Relative prominence for SciPy/custom find_peaks.
                                         (Internal default: 0.1 of max envelope value).
                diastolic_systolic_ratio_threshold (float): Threshold for classifying S1/S2 based on
                                                            intervals in custom logic (if pyPCG fails).
                                                            (Internal default: 1.2).
    Returns:
        Dict[str, Any]: Dictionary containing segmentation results:
            's1_segments' (np.ndarray): S1 segment boundaries as an (N,2) array [start, end]. Empty (0,2) if none.
            's2_segments' (np.ndarray): S2 segment boundaries as an (M,2) array [start, end]. Empty (0,2) if none.
            's1_peaks' (List[int]): List of detected S1 peak locations (indices).
            's2_peaks' (List[int]): List of detected S2 peak locations (indices).
            'method' (str): Segmentation method used (e.g., 'peak_detection_pyPCG_debug', 'peak_detection_error').
            'envelope' (List[float]): The envelope array used for segmentation, converted to a list.
            'cycles' (List[Dict]]): Initially empty; populated by calling function if successful.
                                     (Note: This key is added by the caller, segment_heart_sounds, not this function directly in all paths).
    """
    # Validate envelope_data: must be a non-empty 1D NumPy array.
    if not isinstance(envelope_data, np.ndarray) or envelope_data.ndim != 1 or envelope_data.size == 0:
        logger.error(f"Invalid 'envelope_data' provided to _segment_with_peak_detection. Expected non-empty 1D NumPy array, got {type(envelope_data)} with shape {envelope_data.shape if isinstance(envelope_data, np.ndarray) else 'N/A'}.")
        return {
            's1_starts': np.array([], dtype=int), 's1_ends': np.array([], dtype=int),
            's2_starts': np.array([], dtype=int), 's2_ends': np.array([], dtype=int),
            'method': 'peak_detection_error_invalid_envelope',
            'envelope': [], # Cannot reliably provide envelope if it's invalid
            'cycles': []
        }

    print(f"🔍 SEGMENTATION: _segment_with_peak_detection called with envelope shape: {envelope_data.shape}")
    print(f"🔍 SEGMENTATION: kwargs = {kwargs}")
    logger.debug(f"_segment_with_peak_detection called. Envelope data shape: {envelope_data.shape}, kwargs: {kwargs}")

    try:
        # envelope_data is validated and guaranteed to be a np.ndarray here.
        envelope_pcg_obj = pcg_signal(envelope_data, fs=fs)
        print(f"🔍 SEGMENTATION: Created envelope_pcg_obj from envelope_data. Shape = {envelope_pcg_obj.data.shape}, FS = {envelope_pcg_obj.fs}")

        peak_params_config = kwargs.get('params', {}).get('peak_detection', {})
        logger.debug(f"Parameters from config for peak detection: {peak_params_config}")

        adv_peak_call_args = {}
        if 'peak_threshold' in peak_params_config:
            adv_peak_call_args['percent_th'] = peak_params_config['peak_threshold']
        
        peak_sort_diff_call_args = {}
        if 'hr_win' in peak_params_config:
            peak_sort_diff_call_args['hr_win'] = peak_params_config['hr_win']
        if 'dia_length_coeff' in peak_params_config:
            peak_sort_diff_call_args['dia_length_coeff'] = peak_params_config['dia_length_coeff']

        segment_peaks_call_args = {}
        if 'start_drop' in peak_params_config:
            segment_peaks_call_args['start_drop'] = peak_params_config['start_drop']
        if 'end_drop' in peak_params_config:
            segment_peaks_call_args['end_drop'] = peak_params_config['end_drop']
        
        logger.debug(f"adv_peak_call_args: {adv_peak_call_args}")
        logger.debug(f"peak_sort_diff_call_args: {peak_sort_diff_call_args}")
        logger.debug(f"segment_peaks_call_args: {segment_peaks_call_args}")

        if not HAS_PYPCG:
            logger.error("pyPCG components are not available. Cannot proceed with pyPCG-based segmentation.")
            return {
                's1_segments': np.empty((0, 2), dtype=int),
                's2_segments': np.empty((0, 2), dtype=int),
                's1_peaks': [],
                's2_peaks': [],
                'method': 'peak_detection_error_no_pypcg',
                'envelope': envelope_data.tolist() if hasattr(envelope_data, 'tolist') else [],
                'cycles': []
            }

        # 1. Detect initial peaks
        initial_peak_indices = _pycg_detect_initial_peaks(envelope_pcg_obj, **adv_peak_call_args)

        # 2. Classify peaks
        s1_peaks, s2_peaks = _pycg_classify_peaks(initial_peak_indices, fs, **peak_sort_diff_call_args)
        
        # 3. Determine S1 segment boundaries
        s1_starts, s1_ends = _pycg_determine_segment_boundaries(
            s1_peaks, envelope_pcg_obj, "S1", **segment_peaks_call_args
        )
        
        # 4. Determine S2 segment boundaries
        s2_starts, s2_ends = _pycg_determine_segment_boundaries(
            s2_peaks, envelope_pcg_obj, "S2", **segment_peaks_call_args
        )

        # Construct 2D segment arrays
        s1_segments_arr = np.array([]) 
        if s1_starts.size > 0 and s1_ends.size > 0 and len(s1_starts) == len(s1_ends):
            s1_segments_arr = np.column_stack((s1_starts, s1_ends))
        
        s2_segments_arr = np.array([])
        if s2_starts.size > 0 and s2_ends.size > 0 and len(s2_starts) == len(s2_ends):
            s2_segments_arr = np.column_stack((s2_starts, s2_ends))

        if s1_segments_arr.size == 0:
            s1_segments_arr = np.empty((0, 2), dtype=int)
        
        if s2_segments_arr.size == 0:
            s2_segments_arr = np.empty((0, 2), dtype=int)
        
        print(f"🔍 SEGMENTATION_RETURN: s1_segments_arr shape: {s1_segments_arr.shape}, s2_segments_arr shape: {s2_segments_arr.shape}")

        return {
            's1_segments': s1_segments_arr,
            's2_segments': s2_segments_arr,
            's1_peaks': s1_peaks.tolist() if hasattr(s1_peaks, 'tolist') else list(s1_peaks),
            's2_peaks': s2_peaks.tolist() if hasattr(s2_peaks, 'tolist') else list(s2_peaks),
            'method': 'peak_detection_pyPCG_helpers',
            'envelope': envelope_data.tolist() if hasattr(envelope_data, 'tolist') else []
        }
        
    except Exception as e_outer_peak:
        print(f"❌ SEGMENTATION ERROR in _segment_with_peak_detection (outer try-catch): {e_outer_peak}")
        logger.error(f"Error in peak detection segmentation (outer try-catch): {str(e_outer_peak)}", exc_info=True)
        final_env_list = envelope_data.tolist() if hasattr(envelope_data, 'tolist') else []

        return {
            's1_segments': np.empty((0, 2), dtype=int),
            's2_segments': np.empty((0, 2), dtype=int),
            's1_peaks': [],
            's2_peaks': [],
            'method': 'peak_detection_error_outer',
            'envelope': final_env_list,
            'cycles': [] 
        }

def create_heart_cycle_segments(segmentation_results: Dict[str, Any], fs: Optional[float] = None) -> List[Dict[str, Any]]:
    logger.info("TESTING INFO LOG FROM create_heart_cycle_segments")
    logger.debug(f"create_heart_cycle_segments: Initial segmentation_results keys: {list(segmentation_results.keys())}")
    """Convert segment boundaries from segmentation_results to a list of heart cycles.
    
    Args:
        segmentation_results: Dictionary containing 's1_segments' (Nx2 array) 
                              and 's2_segments' (Mx2 array).
        
    Returns:
        List of dictionaries, each representing a heart cycle with its segments,
        including timing information like durations and intervals.
    """
    logger.debug(f"create_heart_cycle_segments called with keys: {list(segmentation_results.keys())}")
    cycles = []
    
    try:
        required_keys = ['s1_segments', 's2_segments']
        if not all(key in segmentation_results and isinstance(segmentation_results[key], np.ndarray) for key in required_keys):
            logger.warning(f"Missing or invalid required segment keys ({required_keys}) in input for create_heart_cycle_segments. Received keys: {list(segmentation_results.keys())}")
            return []

        s1_segments_arr = segmentation_results['s1_segments']
        s2_segments_arr = segmentation_results['s2_segments']

        logger.debug(f"Received s1_segments shape: {s1_segments_arr.shape}, s2_segments shape: {s2_segments_arr.shape}")

        if s1_segments_arr.ndim != 2 or s1_segments_arr.shape[1] != 2 or s1_segments_arr.shape[0] == 0:
            logger.warning(f"Invalid s1_segments array shape: {s1_segments_arr.shape}. Expected (N, 2) with N > 0.")
            return []
        if s2_segments_arr.ndim != 2 or s2_segments_arr.shape[1] != 2 or s2_segments_arr.shape[0] == 0:
            logger.warning(f"Invalid s2_segments array shape: {s2_segments_arr.shape}. Expected (M, 2) with M > 0.")
            return []
            
        s1_starts = s1_segments_arr[:, 0]
        s1_ends = s1_segments_arr[:, 1]
        s2_starts = s2_segments_arr[:, 0]
        s2_ends = s2_segments_arr[:, 1]

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

            # Detailed logging for validity checks
            logger.debug(f"Cycle candidate {i}: S1 ({s1_starts[i]}-{s1_ends[i]}), S2 ({s2_starts[i]}-{s2_ends[i]})")
            logger.debug(f"  Check valid_s1 (s1_starts < s1_ends): {s1_starts[i]} < {s1_ends[i]} -> {valid_s1}")
            logger.debug(f"  Check valid_s2 (s2_starts < s2_ends): {s2_starts[i]} < {s2_ends[i]} -> {valid_s2}")
            logger.debug(f"  Check s1_precedes_s2_start (s1_starts < s2_starts): {s1_starts[i]} < {s2_starts[i]} -> {s1_precedes_s2_start}")
            logger.debug(f"  Check s1_ends_before_s2_starts (s1_ends < s2_starts): {s1_ends[i]} < {s2_starts[i]} -> {s1_ends_before_s2_starts}")
            logger.debug(f"  Check s1_ends_before_s2_ends (s1_ends < s2_ends): {s1_ends[i]} < {s2_ends[i]} -> {s1_ends_before_s2_ends}")

            if not (valid_s1 and valid_s2 and s1_precedes_s2_start and s1_ends_before_s2_starts and s1_ends_before_s2_ends):
                logger.debug(f"---> Skipping cycle candidate {i} due to failed checks.")
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
    """Segment heart sounds from a PCG signal using the specified method.

    Currently, the primary supported method is 'peak_detection', which relies on
    a pre-computed envelope for identifying S1 and S2 sounds.

    Args:
        signal_data: Input PCG signal as a NumPy array. (Note: While accepted, this parameter
                     is not directly used by the 'peak_detection' method if 'envelope_data'
                     is provided. It's maintained for API consistency and potential future methods).
        fs: Sampling frequency in Hz.
        envelope_data: Pre-computed envelope as a NumPy array.
                       This is REQUIRED if `method` is 'peak_detection'.
        method: Segmentation method to use. Currently, 'peak_detection' is the primary
                supported method.
        **kwargs: Additional parameters:
            include_cycles (bool): If True (default), attempts to create detailed heart
                                   cycle segments from S1/S2 detections and includes
                                   them under the 'cycles' key in the results.
            
            Other keyword arguments are passed down to the chosen segmentation method.
            For the 'peak_detection' method, these are passed to the
            `_segment_with_peak_detection` function. Please refer to the
            docstring of `_segment_with_peak_detection` for details on parameters
            like `peak_threshold`, `hr_win`, `dia_length_coeff`, `start_drop`, `end_drop`, etc.

    Returns:
        Dict[str, Any]: A dictionary containing segmentation results:
            's1_segments' (np.ndarray): S1 segment boundaries as an (N,2) array [start, end].
                                        Shape is (0,2) if no segments are found.
            's2_segments' (np.ndarray): S2 segment boundaries as an (M,2) array [start, end].
                                        Shape is (0,2) if no segments are found.
            's1_peaks' (List[int]): List of detected S1 peak locations (indices). Empty if none.
            's2_peaks' (List[int]): List of detected S2 peak locations (indices). Empty if none.
            'method' (str): The segmentation method identifier (e.g., 'peak_detection_pyPCG_debug',
                            'peak_detection_error').
            'envelope' (List[float]): The envelope array used for segmentation, converted to a list.
                                      Empty if an error occurred before envelope processing.
            'cycles' (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                             represents a heart cycle (S1, systole, S2, diastole components).
                                             This key is present and populated if `include_cycles`
                                             is True and cycle creation is successful. Otherwise,
                                             it will be an empty list.

    Raises:
        ValueError: If `signal_data` or `fs` are invalid, if an unsupported segmentation
                    `method` is specified, or if `envelope_data` is not provided when
                    required by the chosen method (e.g., for 'peak_detection').
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

    # Validate envelope_data before dispatching
    if envelope_data is None or not isinstance(envelope_data, np.ndarray) or envelope_data.size == 0:
        logger.error(f"Invalid 'envelope_data' provided to segment_heart_sounds. Expected non-empty NumPy array, got {type(envelope_data)}.")
        return {
            's1_starts': np.array([], dtype=int), 's1_ends': np.array([], dtype=int),
            's2_starts': np.array([], dtype=int), 's2_ends': np.array([], dtype=int),
            'method': f'{method}_error_invalid_envelope',
            'envelope': [],
            'cycles': []
        }
    if np.all(np.isfinite(envelope_data)):
        logger.debug(f"  Input envelope_data stats - Min: {np.min(envelope_data):.4f}, Max: {np.max(envelope_data):.4f}, Mean: {np.mean(envelope_data):.4f}, Std: {np.std(envelope_data):.4f}, Length: {len(envelope_data)}")
    else:
        logger.warning("  Input envelope_data contains non-finite values. Stats may be unreliable.")

    if method not in ['peak_detection']:
        logger.error(f"Unsupported segmentation method requested: {method}")
        raise ValueError(f"Unsupported segmentation method: {method}")


    try:
        other_kwargs = {k: v for k, v in kwargs.items() if k not in ['fs', 'envelope', 'envelope_data', 'signal_data']}
        logger.debug(f"Dispatching to segmentation method '{method}' with fs={fs} and other_kwargs={other_kwargs}")

        results = {}
        if method == 'peak_detection':
            logger.debug("segment_heart_sounds: Routing to _segment_with_peak_detection.")
            logger.debug(f"Calling _segment_with_peak_detection. Provided envelope_data is present (shape: {envelope_data.shape}).")
            results = _segment_with_peak_detection(envelope_data, fs=fs, **other_kwargs)

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

