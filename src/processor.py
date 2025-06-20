"""
Heart Sound Processor module for coordinating the heart sound analysis pipeline.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path

# Import internal modules
from .segmentation import segment_heart_sounds, create_heart_cycle_segments
from .io import AudioLoader, HAS_PYPCG, PCGSignalType # Import shared status and type

# Configure logging
logger = logging.getLogger(__name__)

# Initialize pyPCG components to None. They will be imported if HAS_PYPCG is True.
pcg_signal = None
pyPCG_filter = None
wt_denoise_sth = None
process_pipeline = None
pyPCG_normalize = None
pyPCG_homomorphic_envelope = None
adv_peak = None # Example, if used from pyPCG.segment

PROCESSOR_HAS_PYPCG_COMPONENTS = False # Local status for processor's needs

if HAS_PYPCG:
    try:
        from pyPCG import pcg_signal
        from pyPCG.preprocessing import filter as pyPCG_filter
        from pyPCG.preprocessing import wt_denoise_sth
        from pyPCG.preprocessing import process_pipeline
        from pyPCG import normalize as pyPCG_normalize
        from pyPCG.preprocessing import homomorphic as pyPCG_homomorphic_envelope
        # from pyPCG.segment import adv_peak # If explicitly needed

        # Module-level variables (pyPCG_filter, wt_denoise_sth, etc.) are now directly assigned by the imports
        
        PROCESSOR_HAS_PYPCG_COMPONENTS = True
        logger.info("Processor successfully imported required pyPCG submodules.")
    except ImportError as e_proc_pcg:
        logger.error(f"Processor HAS_PYPCG was True, but failed to import specific pyPCG submodules: {e_proc_pcg}. pyPCG processing will be disabled for processor.")
        PROCESSOR_HAS_PYPCG_COMPONENTS = False
else:
    logger.info("Processor: HAS_PYPCG from io.py is False. pyPCG features will be unavailable.")

class HeartSoundProcessor:
    """Coordinates the heart sound analysis pipeline using pyPCG."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.audio_loader = AudioLoader(config)
        self.segmentation_config = config.get('segmentation', {})
    
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a heart sound audio file end-to-end.

    Prioritizes pyPCG for preprocessing. If any pyPCG processing step fails
    or components are unavailable, segmentation is not attempted.

    Args:
        file_path: Path to the audio file

    Returns:
        Dictionary containing processing results, including status indicators.
    """
        logger.info(f"Starting processing for file: {file_path}")

        # Initialize results and status
        raw_signal_data: Optional[np.ndarray] = None
        processed_signal_data: Optional[np.ndarray] = None
        sample_rate: Optional[int] = None
        extracted_envelope_data: Optional[np.ndarray] = None
        segments: Dict[str, Any] = {'s1_starts': np.array([]), 's1_ends': np.array([]), 
                                    's2_starts': np.array([]), 's2_ends': np.array([]), 
                                    'method': 'none', 'envelope': []}
        heart_cycles: List[Dict[str, Any]] = []
        status_message = "Processing initiated."
        processing_pipeline_successful = False
        envelope_extraction_successful = False
        segmentation_successful = False

        try:
            # 1. Load Audio
            # AudioLoader now returns pcg_signal or (numpy_array, fs) or raises error
            loaded_signal_or_data = self.audio_loader.load_audio(file_path)
            status_message = "Audio loading attempted."

            is_pcg_object = pcg_signal is not None and isinstance(loaded_signal_or_data, pcg_signal)

            if PROCESSOR_HAS_PYPCG_COMPONENTS and is_pcg_object:
                pcg_audio_obj: pcg_signal = loaded_signal_or_data # type: ignore
                status_message = "pyPCG components available and audio loaded as pcg_signal."
                logger.info(status_message)
                
                try:
                    raw_signal_data = np.copy(pcg_audio_obj.data)
                    sample_rate = pcg_audio_obj.fs
                    logger.info(f"pyPCG path: Loaded signal with fs={sample_rate}, shape={raw_signal_data.shape}, dtype={raw_signal_data.dtype}")

                    # Define pyPCG pipeline steps
                    low_cut = self.config.get('audio', {}).get('low_cut_hz', 25)
                    high_cut = self.config.get('audio', {}).get('high_cut_hz', 150)
                    filter_order = self.config.get('audio', {}).get('filter_order', 4)

                    hp_filter_step = {'step': pyPCG_filter, 'params': {'filt_ord': filter_order, 'filt_cutfreq': low_cut, 'filt_type': 'HP'}}
                    lp_filter_step = {'step': pyPCG_filter, 'params': {'filt_ord': filter_order, 'filt_cutfreq': high_cut, 'filt_type': 'LP'}}
                    denoise_step = wt_denoise_sth
                    normalize_step = pyPCG_normalize
                    
                    core_pipeline = process_pipeline(hp_filter_step, lp_filter_step, denoise_step, normalize_step)
                    
                    logger.info("Applying pyPCG core processing pipeline (filter, denoise, normalize)...")
                    processed_pcg_obj = core_pipeline.run(pcg_audio_obj)
                    logger.info("pyPCG core processing pipeline applied.")

                    logger.info("Applying pyPCG envelope extraction...")
                    envelope_pcg_obj = pyPCG_homomorphic_envelope(processed_pcg_obj)
                    extracted_envelope_data = envelope_pcg_obj.data
                    logger.info("pyPCG envelope extraction applied.")
                    if extracted_envelope_data is not None and extracted_envelope_data.size > 0:
                        logger.info(f"Envelope stats: min={np.min(extracted_envelope_data)}, max={np.max(extracted_envelope_data)}, shape={extracted_envelope_data.shape}, dtype={extracted_envelope_data.dtype})")
                        envelope_extraction_successful = True # Set flag here
                        if np.min(extracted_envelope_data) < 0:
                            logger.error("CRITICAL ERROR: Envelope contains negative values!")
                    elif extracted_envelope_data is None or extracted_envelope_data.size == 0:
                        logger.warning("Envelope data is None or empty after extraction.")
                        envelope_extraction_successful = False # Explicitly false


                    processed_signal_data = processed_pcg_obj.data
                    if envelope_extraction_successful:
                        processing_pipeline_successful = True
                        status_message += "pyPCG core processing and envelope extraction applied."
                        logger.info("pyPCG core processing and envelope extraction applied.")
                    else:
                        processing_pipeline_successful = False # Envelope extraction failed
                        status_message += "pyPCG core processing applied, but envelope extraction failed."
                        logger.warning("pyPCG core processing applied, but envelope extraction failed.")

                except Exception as e_proc:
                    logger.error(f"Error during pyPCG processing pipeline: {e_proc}", exc_info=True)
                    status_message += " pyPCG processing or envelope extraction failed, or pyPCG not available."
                    # raw_signal_data and sample_rate might still be set if error was mid-pipeline
                    processed_signal_data = None 
                    extracted_envelope_data = None
                    processing_pipeline_successful = False
            
            else: # PROCESSOR_HAS_PYPCG_COMPONENTS is False or audio not loaded as pcg_signal
                logger.warning("Processor's pyPCG components not available (PROCESSOR_HAS_PYPCG_COMPONENTS is False) or audio not loaded as pcg_signal. Cannot proceed with pyPCG processing.")
                status_message = "Processor's pyPCG components not available or audio not loaded as pcg_signal. Cannot proceed with pyPCG processing."
                processing_pipeline_successful = False
                # Ensure raw_signal_data and sample_rate are populated if audio was loaded as numpy array by AudioLoader
                if isinstance(loaded_signal_or_data, tuple) and len(loaded_signal_or_data) == 2:
                     _raw, _sr = loaded_signal_or_data
                     if isinstance(_raw, np.ndarray) and raw_signal_data is None: raw_signal_data = _raw
                     if isinstance(_sr, (int, float)) and sample_rate is None: sample_rate = int(_sr)

            # 2. Segmentation (only if pyPCG processing was successful)
            if processing_pipeline_successful and processed_signal_data is not None and sample_rate is not None:
                logger.info("Proceeding to segmentation.")
                try:
                    segmentation_kwargs = self.segmentation_config # Use the specific segmentation config loaded in __init__
                    segmentation_results = segment_heart_sounds(
                        processed_signal_data,
                        fs=sample_rate,
                        envelope_data=extracted_envelope_data,
                        **segmentation_kwargs
                    )
                    # Comprehensive debug print for segmentation_results
                    debug_info = {}
                    for key, value in segmentation_results.items():
                        if isinstance(value, np.ndarray):
                            debug_info[key] = f"ndarray, shape={value.shape}, dtype={value.dtype}"
                        elif isinstance(value, (list, tuple)):
                            debug_info[key] = f"{type(value).__name__}, len={len(value)}"
                        else:
                            debug_info[key] = f"type={type(value).__name__}"
                    print(f"🔍 PROCESSOR DEBUG: segmentation_results content: {debug_info}")

                    s1_segments_array = segmentation_results.get("s1_segments", np.array([]))
                    s2_segments_array = segmentation_results.get("s2_segments", np.array([]))

                    # Robust extraction of S1 and S2 starts and ends
                    s1_starts, s1_ends = (s1_segments_array[:, 0], s1_segments_array[:, 1]) if s1_segments_array.ndim == 2 and s1_segments_array.shape[0] > 0 else (np.array([]), np.array([]))
                    s2_starts, s2_ends = (s2_segments_array[:, 0], s2_segments_array[:, 1]) if s2_segments_array.ndim == 2 and s2_segments_array.shape[0] > 0 else (np.array([]), np.array([]))

                    segments = {
                        's1_starts': s1_starts,
                        's1_ends': s1_ends,
                        's2_starts': s2_starts,
                        's2_ends': s2_ends,
                        'method': segmentation_results.get('method', 'peak_detection_pycg'), # Get method from results
                        'envelope': segmentation_results.get('envelope', extracted_envelope_data) # Prefer envelope from results
                    }
                    
                    logger.info(f"Segmentation completed. Found {len(s1_starts)} S1 and {len(s2_starts)} S2 sounds.")
                    heart_cycles = create_heart_cycle_segments(segmentation_results)
                    segmentation_successful = True
                    status_message += " Segmentation successful."
                except Exception as e_seg:
                    logger.error(f"Error during segmentation: {e_seg}", exc_info=True)
                    status_message += f" Segmentation failed: {str(e_seg)}"
                    segmentation_successful = False
                    # segments and heart_cycles remain in their default empty state
            elif processing_pipeline_successful: # but processed_signal_data or sample_rate is None (should not happen if successful)
                status_message += " Segmentation skipped: processed data or sample rate missing despite reported processing success."
                logger.error(status_message)
                segmentation_successful = False
            else: # processing_pipeline_successful is False
                status_message += " Segmentation skipped due to pyPCG processing failure or unavailability."
                logger.warning("Segmentation skipped as pyPCG processing was not successful or prerequisites not met.")
                segmentation_successful = False

        except Exception as e_main:
            logger.error(f"Unhandled error in process_file for {file_path}: {e_main}", exc_info=True)
            status_message = f"Critical error in process_file: {str(e_main)}"
            # Ensure results are in a default failure state
            processing_pipeline_successful = False
            segmentation_successful = False
            # raw_signal_data, sample_rate might have been set before the error

        # Prepare final results dictionary
        final_results = {
            'file_path': file_path,
            'raw_audio_data': raw_signal_data,  # Original raw audio data, previously 'signal' or 'raw_signal_data'
            'processed_audio_data': processed_signal_data,  # Audio data after pyPCG processing, previously 'processed' or 'processed_signal_for_segmentation'
            'sample_rate': sample_rate,
            'envelope': extracted_envelope_data,
            'segments': segments,
            'heart_cycles': heart_cycles,
            'status_message': status_message,
            'processing_pipeline_successful': processing_pipeline_successful,
            'envelope_extraction_successful': envelope_extraction_successful,
            'segmentation_successful': segmentation_successful
        }

        logger.info(f"Finished processing for file: {file_path}. Status: {status_message}")
        return final_results

    def _extract_features(self, signal, segments) -> Dict[str, Any]:
        """Extract features from the segmented signal.
        
        Args:
            signal: Preprocessed PCG signal
            segments: Dictionary containing segmentation results
            
        Returns:
            Dictionary of extracted features
        """
        # Placeholder for feature extraction logic
        return {}
        
    def batch_process(self, file_paths) -> List[Dict[str, Any]]:
        """Process multiple audio files.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processing results
        """
        return [self.process_file(fp) for fp in file_paths]
