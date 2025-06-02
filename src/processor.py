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
from .preprocessing_pipeline import SignalPreprocessor
from .envelope_extractor import EnvelopeExtractor

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
        self.signal_preprocessor = SignalPreprocessor(self.config)
        self.envelope_extractor = EnvelopeExtractor(self.config)
    
    
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
                
                input_for_preprocessor = None # Will be pcg_signal object or numpy array

                if is_pcg_object:
                    # loaded_signal_or_data is a pcg_signal object
                    pcg_audio_obj: PCGSignalType = loaded_signal_or_data # type: ignore
                    if pcg_audio_obj is not None:
                        raw_signal_data = np.copy(pcg_audio_obj.data) # Keep original raw data
                        sample_rate = pcg_audio_obj.fs
                        input_for_preprocessor = pcg_audio_obj # Pass the object to preprocessor
                        logger.info(f"Audio loaded as pcg_signal. fs={sample_rate}, shape={raw_signal_data.shape if raw_signal_data is not None else 'N/A'}")
                        status_message += " Audio loaded as pyPCG object."
                    else:
                        logger.error("is_pcg_object is True but loaded_signal_or_data is None. This should not happen.")
                        status_message += " Critical error: loaded_signal_or_data is None despite being pcg_object type."
                        # Flags will remain false, leading to segmentation skip

                elif isinstance(loaded_signal_or_data, tuple) and len(loaded_signal_or_data) == 2:
                    _raw, _sr = loaded_signal_or_data
                    if isinstance(_raw, np.ndarray):
                        raw_signal_data = _raw # Keep original raw data
                        input_for_preprocessor = raw_signal_data # Pass numpy array to preprocessor
                    if isinstance(_sr, (int, float)):
                        sample_rate = int(_sr)
                    
                    if raw_signal_data is not None and sample_rate is not None:
                        logger.info(f"Audio loaded as numpy array. fs={sample_rate}, shape={raw_signal_data.shape}")
                        status_message += " Audio loaded as numpy array."
                    else:
                        logger.error("Failed to extract numpy array or sample rate from loader's tuple output.")
                        status_message += " Failed to extract audio data/sample rate from loader output."
                        # Flags will remain false
                else:
                    logger.error(f"Loaded audio type unhandled or data missing: {type(loaded_signal_or_data)}. Cannot proceed.")
                    status_message += " Unhandled audio load type or missing data."
                    # Flags will remain false

                # Proceed with preprocessing and envelope extraction if input is valid
                if input_for_preprocessor is not None and sample_rate is not None:
                    logger.info("Attempting signal preprocessing...")
                    # preprocess now returns (Optional[Dict[str, Union[np.ndarray, PCGSignalType]]], str)
                    intermediate_signals_dict, preprocess_status = self.signal_preprocessor.preprocess(input_for_preprocessor, sample_rate)
                    status_message += f" {preprocess_status}"

                    # Initialize intermediate signal storage in results
                    results_intermediate_signals = {
                        'raw_for_preprocessing': None,
                        'after_highpass': None,
                        'after_lowpass': None,
                        'after_denoise': None
                    }

                    if intermediate_signals_dict is not None:
                        results_intermediate_signals['raw_for_preprocessing'] = intermediate_signals_dict.get('raw_for_preprocessing')
                        results_intermediate_signals['after_highpass'] = intermediate_signals_dict.get('after_highpass')
                        results_intermediate_signals['after_lowpass'] = intermediate_signals_dict.get('after_lowpass')
                        results_intermediate_signals['after_denoise'] = intermediate_signals_dict.get('after_denoise')
                        
                        processed_signal_data = intermediate_signals_dict.get('final_processed_data') # This is a NumPy array
                        processed_pcg_object_for_envelope = intermediate_signals_dict.get('final_processed_signal_object') # This is a PCGSignalType object

                        if processed_signal_data is not None and processed_pcg_object_for_envelope is not None and pcg_signal and isinstance(processed_pcg_object_for_envelope, pcg_signal):
                            sample_rate = processed_pcg_object_for_envelope.fs # Update sample rate from processed object
                            logger.info(f"Signal preprocessing step completed. Processed data shape: {processed_signal_data.shape}")
                            processing_pipeline_successful = True

                            logger.info("Attempting envelope extraction...")
                            # extract_envelope now takes PCGSignalType object
                            extracted_envelope_data, envelope_status = self.envelope_extractor.extract_envelope(processed_pcg_object_for_envelope)
                            status_message += f" {envelope_status}"

                            if extracted_envelope_data is not None and extracted_envelope_data.size > 0:
                                logger.info(f"Envelope extraction step completed. Envelope shape: {extracted_envelope_data.shape}")
                                envelope_extraction_successful = True
                            else:
                                logger.warning("Envelope extraction returned None or empty data.")
                                envelope_extraction_successful = False
                        else:
                            logger.warning("Signal preprocessing did not return valid final processed data or signal object. Skipping further processing.")
                            processing_pipeline_successful = False
                            envelope_extraction_successful = False
                            processed_signal_data = None # Ensure data is None
                    else: # intermediate_signals_dict was None (preprocessing failed)
                        logger.warning("Signal preprocessing returned None. Skipping further processing in this path.")
                        processing_pipeline_successful = False
                        envelope_extraction_successful = False
                        processed_signal_data = None # Ensure data is None if preprocessing failed
                        extracted_envelope_data = np.array([]) # Ensure envelope is empty
                        # status_message already updated by preprocess_status
            else: # Corresponds to: if input_for_preprocessor is None and sample_rate is None (from line 142):
                logger.error("Cannot proceed with preprocessing: essential audio data (input_for_preprocessor) or sample rate is missing after loading stage.")
                # Flags are already initialized to False at the start of the method.
                # Ensure they remain so if this path is taken due to missing inputs for preprocessing.
                processing_pipeline_successful = False
                envelope_extraction_successful = False
                # processed_signal_data and extracted_envelope_data retain their initial values (None, empty array)
                # if this path is taken.

            # 3. Segmentation (only if pyPCG processing and envelope extraction were successful)
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
                    segments = segmentation_results # Ensure the 'segments' variable for final_results gets the actual data

                    s1_segments_array = segments.get("s1_segments", np.array([])) # Use 'segments' dict directly
                    s2_segments_array = segments.get("s2_segments", np.array([])) # Use 'segments' dict directly

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
            'raw_audio_data': raw_signal_data,  # Original raw audio data
            # Intermediate preprocessing signals
            'raw_for_preprocessing': results_intermediate_signals.get('raw_for_preprocessing') if 'results_intermediate_signals' in locals() else None,
            'after_highpass': results_intermediate_signals.get('after_highpass') if 'results_intermediate_signals' in locals() else None,
            'after_lowpass': results_intermediate_signals.get('after_lowpass') if 'results_intermediate_signals' in locals() else None,
            'after_denoise': results_intermediate_signals.get('after_denoise') if 'results_intermediate_signals' in locals() else None,
            # Final processed signal (NumPy array)
            'processed_audio_data': processed_signal_data,
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
