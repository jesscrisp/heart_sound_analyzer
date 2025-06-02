"""
Module for signal preprocessing tasks.
"""
import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# Import shared pyPCG status and type from io.py
from .io import HAS_PYPCG, PCGSignalType

# Initialize specific pyPCG components to None
pcg_signal = None # Will be aliased to PCGSignalType if HAS_PYPCG is True and import succeeds
pyPCG_filter = None
wt_denoise_sth = None
process_pipeline = None
pyPCG_normalize = None

HAS_PYPCG_PREPROCESSING = False # Local status for this module's specific needs

if HAS_PYPCG:
    try:
        # pcg_signal is already available via PCGSignalType from io.py if HAS_PYPCG is true
        # We still need to assign it to a local 'pcg_signal' for consistent use in this module
        # if we are not using PCGSignalType directly everywhere.
        # However, the original code imported it directly, so let's see if we can simplify.
        # If PCGSignalType from io is already the pcg_signal class, we might not need a separate import here.
        # For clarity and consistency with how it was used, let's try to import it again but know it might be redundant if PCGSignalType is correctly aliased.
        from pyPCG import pcg_signal as pcg_signal_module # Explicit import for clarity
        from pyPCG.preprocessing import filter as pyPCG_filter_module
        from pyPCG.preprocessing import wt_denoise_sth as wt_denoise_sth_module
        from pyPCG.preprocessing import process_pipeline as process_pipeline_module
        from pyPCG import normalize as pyPCG_normalize_module

        # Assign to module-level names
        pcg_signal = pcg_signal_module
        pyPCG_filter = pyPCG_filter_module
        wt_denoise_sth = wt_denoise_sth_module
        process_pipeline = process_pipeline_module
        pyPCG_normalize = pyPCG_normalize_module
        
        HAS_PYPCG_PREPROCESSING = True
        logger.info("SignalPreprocessor: Successfully imported specific pyPCG preprocessing components.")
    except ImportError as e:
        logger.warning(f"SignalPreprocessor: HAS_PYPCG was True, but failed to import specific preprocessing components: {e}. Preprocessing features will be limited.")
        # Variables remain None as initialized
else:
    logger.info("SignalPreprocessor: HAS_PYPCG from io.py is False. pyPCG preprocessing components will not be imported.")

class SignalPreprocessor:
    """Handles the preprocessing of audio signals, prioritizing pyPCG methods."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the SignalPreprocessor.

        Args:
            config: Configuration dictionary, potentially containing pyPCG parameters.
        """
        self.config = config
        self.pycpg_processing_params = self.config.get('pycpg_processing_params', {})
        logger.info("SignalPreprocessor initialized.")

    def preprocess(self, audio_data: Union[np.ndarray, PCGSignalType], sample_rate: int) -> Tuple[Optional[PCGSignalType], str]:
        """
        Applies a preprocessing pipeline (filter, denoise, normalize) to the audio data.
        Prioritizes pyPCG if available and the input is a pcg_signal object or can be converted.

        Args:
            audio_data: The input audio data, either as a NumPy array or a pyPCG.pcg_signal object.
            sample_rate: The sample rate of the audio data.

        Returns:
            A tuple containing:
                - The processed audio data as a PCGSignalType object, or None if processing fails.
                - A status message string.
        """
        if not HAS_PYPCG_PREPROCESSING or not all([pcg_signal, pyPCG_filter, wt_denoise_sth, process_pipeline, pyPCG_normalize]):
            logger.warning("pyPCG preprocessing components not fully available. Skipping pyPCG preprocessing.")
            # Fallback logic when pyPCG components are not fully available
            if pcg_signal and isinstance(audio_data, pcg_signal): # Check if it's already a pcg_signal
                return audio_data, "pyPCG preprocessing components unavailable; returning original pcg_signal object."
            elif isinstance(audio_data, np.ndarray):
                # If it's a NumPy array and we can't process it with pyPCG (due to missing components),
                # we cannot convert it to PCGSignalType here without pcg_signal constructor.
                # The expectation is to return a PCGSignalType or None.
                logger.warning("Input is NumPy array, but pyPCG components are unavailable for conversion/processing.")
                return None, "pyPCG components unavailable to process or convert NumPy array."
            else: # Neither a known pcg_signal nor a NumPy array, or pcg_signal itself is None
                return None, "pyPCG components unavailable and input type unsuitable or unknown."

        try:
            pcg_input_signal: PCGSignalType
            if isinstance(audio_data, np.ndarray):
                logger.info("Input is NumPy array, converting to pcg_signal object for pyPCG processing.")
                pcg_input_signal = pcg_signal(data=audio_data, fs=sample_rate) # pyPCG normalize step is separate
            elif isinstance(audio_data, pcg_signal):
                pcg_input_signal = audio_data
                if pcg_input_signal.fs != sample_rate:
                    logger.warning(f"pcg_signal object fs ({pcg_input_signal.fs}) differs from provided sample_rate ({sample_rate}). Using pcg_signal's fs.")
                    # Or, could raise an error or re-sample. For now, trust the object's fs.
            else:
                logger.error(f"Unsupported audio_data type for pyPCG preprocessing: {type(audio_data)}")
                return None, f"Unsupported audio_data type: {type(audio_data)}"

            # Define pyPCG pipeline steps using config from self.config (passed during __init__)
            # self.pycpg_processing_params is available, but original used self.config.get('audio', {})
            audio_config = self.config.get('audio', {})
            low_cut = audio_config.get('low_cut_hz', 25)
            high_cut = audio_config.get('high_cut_hz', 400) # Adjusted based on common PCG range, was 150
            filter_order = audio_config.get('filter_order', 4)

            # Ensure pyPCG components are callable
            if not callable(pyPCG_filter) or not callable(wt_denoise_sth) or not callable(pyPCG_normalize) or not callable(process_pipeline):
                logger.error("One or more pyPCG processing components are not callable.")
                return None, "pyPCG components not callable."

            hp_filter_step = {'step': pyPCG_filter, 'params': {'filt_ord': filter_order, 'filt_cutfreq': low_cut, 'filt_type': 'HP'}}
            lp_filter_step = {'step': pyPCG_filter, 'params': {'filt_ord': filter_order, 'filt_cutfreq': high_cut, 'filt_type': 'LP'}}
            denoise_step = wt_denoise_sth # wt_denoise_sth is a function, not a dict with 'step'
            normalize_step = pyPCG_normalize # pyPCG_normalize is a function
            
            # process_pipeline expects functions or dicts with 'step' and 'params'
            # For functions like wt_denoise_sth and pyPCG_normalize, they are passed directly. 
            core_pipeline = process_pipeline(hp_filter_step, lp_filter_step, denoise_step, normalize_step)
            
            logger.info(f"Applying pyPCG core processing pipeline (HPF@{low_cut}Hz, LPF@{high_cut}Hz, Denoise, Normalize)...")
            processed_pcg_obj = core_pipeline.run(pcg_input_signal)
            logger.info("pyPCG core processing pipeline applied successfully.")
            
            return processed_pcg_obj, "pyPCG preprocessing successful."

        except Exception as e:
            logger.error(f"Error during pyPCG preprocessing pipeline: {e}", exc_info=True)
            return None, f"pyPCG preprocessing failed: {str(e)}"
