"""
Module for extracting envelopes from audio signals.
"""
import logging
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Import shared pyPCG status and type from io.py
from .io import HAS_PYPCG, PCGSignalType

# Initialize specific pyPCG components to None
pcg_signal = None # Will be aliased from PCGSignalType or imported if HAS_PYPCG is True
pyPCG_homomorphic_envelope = None

HAS_PYPCG_ENVELOPE = False # Local status for this module's specific needs

if HAS_PYPCG:
    try:
        # If HAS_PYPCG is true, PCGSignalType from io.py is the actual pcg_signal class.
        # We can assign it to a local 'pcg_signal' variable for use within this module,
        # or ensure all uses of pcg_signal refer to PCGSignalType.
        # For consistency with original local naming and clarity:
        from pyPCG import pcg_signal as pcg_signal_module
        from pyPCG.preprocessing import homomorphic as pyPCG_homomorphic_envelope_module
        
        # Assign to module-level names
        pcg_signal = pcg_signal_module
        pyPCG_homomorphic_envelope = pyPCG_homomorphic_envelope_module
        
        HAS_PYPCG_ENVELOPE = True
        logger.info("EnvelopeExtractor: Successfully imported specific pyPCG envelope components.")
    except ImportError as e:
        logger.warning(f"EnvelopeExtractor: HAS_PYPCG was True, but failed to import specific envelope components: {e}. Envelope extraction features will be limited.")
        # Variables remain None as initialized
else:
    logger.info("EnvelopeExtractor: HAS_PYPCG from io.py is False. pyPCG envelope components will not be imported.")

class EnvelopeExtractor:
    """Handles the extraction of envelopes from audio signals."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the EnvelopeExtractor.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.envelope_params = self.config.get('envelope_extraction_params', {})
        logger.info("EnvelopeExtractor initialized.")

    def extract_envelope(self, signal_object: PCGSignalType) -> Tuple[Optional[np.ndarray], str]:
        """
        Extracts the envelope from the signal data.
        Prioritizes pyPCG's homomorphic envelope if available.

        Args:
            signal_data: The audio signal data as a NumPy array.
            sample_rate: The sample rate of the audio signal.

        Returns:
            A tuple containing:
                - The extracted envelope as a NumPy array, or None if extraction fails.
                - A status message string.
        """
        if not HAS_PYPCG_ENVELOPE or not pcg_signal or not pyPCG_homomorphic_envelope:
            logger.warning("pyPCG envelope components not available. Skipping pyPCG envelope extraction.")
            return np.array([]), "pyPCG envelope components unavailable; returning empty array."

        if not (pcg_signal and isinstance(signal_object, pcg_signal)):
            logger.error(f"Input signal_object must be a pyPCG.pcg_signal, got {type(signal_object)}.")
            return np.array([]), f"Invalid input type for signal_object: {type(signal_object)}."

        if not hasattr(signal_object, 'data') or not hasattr(signal_object, 'fs') or signal_object.data is None or signal_object.fs is None:
            logger.error("Input signal_object is missing 'data' or 'fs' attributes, or they are None.")
            return np.array([]), "Invalid pcg_signal object provided."

        if signal_object.data.ndim == 0 or signal_object.data.size == 0:
            logger.warning("Input signal_object.data is empty or scalar. Cannot extract envelope.")
            return np.array([]), "Input signal_object.data is empty or scalar."

        try:
            logger.info(f"Using provided pcg_signal object (fs={signal_object.fs}) for envelope extraction.")
            pcg_input_obj = signal_object

            # Get parameters from config or use defaults
            filt_cutfreq = self.envelope_params.get('filt_cutfreq', 10)

            filt_order = self.envelope_params.get('filt_order', 2)

            logger.info(f"Applying pyPCG homomorphic envelope (cutfreq={filt_cutfreq}Hz, order={filt_order})...")
            
            # Ensure pyPCG_homomorphic_envelope is callable
            if not callable(pyPCG_homomorphic_envelope):
                logger.error("pyPCG_homomorphic_envelope is not callable.")
                return np.array([]), "pyPCG_homomorphic_envelope not callable."

            extracted_envelope = pyPCG_homomorphic_envelope(pcg_input_obj,
                                                            filt_ord=filt_order,
                                                            filt_cutfreq=filt_cutfreq)
            
            logger.info("pyPCG homomorphic envelope extracted successfully.")
            # pyPCG_homomorphic_envelope returns a pcg_signal object; we need its .data attribute
            return extracted_envelope.data, "pyPCG homomorphic envelope extraction successful."

        except Exception as e:
            logger.error(f"Error during pyPCG homomorphic envelope extraction: {e}", exc_info=True)
            return np.array([]), f"pyPCG envelope extraction failed: {str(e)}"
