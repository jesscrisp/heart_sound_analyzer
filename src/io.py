"""
Input/Output operations for heart sound analysis.

This module handles audio file loading (from WAV and potentially other formats
supported by soundfile or pyPCG) and saving (to WAV format).
It aims to provide raw audio data, either as NumPy arrays or pyPCG signal objects.

Note: This module does NOT perform advanced signal processing tasks such as
resampling, filtering (beyond what pyPCG's read_signal_file might apply by
default), or normalization. These tasks are typically handled by other modules
in the pipeline (e.g., processor.py).
"""

# Standard library
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

# Third-party
import numpy as np

try:
    import soundfile as sf
except ImportError as e_sf:
    logging.getLogger(__name__).error("Soundfile library not found. It is required for basic audio loading.")
    raise ImportError("Soundfile library is required.") from e_sf

try:
    from scipy.io import wavfile
    # No specific log here on import, will log if used or if pyPCG is missing for this purpose.
except ImportError:
    # This warning is fine if scipy is truly optional for some paths
    logging.getLogger(__name__).warning("Scipy.io.wavfile not found. Fallback WAV loading/saving might be affected if pyPCG is also unavailable.")

# Conditional pyPCG imports & type definitions
HAS_PYPCG = False
PCGSignalType = Any  # Default type from typing.Any

# Initialize module-level names for pyPCG components that will be imported
pcg_signal = None
read_signal_file = None
pyPCG_write_signal_file = None # Specifically for the save function, imported separately

try:
    # Try to import core pyPCG components for loading and signal representation
    from pyPCG import pcg_signal as pcg_signal_module
    from pyPCG.io import read_signal_file as rsf_module
    
    # Assign to module-level names upon successful import
    pcg_signal = pcg_signal_module
    read_signal_file = rsf_module
    
    logging.getLogger(__name__).info("Successfully imported core pyPCG modules (pcg_signal, read_signal_file).")
    HAS_PYPCG = True  # Mark that core pyPCG features are available
    PCGSignalType = pcg_signal  # Update type alias

    # Now, separately try to import the write_signal_file function
    try:
        from pyPCG.io import write_signal_file as wsf_module
        pyPCG_write_signal_file = wsf_module # Assign if successful
        logging.getLogger(__name__).info("Successfully imported pyPCG.io.write_signal_file.")
    except ImportError as e_wsf:
        logging.getLogger(__name__).warning(f"pyPCG.io.write_signal_file not found or import failed: {e_wsf}. Saving with pyPCG will use fallback if this function is None.")
        # pyPCG_write_signal_file remains None, HAS_PYPCG is still True for loading

except ImportError as e_pcg_core:
    # This block catches failure for pcg_signal or read_signal_file
    logging.getLogger(__name__).error(f"CRITICAL: Core pyPCG modules (pcg_signal or read_signal_file) import failed in io.py: {e_pcg_core}. HAS_PYPCG will be False.")
    # HAS_PYPCG remains False, other pyPCG components remain None

# Module logger
logger = logging.getLogger(__name__)

class AudioLoader:
    """Handles loading, preprocessing, and saving of heart sound audio files."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config.get('audio', {})
    
    def load_audio(self, file_path: Union[str, Path]) -> Union[PCGSignalType, Tuple[np.ndarray, int]]:
        """Load an audio file and return raw signal data and sample rate.
        If pyPCG is available, returns a pcg_signal object.
        Otherwise, returns a tuple (numpy_array, sample_rate).
        No internal resampling or filtering is performed here.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            Union[PCGSignalType, Tuple[np.ndarray, int]]: If pyPCG is available,
                a pcg_signal object. Otherwise, a tuple containing the NumPy array
                of audio data and the integer sample rate.

        Raises:
            FileNotFoundError: If the audio file specified by `file_path` does not exist.
            RuntimeError: If loading the audio file fails due to an I/O error,
                sound file format issue, or other unexpected error during processing.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Audio file not found: {file_path}")
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        try:
            if HAS_PYPCG:
                logger.debug(f"Loading audio with pyPCG: {file_path}")
                # pyPCG.read_signal_file returns (data, fs) tuple
                audio_data, sample_rate = read_signal_file(str(file_path), format='wav')
                logger.info(f"Loaded {file_path} with pyPCG. Original fs: {sample_rate}, Samples: {len(audio_data)}")
                # Create pcg_signal object
                signal_obj = pcg_signal(data=audio_data, fs=sample_rate)
                return signal_obj
            else:
                logger.debug(f"Loading audio with soundfile (fallback for non-pyPCG): {file_path}")
                data, sample_rate = sf.read(str(file_path), dtype='float32', always_2d=False)
                logger.info(f"Loaded {file_path} with soundfile. Original fs: {sample_rate}, Samples: {len(data)}")
                if data.ndim > 1:
                    logger.warning(f"Audio file {file_path} is multi-channel ({data.shape[1]} channels). Using only the first channel.")
                    data = data[:, 0]
                return data, int(sample_rate)

        except (sf.SoundFileError, IOError, OSError) as e_io:
            logger.error(f"I/O or SoundFile error loading {file_path}: {e_io}")
            raise RuntimeError(f"Failed to load audio file {file_path} due to I/O or format error.") from e_io
        except Exception as e_other:
            logger.error(f"Unexpected error loading audio file {file_path}: {e_other}")
            raise RuntimeError(f"An unexpected error occurred while loading {file_path}.") from e_other
    
    def save_audio(self, signal: Union[np.ndarray, PCGSignalType], sample_rate: int, file_path: Union[str, Path]) -> None:
        """Save a signal to a WAV file.
        
        Args:
            signal: The audio signal data to save. This can be a pyPCG `pcg_signal`
                object or a NumPy array.
            sample_rate: The sample rate in Hz. This is primarily used if `signal`
                is a NumPy array. If `signal` is a `pcg_signal` object, its own
                `fs` attribute is typically used by `pyPCG.io.write_signal_file`,
                and this parameter may be ignored for `pcg_signal` objects.
            file_path: The path (string or Path object) where the WAV file will be saved.

        Raises:
            RuntimeError: If saving the audio file fails due to an I/O error or other
                unexpected error during the writing process.
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if signal is a pyPCG object (if pcg_signal was imported) 
            # and if the specific pyPCG_write_signal_file function is available
            if HAS_PYPCG and pcg_signal is not None and isinstance(signal, pcg_signal) and pyPCG_write_signal_file is not None:
                logger.debug(f"Attempting to save with pyPCG_write_signal_file to {file_path}")
                pyPCG_write_signal_file(signal, str(file_path)) # Use the imported function
            else:
                if HAS_PYPCG and pcg_signal is not None and isinstance(signal, pcg_signal) and pyPCG_write_signal_file is None:
                    logger.warning(f"pyPCG core is available but pyPCG_write_signal_file is not. Using fallback to save {file_path}")
                # Fallback to scipy/soundfile for non-pyPCG objects or if pyPCG_write_signal_file is unavailable
                # Ensure proper data type and range
                signal_array = np.asarray(signal, dtype=np.float32)
                
                # Ensure signal is 1D (fix the shape error)
                if signal_array.ndim > 1:
                    logger.debug(f"save_audio (scipy fallback): signal_array was {signal_array.ndim}D, flattening to 1D.")
                    signal_array = signal_array.flatten()
                
                signal_array = np.clip(signal_array, -1.0, 1.0)
    
                # Convert to int16 for wav file, as expected by scipy.io.wavfile.write for standard PCM WAV
                signal_int16 = (signal_array * 32767).astype(np.int16)
                logger.debug(f"save_audio (scipy fallback): Saving signal of shape {signal_int16.shape}, dtype {signal_int16.dtype}, fs {sample_rate} to {file_path}")
                wavfile.write(str(file_path), sample_rate, signal_int16)
            
            logger.info(f"Saved audio to {file_path}")

        except (IOError, OSError) as e_io:
            logger.error(f"I/O error saving audio to {file_path}: {e_io}")
            raise RuntimeError(f"Failed to save audio file {file_path} due to I/O error.") from e_io
        except Exception as e_other:
            logger.error(f"Unexpected error saving audio to {file_path}: {e_other}")
            # Re-raise the original exception to preserve its type and traceback for unexpected errors
            raise
    
