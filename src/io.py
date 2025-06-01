"""
Input/Output operations for heart sound analysis.
Handles audio file loading, saving, and basic preprocessing.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
from functools import partial
import sys

# Get module logger first
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True # Ensure it propagates to root logger

# Try to import pyPCG modules with fallbacks
try:
    from pyPCG import pcg_signal # For type hinting and potentially creating objects
    from pyPCG.io import read_signal_file # For loading pyPCG signals
    logger.info("Essential pyPCG modules for io.py (pcg_signal, read_signal_file) imported.")
    HAS_PYPCG = True
except ImportError as e:
    logger.debug(f"Core pyPCG import failed in io.py: {e}. Will rely on scipy for audio loading.")
    HAS_PYPCG = False

# Always import soundfile for basic audio loading, and scipy for fallback if pyPCG fails for specific tasks
try:
    import soundfile as sf
except ImportError as e:
    logger.error("Soundfile library not found. It is required for basic audio loading.")
    raise ImportError("Soundfile library is required.") from e

if not HAS_PYPCG:
    try:
        from scipy.io import wavfile # For fallback .wav reading if soundfile fails or for specific cases
        # import scipy.signal as sp_signal # Not needed directly in io.py anymore
        logger.info("Scipy.io.wavfile imported for fallback WAV loading.")
    except ImportError as ie:
        logger.warning("Scipy.io.wavfile not found. Fallback WAV loading might be affected.")

class AudioLoader:
    """Handles loading, preprocessing, and saving of heart sound audio files."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config.get('audio', {})
        self.target_sample_rate = self.config.get('sample_rate', 4000) # Retained for now, but not used in io.py
    
    def load_audio(self, file_path: Union[str, Path]) -> Union['pcg_signal', Tuple[np.ndarray, int]]:
        """Load an audio file and return raw signal data and sample rate.
        If pyPCG is available, returns a pcg_signal object.
        Otherwise, returns a tuple (numpy_array, sample_rate).
        No internal resampling or filtering is performed here.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Union[pcg_signal, Tuple[np.ndarray, int]]: Raw signal object or (data, rate)
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

        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            # Consider if specific exceptions should be caught or if generic is okay
            raise RuntimeError(f"Failed to load audio file {file_path}") from e
    
    def save_audio(self, signal: np.ndarray, sample_rate: int, file_path: Union[str, Path]) -> None:
        """Save a signal to a WAV file.
        
        Args:
            signal: Signal data to save
            sample_rate: Sample rate in Hz
            file_path: Output file path
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if HAS_PYPCG and hasattr(signal, 'fs'):
                # Handle pyPCG signal object
                write_signal_file(signal, str(file_path))
            else:
                # Fallback to scipy
                from scipy.io import wavfile
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
            
        except Exception as e:
            logger.error(f"Error saving audio to {file_path}: {e}")
            raise
    
    @staticmethod
    def save_segmentation_results(
        results: Dict[str, Any],
        output_dir: str,
        base_name: Optional[str] = None
    ) -> None:
        """Save segmentation results to files.
        
        Args:
            results: Dictionary containing segmentation results
            output_dir: Directory to save results
            base_name: Base name for output files (defaults to input filename)
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if base_name is None:
                base_name = Path(results.get('file_path', 'output')).stem
            
            # Save segments to CSV
            segments = results.get('segments', {})
            if segments:
                import pandas as pd
                
                # Convert segments to DataFrame
                df = pd.DataFrame({
                    's1_start': segments.get('s1_starts', []),
                    's1_end': segments.get('s1_ends', []),
                    's2_start': segments.get('s2_starts', []),
                    's2_end': segments.get('s2_ends', []),
                })
                
                # Save to CSV
                csv_path = output_dir / f"{base_name}_segments.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved segmentation results to {csv_path}")
                
        except Exception as e:
            logger.error(f"Error saving segmentation results: {e}")
            raise
