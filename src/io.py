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
    from pyPCG import pcg_signal
    from pyPCG.io import read_signal_file
    from pyPCG.preprocessing import resample, filter as pcg_filter, envelope as pcg_envelope
    from pyPCG import normalize as pcg_normalize
    logger.info("All pyPCG modules for io.py appear to be imported.")
    HAS_PYPCG = True
    logger.info("pyPCG successfully confirmed in io.py. Using pyPCG for I/O and preprocessing.")
except ImportError as e:
    logger.debug(f"pyPCG import failed: {e}")
    try:
        # Fallback to scipy if pyPCG not available
        from scipy.io import wavfile
        import scipy.signal as sp_signal
        HAS_PYPCG = False
        logger.warning("pyPCG not found, falling back to scipy for basic audio operations")
    except ImportError as ie:
        logger.error("Neither pyPCG nor scipy is available for audio processing")
        raise ImportError("Either pyPCG or scipy is required for audio processing") from ie

class AudioLoader:
    """Handles loading, preprocessing, and saving of heart sound audio files."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config.get('audio', {})
        self.target_sample_rate = self.config.get('sample_rate', 4000)
        self._init_processing_pipeline()
    
    def _init_processing_pipeline(self):
        """Initialize the signal processing pipeline."""
        if HAS_PYPCG:
            # Use pyPCG's filter functions if available
            self.lp_filter = partial(
                pcg_filter,
                filt_ord=4,
                filt_cutfreq=400,
                filt_type='LP'
            )
            
            self.hp_filter = partial(
                pcg_filter,
                filt_ord=4,
                filt_cutfreq=20,
                filt_type='HP'
            )
        else:
            # Fallback to scipy.signal for filtering
            self.lp_filter = partial(
                self._scipy_filter,
                lowcut=None,
                highcut=400,
                order=4
            )
            self.hp_filter = partial(
                self._scipy_filter,
                lowcut=20,
                highcut=None,
                order=4
            )
    
    def _scipy_filter(self, signal: np.ndarray, sample_rate: float, lowcut: Optional[float] = None, 
                     highcut: Optional[float] = None, order: int = 4) -> np.ndarray:
        """Apply a Butterworth filter using scipy.signal."""
        nyq = 0.5 * sample_rate
        btype = 'band'
        
        if lowcut and highcut:
            low = lowcut / nyq
            high = highcut / nyq
            Wn = [low, high]
        elif lowcut:
            Wn = lowcut / nyq
            btype = 'high'
        elif highcut:
            Wn = highcut / nyq
            btype = 'low'
        else:
            return signal
            
        b, a = sp_signal.butter(order, Wn, btype=btype, analog=False)
        return sp_signal.filtfilt(b, a, signal)

    def load_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load an audio file and return signal data and sample rate.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            tuple: (signal_data, sample_rate)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        logger.info(f"Loading audio file: {file_path}")
        
        try:
            if HAS_PYPCG:
                # Use pyPCG's file loading
                file_ext = file_path.suffix.lower().lstrip('.')
                data, fs = read_signal_file(str(file_path), format=file_ext)
                signal = pcg_signal(data, fs)
                
                # Resample if needed
                if fs != self.target_sample_rate:
                    logger.info(f"Resampling from {fs}Hz to {self.target_sample_rate}Hz")
                    signal = resample(signal, fs, self.target_sample_rate)
                    fs = self.target_sample_rate
                return signal.data, fs
                
            else:
                # Fallback to scipy
                fs, data = wavfile.read(str(file_path))
                
                # Convert to mono if stereo
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                    
                # Resample if needed
                if fs != self.target_sample_rate:
                    logger.info(f"Resampling from {fs}Hz to {self.target_sample_rate}Hz")
                    num_samples = int(len(data) * self.target_sample_rate / fs)
                    data = sp_signal.resample(data, num_samples)
                    fs = self.target_sample_rate
                    
                return data, fs
                
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def preprocess(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply preprocessing steps to the signal.
        
        Args:
            signal: Input signal as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed signal as numpy array
        """
        try:
            logger.info("Preprocessing signal...")
            
            # Convert to float32 if needed
            if signal.dtype != np.float32:
                signal = signal.astype(np.float32)
                
            # Apply processing steps
            if HAS_PYPCG:
                # Create pyPCG signal object
                pcg_sig = pcg_signal(signal, sample_rate)
                
                # Apply processing chain
                processed = pcg_normalize(pcg_sig)
                processed = self.lp_filter(processed)
                processed = self.hp_filter(processed)
                
                # Extract envelope if needed
                if hasattr(processed, 'data'):
                    processed = pcg_envelope(processed)
                    return processed.data
                return processed
                
            else:
                # Fallback processing with scipy
                # Normalize
                signal = signal / np.max(np.abs(signal))
                
                # Apply filters
                signal = self.lp_filter(signal, sample_rate)
                signal = self.hp_filter(signal, sample_rate)
                
                return signal
                
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise
    
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
                signal = np.asarray(signal, dtype=np.float32)
                signal = np.clip(signal, -1.0, 1.0)
                wavfile.write(str(file_path), sample_rate, signal)
                
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
