"""
Shared pytest fixtures for testing.
"""
import pytest
import numpy as np
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / 'test_data'

@pytest.fixture
def sample_audio_path():
    """Path to a sample audio file for testing."""
    return TEST_DATA_DIR / 'sample_heartbeat.wav'

@pytest.fixture
def sample_signal():
    """Generate a simple synthetic heart sound signal."""
    fs = 4000  # 4 kHz sample rate
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second of audio
    
    # Create a simple synthetic heart sound (S1 and S2)
    # Ensure s2 is the same length as the target slice
    s1 = 0.5 * np.sin(2 * np.pi * 50 * t) * np.exp(-20 * t)
    s2_duration = int(0.2 * fs)  # 200ms for S2
    s2_t = np.linspace(0, 0.2, s2_duration, endpoint=False)
    s2 = 0.3 * np.sin(2 * np.pi * 100 * s2_t) * np.exp(-30 * s2_t)
    
    signal = np.zeros_like(t)
    signal[:len(s1)] += s1
    
    # Place S2 at 400ms
    s2_start = int(0.4 * fs)
    s2_end = s2_start + len(s2)
    if s2_end > len(signal):
        s2 = s2[:len(signal) - s2_start]
    signal[s2_start:s2_start + len(s2)] += s2
    
    return signal, fs

@pytest.fixture
def sample_segments():
    """Sample segmentation results for testing."""
    return {
        's1_starts': [100, 2000],
        's1_ends': [300, 2200],
        's2_starts': [1500, 3500],
        's2_ends': [1700, 3700],
        'method': 'envelope'
    }
