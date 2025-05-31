"""Configuration loading and management."""
from pathlib import Path
from typing import Dict, Any
import yaml
import os

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file. If None, uses default config.
        
    Returns:
        Dictionary containing configuration parameters
    """
    # Default configuration
    default_config = {
        'audio': {
            'sample_rate': 4000,
            'normalize': True
        },
        'segmentation': {
            'method': 'peak_detection',
            'min_heart_rate': 40,
            'max_heart_rate': 200,
            'peak_detection': {
                'start_drop': 0.8,
                'end_drop': 0.7,
                'peak_threshold': 0.1,
                'peak_distance': 0.15,
                'peak_width': [0.01, 0.15]
            }
        },
        'preprocessing': {
            'lowcut': 25,
            'highcut': 400
        },
        'output': {
            'save_plots': True,
            'plot_format': 'png',
            'show_plots': False,
            'debug': False
        }
    }
    
    # If no config path provided, use default
    if not config_path:
        return default_config
    
    # Try to load config from file
    try:
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Merge with default config (file config overrides defaults)
        if file_config:
            return _deep_merge(default_config, file_config)
        return default_config
            
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return default_config

def _deep_merge(d1: Dict, d2: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = d1.copy()
    for k, v in d2.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
