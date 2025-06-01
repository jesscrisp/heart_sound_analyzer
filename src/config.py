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
                'peak_threshold': 0.1, # This default will be overridden by file if loaded
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
        print("Warning: No configuration file path provided. Using default configuration.")
        return default_config
    
    # Try to load config from file
    try:
        # Ensure the path is absolute or correctly relative to the CWD
        # When run with `python -m src.main`, CWD is project root.
        # `config_path` (e.g., 'config/config.yaml') should be relative to project root.
        path_to_open = Path(config_path)
        if not path_to_open.is_absolute():
            # This assumes that if it's not absolute, it's relative to the CWD (project root)
            # which is usually the case for command line args.
            pass # Keep it as is, Python's open will resolve from CWD

        with open(path_to_open, 'r') as f:
            file_config = yaml.safe_load(f)
        
        # Merge with default config (file config overrides defaults)
        if file_config:
            # print(f"Successfully loaded config from {path_to_open.resolve()}") # Debug print
            return _deep_merge(default_config, file_config)
        # print(f"Warning: Config file {path_to_open.resolve()} was empty. Using default configuration.")
        return default_config
            
    except FileNotFoundError:
        print(f"Warning: Configuration file not found at {Path(config_path).resolve()}. Using default configuration.")
        return default_config
    except Exception as e:
        print(f"Warning: Could not load config from {Path(config_path).resolve()}: {e}. Using default configuration.")
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
