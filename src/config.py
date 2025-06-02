"""Configuration loading and management."""
import logging
from pathlib import Path
from typing import Dict, Any
import yaml
import os

logger = logging.getLogger(__name__)

# Default configuration file path, relative to the project root
# Assumes the script is run from the project root, or this path is relative to where config.py is located.
# To make it robustly point to PROJECT_ROOT/config/config.yaml from PROJECT_ROOT/src/config.py:
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Get to PyPCG_first_attempt
DEFAULT_PROJECT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

def load_config(cli_config_path_str: str = None) -> Dict[str, Any]:
    print("[CONFIG.PY-DEBUG] load_config entered.") # CASCADE DEBUG
    """Load configuration from YAML file.

    If cli_config_path_str is provided, attempts to load from that path.
    Otherwise, attempts to load from DEFAULT_PROJECT_CONFIG_PATH.
    Falls back to internal defaults if file loading fails.

    Args:
        cli_config_path_str: Path to the YAML configuration file from CLI (optional).

    Returns:
        Dictionary containing configuration parameters.
    """
    # Internal default configuration
    default_config = {
        'audio': {
            'sample_rate': 4000,
            'normalize': True
        },
        'preprocessing': {
            'resample_rate': 1000,
            'normalize_after_preprocessing': True,
            'filter': {
                'enabled': True,
                'lowcut': 25,
                'highcut': 200,
                'order': 4
            },
            'envelope': {
                'method': 'hilbert',
                'window_size': 0.01,
                'power': 1
            }
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
            },
            'lr_hsmm': {
                'model_path': 'models/lr_hsmm_model.json',
                'state_mapping': {
                    's1': 'S1',
                    's2': 'S2',
                    'systole': 'SYSTOLE',
                    'diastole': 'DIASTOLE'
                }
            }
        },
        'output': {
            'save_plots': True,
            'plot_format': 'png',
            'show_plots': False,
            'debug': False
        },
        'paths': {
            'data_raw': 'heart_sound_analyzer/data/raw',
            'data_processed': 'heart_sound_analyzer/data/processed',
            'results': 'heart_sound_analyzer/data/results',
            'models': 'heart_sound_analyzer/models'
        },
        'logging': {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }

    path_to_try_loading = None
    is_cli_path = False
    config_source_description = ""

    if cli_config_path_str:
        path_to_try_loading = Path(cli_config_path_str)
        is_cli_path = True
        config_source_description = f"CLI path: '{path_to_try_loading}'"
        print(f"[CONFIG.PY-DEBUG] CLI configuration path provided: '{path_to_try_loading}'. Attempting to load.") # CASCADE DEBUG
    else:
        # Check for DEFAULT_CONFIG_FILE environment variable
        default_config_file_env = os.getenv('DEFAULT_CONFIG_FILE')
        if default_config_file_env:
            path_to_try_loading = PROJECT_ROOT / default_config_file_env
            config_source_description = f"environment variable DEFAULT_CONFIG_FILE: '{path_to_try_loading}'"
            print(f"[CONFIG.PY-DEBUG] Using configuration path from environment variable DEFAULT_CONFIG_FILE: '{path_to_try_loading}'.") # CASCADE DEBUG
        else:
            path_to_try_loading = DEFAULT_PROJECT_CONFIG_PATH
            config_source_description = f"default project path: '{path_to_try_loading}'"
            print(f"[CONFIG.PY-DEBUG] No CLI path or DEFAULT_CONFIG_FILE env var. Attempting to load from default project path: '{path_to_try_loading}'.") # CASCADE DEBUG

    try:
        # Using resolve() for a more absolute path in logs, helps in debugging.
        # Note: resolve() will raise FileNotFoundError if the path doesn't exist, so call it carefully.
        # For logging the path that was *attempted*, path_to_try_loading (as string) is fine.
        print(f"[CONFIG.PY-DEBUG] Attempting to open and load YAML from: {path_to_try_loading}") # CASCADE DEBUG
        with open(path_to_try_loading, 'r') as f:
            file_config = yaml.safe_load(f)
            print(f"[CONFIG.PY-DEBUG] YAML loaded. Content snippet: {str(file_config)[:200]}...") # CASCADE DEBUG

        if file_config:
            print(f"[CONFIG.PY-DEBUG] Successfully loaded and merged configuration from {config_source_description} (resolved: '{path_to_try_loading.resolve()}').") # CASCADE DEBUG
            merged_config = _deep_merge(default_config, file_config)
            print(f"[CONFIG.PY-DEBUG] Merged config 'output' section: {merged_config.get('output')}") # CASCADE DEBUG
            return merged_config
        else:
            print(f"[CONFIG.PY-DEBUG] Configuration file from {config_source_description} (resolved: '{path_to_try_loading.resolve()}') is empty. Using internal default configuration.") # CASCADE DEBUG
            print(f"[CONFIG.PY-DEBUG] Default config 'output' section: {default_config.get('output')}") # CASCADE DEBUG
            return default_config

    except FileNotFoundError:
        resolved_path_str = str(path_to_try_loading.resolve(strict=False)) # Get path string even if it doesn't exist
        if is_cli_path:
            print(f"[CONFIG.PY-DEBUG] WARNING: Configuration file specified via {config_source_description} not found: '{resolved_path_str}'. Using internal default configuration.") # CASCADE DEBUG
        else:
            print(f"[CONFIG.PY-DEBUG] INFO: Configuration file from {config_source_description} not found at '{resolved_path_str}'. Using internal default configuration.") # CASCADE DEBUG
        print(f"[CONFIG.PY-DEBUG] Default config 'output' section (due to FileNotFoundError): {default_config.get('output')}") # CASCADE DEBUG
        return default_config
    except Exception as e:
        resolved_path_str = str(path_to_try_loading.resolve(strict=False))
        print(f"[CONFIG.PY-DEBUG] WARNING: Error loading configuration from {config_source_description} (resolved: '{resolved_path_str}'): {e}. Using internal default configuration.") # CASCADE DEBUG
        print(f"[CONFIG.PY-DEBUG] Default config 'output' section (due to Exception): {default_config.get('output')}") # CASCADE DEBUG
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
