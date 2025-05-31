# Heart Sound Analyzer

A Python application for segmenting and analyzing heart sounds using the pyPCG library.

## Features

- Load and preprocess heart sound recordings (WAV format)
- Segment heart sounds into S1 and S2 components
- Visualize segmentation results
- Export results to various formats

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd heart_sound_analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python -m src.main path/to/your/audio.wav --output-dir results/
```

### Configuration

Modify `config/settings.yaml` to adjust processing parameters.

## Project Structure

```
heart_sound_analyzer/
├── config/             # Configuration files
├── data/               # Data directories (raw/processed/results)
├── src/                # Source code
│   ├── __init__.py
│   ├── config.py      # Configuration loading
│   ├── io.py         # File I/O operations
│   ├── main.py       # Main entry point
│   ├── processor.py  # Processing pipeline
│   └── visualization.py
└── tests/             # Unit tests
```

## License

[Your License Here]
