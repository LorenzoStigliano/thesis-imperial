"""
Directory Configurations for Thesis Project

This configuration file defines directory paths for organizing data, models, and figures
related to your thesis project. These paths help maintain a structured organization of
different aspects of your research and experiments.

Personal Directory Configuration:
- `DATA_DIR`: Directory containing unprocessed data needed for research.
- `SAVE_DIR_DATA`: Directory to store processed or modified data.
- `SAVE_DIR_MODEL_DATA`: Directory to store GNN model-related data.
- `SAVE_DIR_FIGS`: Directory to store figures and visualizations.

GPUs Directory Configuration (commented out):
- These paths are intended for using GPUs in a different environment.

Modify these directory paths according to your project's directory structure.

Usage:
1. Choose the appropriate directory configuration (personal or GPUs).
2. Update the paths as needed for your environment.
3. Import this configuration into your project files to access the defined paths.

Example:
    from directory_config import DATA_DIR, SAVE_DIR_DATA

    # Use DATA_DIR and SAVE_DIR_DATA to access respective directories.

Note: These paths are essential for maintaining an organized project structure.
"""

#Personal: /Users/lorenzostigliano/Documents/University/Imperial/Summer Term/thesis-imperial

DATA_DIR = '/Users/lorenzostigliano/Documents/University/Imperial/Summer Term/thesis-imperial/data_unprocessed/'
SAVE_DIR_DATA = '/Users/lorenzostigliano/Documents/University/Imperial/Summer Term/thesis-imperial/data/'
SAVE_DIR_MODEL_DATA = '/Users/lorenzostigliano/Documents/University/Imperial/Summer Term/thesis-imperial/model_data/'
SAVE_DIR_FIGS = '/Users/lorenzostigliano/Documents/University/Imperial/Summer Term/thesis-imperial/figures/'

"""
#GPUs: /vol/bitbucket/ls1121/

DATA_DIR = '/vol/bitbucket/ls1121/data_unprocessed/data_rh/'
SAVE_DIR_DATA = '/vol/bitbucket/ls1121/data/'
SAVE_DIR_MODEL_DATA = '/vol/bitbucket/ls1121/model_data/'
SAVE_DIR_FIGS = '/homes/ls1121/thesis-imperial/figures/'
"""