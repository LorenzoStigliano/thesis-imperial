===========================================
Producing Thesis Results: Setup and Running
===========================================

This guide outlines the steps required to reproduce the results presented in the thesis. The results are generated using a series of scripts that train and evaluate various models on different datasets. Before running the scripts, ensure you have the required environment set up correctly.

====================
Environment Setup
====================

1. Create a Virtual Environment:

It is recommended to create a virtual environment to manage the dependencies for this project.
Run the following command in your terminal: python3 -m venv thesis-env

2. Activate the Virtual Environment:

Activate the created virtual environment using the following command:
For Linux/macOS: source thesis-env/bin/activate
For Windows: .\thesis-env\Scripts\activate

3. Install Dependencies:

Install the required Python packages within the virtual environment: pip install -r requirements.txt

4. CUDA Compatibility:

Make sure you have the correct version of CUDA installed. The scripts rely on CUDA 10.1.243. Ensure that this version of CUDA is available on your system.

====================
Running the Scripts
====================

1. Set CUDA and Environment Variables:

Before running the scripts, set the necessary environment variables in each script.

Activate virtual environment
source /vol/bitbucket/ls1121/doscond/bin/activate

Set PATH environment variable
export PATH=/vol/cuda/10.1.243/bin:$PATH

Set CPATH environment variable
export CPATH=/vol/cuda/10.1.243/include:$CPATH

Set LD_LIBRARY_PATH environment variable
export LD_LIBRARY_PATH=/vol/cuda/10.1.243/lib64:$LD_LIBRARY_PATH


2. Configure Model Parameters:

The model configurations are stored in the "model_config" folder. 
Each model has its own configuration file. These configuration files need to be manually 
configured to specify hyperparameters and settings for each model. We provide configs for
GCN to GCN tables.

3. Run the Scripts

To run the script simply: source /path/to/script
Note: run the scripts in order: setup_run_A, setup_run_B and setup_run_C, since they depend each other

4. Output:

The scripts will generate results based on the provided dataset and configurations. The output will include various metrics and evaluation data for the models trained.

===================================
Note: It's crucial to follow the order, configure model parameters in the respective configuration files, and ensure that the CUDA version is set correctly to reproduce the results accurately.
===================================

====================
Contact
====================

If you encounter any issues or have questions regarding the process, feel free to contact lorenzo.stigliano22@imperial.ac.uk
