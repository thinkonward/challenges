## Building Instinct: Where power meets predictions: 1st place solution code (public)

Competition: Building Instinct: Where power meets predictions
Team name: [REDACTED]

This repo contains: 

- Explanation of the approach.
- Explanation of code needed for downloading external data, preprocess, training and inference.
- Requirements and steps needed for running the solution

## APPROACH

I did other markdown for explaining it: "Approach.md"

### Environment

To setup the environment:
* Install python3.9 or higher 
* Install `requirements.txt` in the fresh python environment (pip install -r requirements.txt)

### Steps to reproduce LB results

1. Download and extract the data of the competition in ./data/onward
2. Install requirements.txt
3. Make sure to install **aws** library in your machine for running `download_external_data.py`
4. Make sure to install 'trash-empty' library on your machine for running line 177 of `download_external_data.py` (Every time I download and process a new state, I delete it for saving memory)
5. Run the script `runs.py` where you will find the steps of the pipeline (download,preprocess,training,inference)

In the runs.py is every step of the pipeline. 

### Explanation of each file

- train.py : parameterized script that serves to train models for both commercial metadata and residential metadata
- train_binary.py : script to train the models that allow us to differentiate between residential and commercial
- runs.py: This script is made so that the entire solution can be reproduced step by step (where the steps match those of the image posted in approach.md)
- lofo.py: Script used to make variable selection
- inference.py : same that inference notebook but save as a script
- inference.ipynb : notebook for running the inference in any unseen data

- ./data/ : all kinds of data will be stored here, below I explain how:
  - ./data/onward : data download from onward platform
  - ./data/ext_data* : raw data downloaded from the data lake, we use these folders as temporary folders, since once the data in these folders are processed they will be automatically deleted (therefore their natural state is empty)
  - ./data/ext_proc* : the raw data downloaded in ext_data* is unified into a single dataframe for each state and only the necessary columns are saved in these folders
  - ./data/data_ready* : we run the preprocessing of the time series and extract many variables that are stored in these folders
  - ./data/labels: here we store the labels of each of the downloaded external datasets
  -./data/long_lat.csv: csv with the geographic information of each state

- ./imp/ : here we store the importance of the variables extracted from `lofo.py`
- ./models/ : all models checkpoints are here
  - ./models/MODELS_binary.pkl -> model for predicting residential or commercial
  - ./models/MODELS_commercial.pkl -> model for predicting commercial metadata
  - ./models/MODELS_residential.pkl -> model for predicting residential metadata

- ./src/ : some scripts are saved here


