# Solution Information
## Challenge: Building Instinct: Where power meets predictions
## Team: [REDACTED]
## Email: [REDACTED]

### Python Version: 3.12.4

### Project Structure 

* ```~/solution_pipeline.ipynb``` - main notebook file, pipeline entrypoint;
* ```~/src``` - source code directory with helping modules for the pipeline;
* ```~/data``` - placeholder directory for the pipeline data;
* ```~/config``` - directory with pipeline configuration files;
* ```~/experiment``` - directory with saved model;
* ```~requirements.txt``` - list of required dependencies;

### Solution Reproduction Note
To  reproduce the solution follow the instructions in ```solution_pipeline.ipynb```

To retrain all ensure to put every train parquet files under data\original_dataset\building-instinct-train-data and 
data\original_dataset\building-instinct-train-label.

To predict only the houldout dataset put every test parquet files under data\original_dataset\building-instinct-test-data.

If there are any issues during the execution, please contact via email.

# Dev Info

## Set up data

Then unzip inside folder original_data

Create the required environment by executing following command:
```
//create venv
python -m venv .venv

//activate .venv
source .venv\Scripts\activate

//upgrade pip
python -m pip install --upgrade pip

//instal package in editable mode
python -m pip install -e .

//clean egg-info artifact
python setup.py clean
```

Download data for each state from https://www.census.gov/quickfacts/fact/table/{STATE}/PST045223
```
python concat_data.py
command.ps1
python create_additional_data.py
python create_economics_dataset.py
python preprocess.py --add
```

## Training phase

```
python train.py
```

## Inference
```
python inference.py
```