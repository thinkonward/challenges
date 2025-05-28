import subprocess

command_list = [
    # Download process and clean data
    'python src/downloader.py',
    'python src/cleaner.py',
    'python src/storm_outages.py',
    'python src/meteorological_api.py',
    # Generate datasets for model
    'python src/data_dataset_creation.py',
    'python src/feature_generation.py',
    'python src/dataset_splitting.py',
    # Model generation and explanation
    'python src/training_model.py',
    'python src/conformal_predictions.py'
]

for command in command_list:
    subprocess.run(command.split(' '))
