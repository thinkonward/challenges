import geopandas as gpd
import json
import os
import pandas as pd
import pickle
import urllib
import yaml

from io import BytesIO
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor


RANDOM_SEED = 42
CUSTOMERS_OUT_NB = 10**3.5 # 10**3.5
STATE_ABBREVIATIONS = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    'United States Virgin Islands': 'VI'
}

SHAP_COLOR_PALLETE = (
    '#008bfb',
    '#007bf4',
    '#3569e8',
    '#6657d9',
    '#8443c6',
    '#a21eaa',
    '#bc009f',
    '#d6008e',
    '#e9007d',
    '#f80068',
    '#ff0055',
)

SET_COLOR_DICT = {
    'train': '#f8cc62',
    'test': '#bba681',
    'eval': '#737373',
    'cal': '#41596a',
    'OOT': '#7f95a4',
}


def get_general_path():
    """
    Get the general (root) path of the project by navigating one level up from the current file location.

    :return: general_path: str
        The absolute path to the project's root directory.
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    general_path = os.path.join(file_path, '..')
    return general_path


def join_paths(*p1):
    """
    Join multiple path components into a single path string.

    :param p1: str
        One or more path components to be joined.
    :return: joined_path: str
        The combined path.
    """
    joined_path = os.path.join(*p1)
    return joined_path


def check_if_filepath_exists(filepath):
    """
    Check whether a given file or directory path exists.

    :param filepath: str
        Path to the file or directory to check.
    :return: exists: bool
        True if the path exists, False otherwise.
    """
    exists = os.path.exists(filepath)
    return exists


def make_desired_folder(data_file_path):
    """
    Create a folder at the specified relative path inside the project's root directory if it doesn't already exist.

    :param data_file_path: str
        Relative path from the project root where the folder should be created.
    :return: None
    """
    general_path = get_general_path()
    file_path = join_paths(general_path, data_file_path)
    exists = check_if_filepath_exists(file_path)
    if not exists:
        os.makedirs(file_path)
    return None


def get_data_path(name):
    """
    Construct the full path to a file located in the project's 'data' folder.

    :param name: str
        Name of the file or subdirectory inside the 'data' folder.
    :return: file_path: str
        Full path to the requested data file or directory.
    """
    general_path = get_general_path()
    file_path = join_paths(general_path, 'data', name)
    return file_path


def save_dataframe(filepath, dataframe, file_format='parquet'):
    """
    Save a pandas DataFrame to disk in the specified file format.

    Supported formats: 'parquet', 'csv', 'pickle'.

    :param filepath: str
        Path where the DataFrame will be saved.
    :param dataframe: pandas DataFrame
        The DataFrame to be saved.
    :param file_format: str, optional
        File format to use for saving. Options are 'parquet', 'csv', or 'pickle'. Default is 'parquet'.
    :return: None
    """
    if file_format == 'parquet':
        dataframe.to_parquet(filepath)
    elif file_format == 'csv':
        dataframe.to_csv(filepath)
    elif file_format == 'pickle':
        dataframe.to_pickle(filepath)
    print(f'Data was saved into `{filepath}`.')


def get_config_file():
    """
    Load the configuration file containing URL links and settings.

    This function:
      - Constructs the path to 'config.yaml' located in the 'config' folder.
      - Loads and parses the YAML configuration into a Python dictionary.

    :return: config: dict
        Dictionary containing configuration values (e.g., URLs, paths, flags).
    """
    general_path = get_general_path()
    yaml_path = join_paths(general_path, 'config','config.yaml')
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    return config


def get_model_hyperparameters():
    """
    Load model hyperparameters from a JSON configuration file.

    This function:
      - Constructs the path to 'model_params.json' located in the 'config' folder.
      - Reads and parses the JSON file into a dictionary.

    :return: hyperparams: dict
        Dictionary containing model hyperparameters.
    """
    general_path = get_general_path()
    json_path = join_paths(general_path, 'config', 'model_params.json')
    with open(json_path, 'r') as f:
        hyperparams = json.loads(f)
    return hyperparams


def get_parameters_file():
    """
    Access and load the parameter definitions from the POWER_Parameter_Manager CSV file.

    This function:
      - Constructs the path to the parameters file located under the 'config' folder.
      - Reads the file and extracts the first 20 rows from the 'Parameter(s):' column.

    :return: params: dict
        Dictionary of parameter names from the 'Parameter(s):' column.
    """
    general_path = get_general_path()
    params_path = join_paths(general_path, 'config','POWER_Parameter_Manager.csv')
    params = pd.read_csv(params_path, engine='python', header=1).head(20).to_dict()['Parameter(s):']
    return params


def key_list(dictionary):
    """
    Get the keys of a dictionary as a list.

    :param dictionary: dict
        Dictionary from which to extract keys.
    :return: key_list_dict: list
        List containing the keys of the input dictionary.
    """
    key_list_dict = list(dictionary.keys())
    return key_list_dict


def save_json_from_url_zip(url, save_data_path, verbose=False):
    """
    Download a ZIP archive from a URL and extract all JSON files.

    This function:
      - Sends a request to download the ZIP archive.
      - Extracts only `.json` files into the specified directory.
      - Optionally prints the name of each extracted file if `verbose` is True.

    :param url: str
        URL pointing to the ZIP archive containing JSON files.
    :param save_data_path: str
        Directory where the extracted JSON files will be saved.
    :param verbose: bool, optional
        If True, prints the name of each file extracted. Default is False.
    :return: None
    """
    print('Downloading info...')
    req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
    url_response = urllib.request.urlopen(req)
    zip_file = ZipFile(BytesIO(url_response.read()))
    for f in zip_file.namelist():
        if f.endswith('.json'):
            zip_file.extract(f, path=save_data_path)
            if verbose:
                print(f'Extracting {f} into {save_data_path}')
    print(f'Done with extraction into {save_data_path}.')
    return None


def save_json_from_url_zip_parallel(url, save_data_path, verbose=False):
    """
    Download a ZIP archive from a URL and extract all JSON files in parallel.

    This function:
      - Sends a request to the given URL and retrieves the ZIP archive.
      - Extracts all `.json` files using multithreading for faster processing.
      - Optionally logs the extracted filenames if `verbose` is True.

    :param url: str
        URL pointing to a ZIP archive containing JSON files.
    :param save_data_path: str
        Directory path where extracted files should be saved.
    :param verbose: bool, optional
        If True, prints the name of each extracted file. Default is False.
    :return: None
    """
    print('Downloading info...')
    req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
    url_response = urllib.request.urlopen(req)
    zip_file = ZipFile(BytesIO(url_response.read()))

    def extract_file(f):
        if f.endswith('.json'):
            zip_file.extract(f, path=save_data_path)
            if verbose:
                print(f'Extracting {f} into {save_data_path}')

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(extract_file, zip_file.namelist())

    print(f'Done with extraction into {save_data_path}.')
    return None


def save_shapefile_from_url_zip(url, save_data_path):
    """
    Download a ZIP file containing shapefiles from a URL and extract the required components.

    This function:
      - Checks if the data already exists at the given path.
      - If not, downloads and extracts `.shp`, `.shx`, and `.dbf` files from the ZIP archive.
      - Identifies the main `.shp` file and returns its full path.

    :param url: str
        URL pointing to a ZIP file containing shapefile components.
    :param save_data_path: str
        Local directory path to save the extracted shapefile components.
    :return: main_file: str
        Full path to the extracted `.shp` file.
    """
    if not check_if_filepath_exists(save_data_path):
        print('Downloading info...')
        req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
        url_response = urllib.request.urlopen(req)
        zip_file = ZipFile(BytesIO(url_response.read()))

        for f in zip_file.namelist():
            print(f)
            if f.endswith('shx') or f.endswith('shp') or f.endswith('dbf'):
                zip_file.extract(f, path=save_data_path)
                print(f'Extracting {f} into {save_data_path}')
            if f.endswith('shp'):
                file_name = f
    else:
        print('Info already exists...')
        for f in os.listdir(save_data_path):
            if f.endswith('shp'):
                file_name = f
    main_file = os.path.join(save_data_path, file_name)
    return main_file


def save_info(main_file, filepath):
    """
    Save geospatial data from a file to a specified path if it doesn't already exist.

    This function:
      - Reads the input file as a GeoDataFrame.
      - Saves it to disk using a helper function if the target path does not already exist.

    :param main_file: str
        Path to the main geospatial file (e.g., shapefile) to be read.
    :param filepath: str
        Destination path to save the GeoDataFrame.
    :return: None
    """
    if not check_if_filepath_exists(filepath):
        city_data = gpd.read_file(main_file)
        save_dataframe(filepath=filepath, dataframe=city_data)
    else:
        print(f'Information is already saved at: {filepath}')


def save_as_json(what, where):
    """
    Save a Python object as a JSON file if the file does not already exist.

    :param what: any
        The object to be serialized and saved (must be JSON-serializable).
    :param where: str
        The file path where the JSON should be saved.
    :return: None
    """
    if not check_if_filepath_exists(where):
        with open(where, 'w') as f:
            json.dump(what, f)


def save_as_pickle(what, where):
    """
    Save an object to a pickle file if the file does not already exist.

    :param what: any
        The object to be saved.
    :param where: str
        The path where the pickle file should be saved.
    :return: None
    """
    if not check_if_filepath_exists(where):
        with open(where, 'wb') as f:
            pickle.dump(what, f)


def read_pickle(where):
    """
    Load and return an object from a pickle file.

    :param where: str
        Path to the pickle file to be loaded.
    :return: loaded_file: any
        The object loaded from the pickle file.
    """
    with open(where, 'rb') as f:
        loaded_file = pickle.load(f)
        return loaded_file


def miniprocess_outage_raw_df(outages):
    """
    Perform basic preprocessing on raw outage data.

    This function:
      - Removes records with missing `customers_out`.
      - Filters for outages with a minimum number of affected customers (defined by `CUSTOMERS_OUT_NB`).
      - Converts `run_start_time` to datetime format.
      - Maps U.S. state names to abbreviations to create a `state_id`.
      - Pads FIPS codes to create uniform `fips_code_id`.
      - Generates a `sub_general_id` combining FIPS and state ID.

    :param outages: pandas DataFrame
        Raw outage data including columns like `customers_out`, `run_start_time`, `state`, and `fips_code`.
    :return: outages: pandas DataFrame
        Cleaned and transformed outage data ready for further processing.
    """
    print('Processing outages...')
    print('Deleting customers_out nulls...')
    outages = outages[outages.customers_out.notna()]  # Filter nan values from customers_out
    # Then we keep only the relevant outage (affecting a high amount of customers)
    print(f'Keeping relevant outages according to CUSTOMERS_OUT_NB={CUSTOMERS_OUT_NB}')
    outages = outages[outages.customers_out >= CUSTOMERS_OUT_NB]
    print('Changing run_start_time to datetime...')
    outages.run_start_time = pd.to_datetime(outages.run_start_time)  # Transform into datetime to manipulate dates
    print('Mapping state_ids...')
    outages["state_id"] = outages.state.map(STATE_ABBREVIATIONS) # Use the state abbreviations to get an ID
    print('Filling fips_code_ids...')
    outages["fips_code_id"] = outages.fips_code.astype(str).str.zfill(5)
    outages["sub_general_id"] = (outages.fips_code_id + '_' + outages.state_id)
    return outages


def get_required_outages_dfs(*years, eaglei_data_path=None):
    """
    Load and preprocess outage data for the specified years from the given EAGLE-I data path.

    This function:
      - Filters files in the directory to include only those matching the given years.
      - Reads each matching CSV file and applies `miniprocess_outage_raw_df`.
      - Concatenates multiple years' data into a single DataFrame if needed.

    :param years: int
        Variable number of year arguments to filter files by (e.g., 2019, 2020, 2021).
    :param eaglei_data_path: str
        Path to the directory containing EAGLE-I outage data files.
    :return: outages_df: pandas DataFrame
        A processed DataFrame containing outage data for the specified years.
    """
    eaglei_data_paths = os.listdir(eaglei_data_path)
    paths = []
    for year in years:
        paths += [join_paths(eaglei_data_path,file) for file in eaglei_data_paths if str(year) in file]
    dfs = []
    for file in paths:
        print(f"Reading file: {file}.")
        outage_data = pd.read_csv(file)
        outage = miniprocess_outage_raw_df(outage_data)
        dfs.append(outage)
    print("Done reading.")
    if len(paths) > 1:
        print("Merging information.")
        outages_df = pd.concat(dfs)
        del dfs
    else:
        outages_df = dfs[0]
    print('Data is ready.')

    return outages_df


def save_pickle_model(model:any, file_name:str='outage_model.pkl')->None:
    """
    Save a model object as a pickle file in the 'models' directory under the general path.

    :param model: any
        The model object to be saved.
    :param file_name: str, optional
        Name of the pickle file. Default is 'outage_model.pkl'.
    :return: None
    """
    general_path = get_general_path()
    model_folder = join_paths(general_path, 'models')
    os.makedirs(model_folder, exist_ok=True)
    model_path = join_paths(model_folder, file_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def get_record_from_df(df, pos):
    """
    Retrieve a specific row from a DataFrame based on its position.

    :param df: pandas DataFrame
        The DataFrame from which to extract the row.
    :param pos: int
        The integer position of the row to retrieve (e.g., 0 for first, -1 for last).
    :return: record: pandas Series
        The selected row as a Series.
    """
    record = df.iloc[pos]
    return record
