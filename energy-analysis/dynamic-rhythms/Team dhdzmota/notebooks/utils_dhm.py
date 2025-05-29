import os
import pandas as pd


CUSTOMERS_OUT_NB = 10**3.5
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


def get_required_outages_dfs(*years, eaglei_data_path=None):
    eaglei_data_paths = os.listdir(eaglei_data_path)
    paths = []
    for year in years:
        paths += [f'{eaglei_data_path}{file}' for file in eaglei_data_paths if str(year) in file]
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


def miniprocess_outage_raw_df(outages):
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
    print(f"Outage Information:\n")
    outages.info()
    print('')
    return outages
