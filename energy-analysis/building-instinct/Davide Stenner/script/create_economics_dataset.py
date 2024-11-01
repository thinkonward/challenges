import os
import sys
sys.path.append(os.getcwd())

if __name__ == '__main__':
    """Create economics dataset"""
    import json
    import polars as pl
    
    from src.utils.import_utils import import_config
    
    config_dict = import_config()
    
    def import_data_census(path_file: str) -> pl.DataFrame:
        data = (
            pl.read_csv(path_file)
            .head(63)
            .drop('Fact Note')
        )
        
        data = (
            data
            .drop(
                [
                    col for col in 
                    data.collect_schema().names()
                    if 'value' in col.lower()
                ]
            )
            .unpivot(index='Fact', variable_name='state', value_name='value')
        )
        return data
                
    data = pl.concat(
        [
            import_data_census(
                os.path.join(
                    r'data\other_dataset\census',
                    file_path
                )
            )
            for file_path in os.listdir(
                r'data\other_dataset\census'
            )
        ],
    )
    data = data.pivot(on='Fact', index='state', values='value')
    
    mapper_state = {
        "AL": "Alabama",
        "AK": "Alaska",
        "AS": "American Samoa",
        "AZ": "Arizona",
        "AR": "Arkansas",
        "CA": "California",
        "CO": "Colorado",
        "CT": "Connecticut",
        "DE": "Delaware",
        "DC": "District Of Columbia",
        "FM": "Federated States Of Micronesia",
        "FL": "Florida",
        "GA": "Georgia",
        "GU": "Guam",
        "HI": "Hawaii",
        "ID": "Idaho",
        "IL": "Illinois",
        "IN": "Indiana",
        "IA": "Iowa",
        "KS": "Kansas",
        "KY": "Kentucky",
        "LA": "Louisiana",
        "ME": "Maine",
        "MH": "Marshall Islands",
        "MD": "Maryland",
        "MA": "Massachusetts",
        "MI": "Michigan",
        "MN": "Minnesota",
        "MS": "Mississippi",
        "MO": "Missouri",
        "MT": "Montana",
        "NE": "Nebraska",
        "NV": "Nevada",
        "NH": "New Hampshire",
        "NJ": "New Jersey",
        "NM": "New Mexico",
        "NY": "New York",
        "NC": "North Carolina",
        "ND": "North Dakota",
        "MP": "Northern Mariana Islands",
        "OH": "Ohio",
        "OK": "Oklahoma",
        "OR": "Oregon",
        "PW": "Palau",
        "PA": "Pennsylvania",
        "PR": "Puerto Rico",
        "RI": "Rhode Island",
        "SC": "South Carolina",
        "SD": "South Dakota",
        "TN": "Tennessee",
        "TX": "Texas",
        "UT": "Utah",
        "VT": "Vermont",
        "VI": "Virgin Islands",
        "VA": "Virginia",
        "WA": "Washington",
        "WV": "West Virginia",
        "WI": "Wisconsin",
        "WY": "Wyoming"
    }
    mapper_state = {
        value_.upper(): key_.upper()
        for key_, value_ in mapper_state.items()
    }
    
    renamer_dict = {
        "Population, percent change - April 1, 2020 (estimates base) to July 1, 2023, (V2023)": "pop_change_2020_2023",
        "Persons under 5 years, percent": "under_5_percent",
        "Persons under 18 years, percent": "under_18_percent",
        "Persons 65 years and over, percent": "over_65_percent",
        "Female persons, percent": "female_percent",
        "White alone, percent": "white_percent",
        "Black or African American alone, percent": "black_percent",
        "American Indian and Alaska Native alone, percent": "native_percent",
        "Asian alone, percent": "asian_percent",
        "Native Hawaiian and Other Pacific Islander alone, percent": "pacific_islander_percent",
        "Two or More Races, percent": "two_or_more_races_percent",
        "Hispanic or Latino, percent": "hispanic_percent",
        "White alone, not Hispanic or Latino, percent": "white_non_hispanic_percent",
        "Foreign born persons, percent, 2018-2022": "foreign_born_percent",
        "Owner-occupied housing unit rate, 2018-2022": "owner_occupied_rate",
        "Living in same house 1 year ago, percent of persons age 1 year+, 2018-2022": "same_house_1yr_percent",
        "Language other than English spoken at home, percent of persons age 5 years+, 2018-2022": "non_english_home_percent",
        "Households with a computer, percent, 2018-2022": "computer_percent",
        "Households with a broadband Internet subscription, percent, 2018-2022": "broadband_percent",
        "High school graduate or higher, percent of persons age 25 years+, 2018-2022": "high_school_grad_percent",
        "Bachelor's degree or higher, percent of persons age 25 years+, 2018-2022": "bachelors_percent",
        "With a disability, under age 65 years, percent, 2018-2022": "disability_percent",
        "Persons without health insurance, under age 65 years, percent": "no_health_insurance_percent",
        "In civilian labor force, total, percent of population age 16 years+, 2018-2022": "labor_force_total_percent",
        "In civilian labor force, female, percent of population age 16 years+, 2018-2022": "labor_force_female_percent",
        "Persons in poverty, percent": "poverty_percent",
        "Total employment, percent change, 2021-2022": "employment_change_2021_2022",
        "Population estimates, July 1, 2023, (V2023)": "pop_estimates_2023",
        "Population estimates base, April 1, 2020, (V2023)": "pop_estimates_base_2020",
        "Population, Census, April 1, 2020": "pop_census_2020",
        "Population, Census, April 1, 2010": "pop_census_2010",
        "Veterans, 2018-2022": "veterans_2018_2022",
        "Housing Units, July 1, 2023, (V2023)": "housing_units_2023",
        "Median value of owner-occupied housing units, 2018-2022": "median_house_value",
        "Median selected monthly owner costs -with a mortgage, 2018-2022": "median_costs_mortgage",
        "Median selected monthly owner costs -without a mortgage, 2018-2022": "median_costs_no_mortgage",
        "Median gross rent, 2018-2022": "median_rent",
        "Building Permits, 2023": "building_permits_2023",
        "Households, 2018-2022": "households_2018_2022",
        "Persons per household, 2018-2022": "persons_per_household",
        "Total accommodation and food services sales, 2017 ($1,000)": "accommodation_sales_2017",
        "Total health care and social assistance receipts/revenue, 2017 ($1,000)": "healthcare_revenue_2017",
        "Total transportation and warehousing receipts/revenue, 2017 ($1,000)": "transport_warehouse_revenue_2017",
        "Total retail sales, 2017 ($1,000)": "retail_sales_2017",
        "Total retail sales per capita, 2017": "retail_sales_per_capita_2017",
        "Mean travel time to work (minutes), workers age 16 years+, 2018-2022": "mean_travel_time",
        "Median household income (in 2022 dollars), 2018-2022": "median_income_2022",
        "Per capita income in past 12 months (in 2022 dollars), 2018-2022": "per_capita_income_2022",
        "Total employer establishments, 2022": "employer_establishments_2022",
        "Total employment, 2022": "total_employment_2022",
        "Total annual payroll, 2022 ($1,000)": "annual_payroll_2022",
        "Total nonemployer establishments, 2021": "nonemployer_establishments_2021",
        "All employer firms, Reference year 2017": "all_employer_firms_2017",
        "Men-owned employer firms, Reference year 2017": "men_owned_firms_2017",
        "Women-owned employer firms, Reference year 2017": "women_owned_firms_2017",
        "Minority-owned employer firms, Reference year 2017": "minority_owned_firms_2017",
        "Nonminority-owned employer firms, Reference year 2017": "nonminority_owned_firms_2017",
        "Veteran-owned employer firms, Reference year 2017": "veteran_owned_firms_2017",
        "Nonveteran-owned employer firms, Reference year 2017": "nonveteran_owned_firms_2017",
        "Population per square mile, 2020": "pop_density_2020",
        "Population per square mile, 2010": "pop_density_2010",
        "Land area in square miles, 2020": "land_area_2020",
        "Land area in square miles, 2010": "land_area_2010"
        }


    percent_col_list = [
        col
        for col in data.columns
        if data.select(pl.col(col).str.contains('%').any()).item()
    ]
    with open(
        os.path.join(
            config_dict['PATH_MAPPER_DATA'], 'mapper_category.json'
        ), 'r'            
    ) as file_dtype:
        
        mapper_dataset = json.load(file_dtype)
        
    (
        data.select(
            [
                'state'
            ] +
            [
                pl.col(col).str.replace('Z', '0').str.replace('%', '').cast(pl.Float64)/100 
                for col in percent_col_list
            ] +
            [
                (
                    pl.col(col)
                    .str.replace_all(',', '', literal=True)
                    .str.replace_all('$', '', literal=True)
                    .replace('S', None)
                    .cast(pl.Float64)
                )
                for col in data.columns if col not in ['state'] + percent_col_list
            ]
        )
        .sort('state')
        .with_columns(
            (
                pl.col('state').str.to_uppercase()
                .replace(mapper_state)
                .replace(mapper_dataset['train_data']['in.state'])
                .cast(pl.UInt8)
            )
        )
        .rename(renamer_dict)
        .write_parquet(
            os.path.join(
                config_dict['PATH_SILVER_PARQUET_DATA'],
                'macro_economics_data.parquet'
            )
        )
    )
