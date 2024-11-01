import logging

import polars as pl
import polars.selectors as cs

from typing import Mapping, Union, Tuple, Dict, Any

def get_mapper_categorical(
        config_dict: Dict[str, Any],
        data: Union[pl.LazyFrame, pl.DataFrame], 
        logger: logging.Logger,
        message_format: str='{col} has over {n_unique} different values'
    ) -> Tuple[Union[pl.LazyFrame, pl.DataFrame], Mapping[str, int]]:
    """
    check dataset and return int remapped dataset and the mapper

    Args:
        data (Union[pl.LazyFrame, pl.DataFrame]): dataset

    Returns:
        Union[pl.LazyFrame, pl.DataFrame], Mapping[str, int]: dataset and mapping dictionary
    """
    mapper_mask_col = {}
    lazy_mode = isinstance(data, pl.LazyFrame)

    categorical_col = (
        data.drop(config_dict['BUILDING_ID']).select(cs.by_dtype(pl.String)).collect_schema().names()
        if lazy_mode
        else 
        data.drop(config_dict['BUILDING_ID']).select(cs.by_dtype(pl.String)).columns
    )
    for col in categorical_col:
        
        unique_values = (
            data.select(col).drop_nulls().collect()[col].unique() 
            if lazy_mode 
            else data[col].drop_nulls().unique()
        )
        
        mapper_mask_col[col] = {
            value: i 
            for i, value in enumerate(unique_values.sort().to_list())
        }
        logger.info(
            message_format.format(
                col=col, 
                n_unique=len(unique_values)
            )
        )
    logger.info('\n\n')
    data = remap_category(data=data, mapper_mask_col=mapper_mask_col)
    return data, mapper_mask_col

def remap_category(
        data: Union[pl.LazyFrame, pl.DataFrame], 
        mapper_mask_col: Dict[str, int]
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
    data = data.with_columns(
        [
            pl.col(col_name).replace(replace_mask, default=None).cast(pl.UInt8)
            for col_name, replace_mask in mapper_mask_col.items()
        ]
    )
    return data

def correct_ordinal_categorical(mapper_label: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    mapper_label["in.number_of_stories_com"] = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9, "11": 10, "12": 11, "14": 12, "20": 13, "30": 14}
    mapper_label["in.vintage_com"] = {
        "Before 1946": 0,
        "1946 to 1959": 1,
        "1960 to 1969": 2,
        "1970 to 1979": 3,
        "1980 to 1989": 4,
        "1990 to 1999": 5,
        "2000 to 2012": 6,
        "2013 to 2018": 7,
    }
    mapper_label["in.weekday_opening_time..hr_com"] = {
        "3.5": 0,
        "3.75": 1,
        "4": 2,
        "4.25": 3,
        "4.5": 4,
        "4.75": 5,
        "5": 6,
        "5.25": 7,
        "5.5": 8,
        "5.75": 9,
        "6": 10,
        "6.25": 11,
        "6.5": 12,
        "6.75": 13,
        "7": 14,
        "7.25": 15,
        "7.5": 16,
        "7.75": 17,
        "8": 18,
        "8.25": 19,
        "8.5": 20,
        "8.75": 21,
        "9": 22,
        "9.25": 23,
        "9.5": 24,
        "9.75": 25,
        "10": 26,
        "10.25": 27,
        "10.5": 28,
        "10.75": 29,
        "11": 30,
        "11.25": 31,
        "11.5": 32,
        "11.75": 33,
        "12": 34,
        "12.25": 35,
    }
    mapper_label["in.weekday_operating_hours..hr_com"] = {
        "5.75": 0,
        "6": 1,
        "6.25": 2,
        "6.5": 3,
        "6.75": 4,
        "7": 5,
        "7.25": 6,
        "7.5": 7,
        "7.75": 8,
        "8": 9,
        "8.25": 10,
        "8.5": 11,
        "8.75": 12,
        "9": 13,
        "9.25": 14,
        "9.5": 15,
        "9.75": 16,
        "10": 17,
        "10.25": 18,
        "10.5": 19,
        "10.75": 20,
        "11": 21,
        "11.25": 22,
        "11.5": 23,
        "11.75": 24,
        "12": 25,
        "12.25": 26,
        "12.5": 27,
        "12.75": 28,
        "13": 29,
        "13.25": 30,
        "13.5": 31,
        "13.75": 32,
        "14": 33,
        "14.25": 34,
        "14.5": 35,
        "14.75": 36,
        "15": 37,
        "15.25": 38,
        "15.5": 39,
        "15.75": 40,
        "16": 41,
        "16.25": 42,
        "16.5": 43,
        "16.75": 44,
        "17": 45,
        "17.25": 46,
        "17.5": 47,
        "17.75": 48,
        "18": 49,
        "18.25": 50,
        "18.5": 51,
        "18.75": 52,
    }
    mapper_label['in.bedrooms_res'] = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4
    }
    mapper_label['in.cooling_setpoint_res'] = {
        "60F": 0,
        "62F": 1,
        "65F": 2,
        "67F": 3,
        "68F": 4,
        "70F": 5,
        "72F": 6,
        "75F": 7,
        "76F": 8,
        "78F": 9,
        "80F": 10
    }
    mapper_label['in.heating_setpoint_res'] = {
        "55F": 0,
        "60F": 1,
        "62F": 2,
        "65F": 3,
        "67F": 4,
        "68F": 5,
        "70F": 6,
        "72F": 7,
        "75F": 8,
        "76F": 9,
        "78F": 10,
        "80F": 11
    }
    mapper_label['in.geometry_floor_area_res'] = {
        "0-499": 0,
        "500-749": 1,
        "750-999": 2,
        "1000-1499": 3,
        "1500-1999": 4,
        "2000-2499": 5,
        "2500-2999": 6,
        "3000-3999": 7,
        "4000+": 8,
    }
    mapper_label['in.income_res'] = {
        "<10000": 0,
        "10000-14999": 1,
        "15000-19999": 2,
        "20000-24999": 3,
        "25000-29999": 4,
        "30000-34999": 5,
        "35000-39999": 6,
        "40000-44999": 7,
        "45000-49999": 8,
        "50000-59999": 9,
        "60000-69999": 10,
        "70000-79999": 11,
        "80000-99999": 12,
        "100000-119999": 13,
        "120000-139999": 14,
        "140000-159999": 15,
        "160000-179999": 16,
        "180000-199999": 17,
        "200000+": 18,
    }
    
    mapper_label['in.vintage_res'] = {
        "<1940": 0,
        "1940s": 1,
        "1950s": 2,
        "1960s": 3,
        "1970s": 4,
        "1980s": 5,
        "1990s": 6,
        "2000s": 7,
        "2010s": 8,
    }
    mapper_label['in.tstat_clg_sp_f..f_com'] = {
        "69": 0,
        "70": 1,
        "71": 2,
        "72": 3,
        "73": 4,
        "74": 5,
        "75": 6,
        "76": 7,
        "77": 8,
        "79": 9,
        "80": 10,
        "999": 11
    }
    mapper_label['in.tstat_htg_sp_f..f_com'] = {
        "61": 0,
        "63": 1,
        "64": 2,
        "65": 3,
        "66": 4,
        "67": 5,
        "68": 6,
        "69": 7,
        "70": 8,
        "71": 9,
        "72": 10,
        "999": 11
    }
    
    return mapper_label