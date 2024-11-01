if __name__=='__main__':
    import os
    import argparse
    import polars as pl
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', type=str)
    parser.add_argument('--type_building', type=str)
    
    args = parser.parse_args()
    
    folder_to_state = os.path.join(
        'data_dump', args.type_building, 'state=' + args.state
    )            
    
    file_path_list = os.listdir(folder_to_state)
        
    state_df_list: list[pl.LazyFrame] = []
            
    for file_path in file_path_list:

        state_df_list.append(
            pl.scan_parquet(
                os.path.join(
                    folder_to_state, file_path
                )
            )
            .select(
                'timestamp', 'out.electricity.total.energy_consumption', 'bldg_id'
            )
            .group_by(
                'bldg_id', 
                (
                    pl.col('timestamp').cast(pl.Datetime)
                    .dt.offset_by('-15m').dt.truncate('1h')
                )
            )
            .agg(
                pl.col('out.electricity.total.energy_consumption').sum(),
                pl.lit(args.state).cast(pl.Utf8).alias('in.state'),
                pl.lit(args.type_building).cast(pl.Utf8).alias('build_type')
            )
            .select(
                ['timestamp', 'out.electricity.total.energy_consumption', 'in.state', 'bldg_id']
            )
            .collect()
        )
        
        os.remove(
            os.path.join(
                folder_to_state, file_path
            )
        )
    
    state_df: pl.DataFrame = pl.concat(state_df_list)
    
    #save state file
    state_df.write_parquet(
        os.path.join(
            folder_to_state, 'data.parquet'    
        )
    )