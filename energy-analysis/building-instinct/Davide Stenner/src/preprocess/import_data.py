import os
import gc
import glob
import polars as pl

from src.base.preprocess.import_data import BaseImport
from src.preprocess.initialize import PreprocessInit

class PreprocessImport(BaseImport, PreprocessInit):
    def scan_all_dataset(self):
        stage_file: str = (
            'TEST' if self.inference
            else 'TRAIN'
        )
        self.base_data: pl.LazyFrame = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_SILVER_PARQUET_DATA'],
                self.config_dict[f'{stage_file}_FEATURE_HOUR_FILE_NAME']
            )
        )
        self.economic_data: pl.LazyFrame = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_SILVER_PARQUET_DATA'],
                'macro_economics_data.parquet'
            )
        )
        if not self.inference:
            self.label_data: pl.LazyFrame = pl.scan_parquet(
                os.path.join(
                    self.config_dict['PATH_SILVER_PARQUET_DATA'],
                    self.config_dict['TRAIN_LABEL_FILE_NAME']
                )
            )

            if self.additional_data:
                self.preprocess_logger.info('Using additional dataset')
                self.add_additional_data()
    
    def reset_all_feature_dataset(self) -> None:
        self.base_data = None
        self.data = None
        self.lazy_feature_list = []
        
        _ = gc.collect()
           
    def add_additional_data(self) -> None:
        hour_data_residential = [
            (
                pl.scan_parquet(
                        os.path.join(
                            self.config_dict['PATH_SILVER_PARQUET_DATA'],
                            file_name
                        )
                    )
                .select(self.base_data.collect_schema().names())
            )
            for file_name in glob.glob(f'train_data_residential_additional_*.parquet')
        ]
        hour_data_commercial = [
            (
                pl.scan_parquet(
                        os.path.join(
                            self.config_dict['PATH_SILVER_PARQUET_DATA'],
                            file_name
                        )
                    )
                .select(self.base_data.collect_schema().names())
            )
            for file_name in glob.glob(f'train_data_commercial_additional_*.parquet')
        ]
        label_data = (
            pl.scan_parquet(
                os.path.join(
                    self.config_dict['PATH_SILVER_PARQUET_DATA'],
                    'train_label_additional.parquet'
                )
            ).select(self.label_data.collect_schema().names())
        )
        self.preprocess_logger.info(f'Added {label_data.select(pl.n_unique(self.build_id)).collect().item()} rows')
        
        self.base_data = pl.concat(
            [self.base_data] + 
            hour_data_residential +
            hour_data_commercial
        )
          
        self.label_data = pl.concat(
            [
                self.label_data, 
                label_data
            ]
        )

    def downcast_feature(self) -> None:
        self.base_data = (
            self.base_data.with_columns(
                pl.col('timestamp').cast(pl.Datetime),
                pl.col('out.electricity.total.energy_consumption').cast(pl.Float32),
                pl.col('in.state').cast(pl.UInt8),
                pl.col(self.build_id).cast(pl.UInt32),
            )
            .rename(
                {
                    'out.electricity.total.energy_consumption': 'energy_consumption',
                    'in.state': 'state'
                }
            )
        )
    def downcast_target(self) -> None:
        self.label_data = self.label_data.with_columns(
            [
                pl.col(self.build_id).cast(pl.UInt32)
            ] +
            [
                pl.col(col).cast(pl.UInt8)
                for col in self.all_target_list
            ]
        )
        
    def downcast_data(self) -> None:
        self.downcast_feature()
        
        if not self.inference:
            self.downcast_target()
                
    def import_all(self) -> None:
        self.scan_all_dataset()
        self.downcast_data()