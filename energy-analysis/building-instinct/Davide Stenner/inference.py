if __name__=='__main__':
    import os
    import argparse    
    import pandas as pd
    import polars as pl
    
    from src.utils.import_utils import import_config, import_params
    from src.preprocess.pipeline import PreprocessPipeline
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lgb', type=str)
    parser.add_argument('--all_model', action='store_true')
    
    args = parser.parse_args()

    config_dict = import_config()    
    
    test_data = pl.read_parquet(
        os.path.join(
            config_dict['PATH_GOLD_PARQUET_DATA'], 
            'test_data.parquet'
        )
    )
    preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
    )
    preprocessor.begin_inference()
    
    if (args.model == 'lgb') | (args.all_model):
        from src.model.lgbm.pipeline import LgbmPipeline
        
        params_model, experiment_name = import_params(model='lgb')
    
        trainer = LgbmPipeline(
            experiment_name=experiment_name + "_lgb",
            params_lgb=params_model,
            config_dict=config_dict, data_columns=preprocessor.feature_list,
            evaluate_shap=False
        )
        trainer.activate_inference()
        
        result_df: pd.DataFrame = trainer.predict(test_data = test_data)
        
        #remap category
        mapper_category = import_config(
            os.path.join(
                config_dict['PATH_MAPPER_DATA'],
                'mapper_category.json'
            )
        )['train_label']
        for type_model, mapper_ in mapper_category.items():
            result_df[type_model] = result_df[type_model].map(
                {
                    value_: key_
                    for key_, value_ in mapper_.items()
                }
            )
        
        #reset prediction
        result_df.loc[
            result_df[config_dict['TARGET_DICT']['BINARY']] == 'commercial',
            config_dict['TARGET_DICT']['RESIDENTIAL']
        ] = pd.NA
        result_df.loc[
            result_df[config_dict['TARGET_DICT']['BINARY']] == 'residential',
            config_dict['TARGET_DICT']['COMMERCIAL']
        ] = pd.NA

        
        (
            result_df
            .sort_values('bldg_id')
            .set_index('bldg_id')
            .to_parquet(
                os.path.join(
                    config_dict['PATH_EXPERIMENT'],
                    experiment_name + "_lgb",
                    'submission.parquet'
                )
            )
        )
    else:
        raise NotImplementedError