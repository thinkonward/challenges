import numpy as np

import polars as pl

from tqdm import tqdm
from typing import Dict
from sklearn.model_selection import StratifiedKFold, KFold
from src.base.preprocess.cv_fold import BaseCVFold
from src.preprocess.initialize import PreprocessInit

def iterative_stratification(
    target_data: np.ndarray, build_id: np.ndarray, n_folds: int
) -> Dict[int, int]:
    
    #only works in dummy (multi label) framework 
    #respect to desired_label_sample_fold sum
    assert ((target_data==0) | (target_data==1)).all()
    
    #http://lpis.csd.auth.gr/publications/sechidis-ecmlpkdd-2011.pdf
    num_rows, num_cols = target_data.shape

    #Calculate the desired number of examples at each subset
    desired_sample_fold: np.ndarray = np.repeat(
        int(num_rows * 1/n_folds), n_folds
    ) 
    
    #Calculate the desired number of examples of each label at each subset
    desired_label_sample_fold: np.ndarray = np.repeat(
        np.floor(target_data.sum(axis=0) * 1/n_folds).reshape((-1, 1)),
        n_folds, axis=1
    )
    
    fold_bucket: list[list[int]] = [[] for _ in range(n_folds)]
    
    bar_ = tqdm(total = target_data.shape[0])
    
    while (target_data.shape[0] > 0):
        #Find the label with the fewest (but at least one) 
        #remaining examples, breaking ties randomly
        remaining_label = target_data.sum(axis=0)
        remaining_label[remaining_label==0] = remaining_label.max()+10
        
        selected_label = np.argmin(remaining_label)
        
        mask_selected_label: np.ndarray = target_data[:, selected_label]==1
        
        label_data_subset = target_data[mask_selected_label]
        #Find the subset(s) with the largest number of desired examples for 
        #this label, breaking ties by considering the largest number of 
        #desired examples, breaking further ties randomly
        max_value = desired_label_sample_fold[label_data_subset, :].max()
        max_subset = np.where(desired_label_sample_fold[label_data_subset, :] == max_value)[0]
        
        if max_subset.shape[0] == 1:
            selected_fold = max_subset
        else:
            #break tie
            max_desired_value = desired_sample_fold.max()
            max_desired_subset = np.where(desired_sample_fold == max_desired_value)[0]
            
            if max_desired_subset.shape[0] == 1:
                selected_fold = max_desired_subset
            else:
                selected_fold = np.random.choice(max_desired_subset, 1)
        
        #update all
        desired_label_sample_fold[label_data_subset, selected_fold] -= 1
        desired_sample_fold[selected_fold] -= 1

        #select one random value of this class
        subset_index = np.arange(target_data[mask_selected_label].shape[0])
        selected_observation_from_subset = np.random.choice(subset_index)
        mask_selected = selected_observation_from_subset == subset_index

        selected_index_observation = build_id[mask_selected_label][mask_selected]

        #update folder
        fold_bucket[selected_fold[0]].append(
            int(selected_index_observation[0])
        )
        element_to_drop = (
            np.where(mask_selected_label)[0][mask_selected]
        )
        build_id = np.delete(
            build_id, element_to_drop, axis=0
        )
        target_data = np.delete(
            target_data, element_to_drop, axis=0
        )

        bar_.update(1)
        
    fold_mapper:Dict[int, int] = {}
    for fold_, bucket in enumerate(fold_bucket):
        fold_mapper.update(
            {
                index_: fold_
                for index_ in bucket
            }
        )

    return fold_mapper

class PreprocessFoldCreator(BaseCVFold, PreprocessInit):   
    def __create_binary_fold(self, selected_target: str, filtered_label_data: pl.LazyFrame) -> pl.LazyFrame:
        
        self.preprocess_logger.info(f'Creating {selected_target} Fold')
        
        splitter_ = StratifiedKFold(self.n_folds, shuffle=True)
        data_binary = (
            filtered_label_data.select(self.build_id, selected_target)
            .collect()
            .to_pandas()
        )

        fold_iterator = enumerate(splitter_.split(data_binary, data_binary[selected_target]))
        fold_mapper: Dict[int, int] = {}
        
        for fold_, (_, test_index) in fold_iterator:
            fold_mapper.update(
                {
                    build_id: fold_
                    for build_id in data_binary.loc[test_index, self.build_id].tolist()
                }
            )
        target_data = self.__create_fold_from_mapper(
            filtered_label_data.select(
                [self.build_id, selected_target] 
            ), fold_mapper
        )
        return target_data

    def __create_random_fold(self, target_col_list: list[str], target: pl.LazyFrame) -> pl.LazyFrame:
        
        self.preprocess_logger.info('Creating Shuffled Fold')
        
        splitter_ = KFold(self.n_folds, shuffle=True)
        
        target_selected_ = (
            target.select(
                [self.build_id] + target_col_list 
            )
        )
        data_binary = (
            target_selected_
            .collect()
            .to_pandas()
        )

        fold_iterator = enumerate(splitter_.split(data_binary))
        fold_mapper: Dict[int, int] = {}
        
        for fold_, (_, test_index) in fold_iterator:
            fold_mapper.update(
                {
                    build_id: fold_
                    for build_id in data_binary.loc[test_index, self.build_id].tolist()
                }
            )
        target_data = self.__create_fold_from_mapper(
            target_selected_, 
            fold_mapper
        )
        return target_data

    def __create_multilabel_fold(self, target_col_list: list[str], target: pl.LazyFrame) -> pl.LazyFrame:

        self.preprocess_logger.info('Creating MultiLabel Fold')

        data_for_split = target.select(target_col_list).collect().to_dummies().to_numpy()        
        build_id = target.select(self.build_id).collect().to_numpy().reshape((-1))
        
        fold_mapper = iterative_stratification(
            target_data=data_for_split, build_id=build_id, n_folds=self.n_folds
        )
        target_data = self.__create_fold_from_mapper(
            target.select(
                [self.build_id] + target_col_list 
            ), 
            fold_mapper
        )
        return target_data
    
    def __create_fold_from_mapper(
            self, 
            data: pl.LazyFrame, fold_mapper: Dict[int, int]
        ) -> pl.LazyFrame:
        data = (
            data
            .clone()
            .with_columns(
                pl.col(self.build_id).replace(fold_mapper).alias('fold').cast(pl.UInt8)
            )
            .with_columns(
                (
                    (
                        pl.when(
                            pl.col('fold') != fold_
                        )
                        .then(pl.lit('t'))
                        .when(
                            pl.col('fold') == fold_
                        ).then(pl.lit('v')).otherwise(pl.lit('n'))
                        .alias(f'fold_{fold_}')
                    )
                    for fold_ in range(self.n_folds)
                )
            )
            .with_columns(
                pl.concat_str(
                    [f'fold_{x}' for x in range(self.n_folds)],
                    separator=', '
                )
                .alias('fold_info')
            )
            .drop(['fold'] + [f'fold_{x}' for x in range(self.n_folds)])
        )
        return data
    
    def create_fold(self) -> None:
        
        self.dict_target = {}
        
        for selected_target in self.all_target_list:
            if selected_target == self.target_col_binary:
                filtered_label_data = self.label_data.clone()
            elif selected_target in self.target_col_com_list:
                filtered_label_data = (
                    self.label_data.clone()
                    .filter(pl.col(self.target_col_binary)==self.commercial_index)
                    .filter(pl.col(selected_target).is_null().not_())
                )
            elif selected_target in self.target_col_res_list:
                filtered_label_data = (
                    self.label_data.clone()
                    .filter(pl.col(self.target_col_binary)!=self.commercial_index)
                    .filter(pl.col(selected_target).is_null().not_())
                )
            else:
                raise ValueError
            
            self.dict_target[f'train_{selected_target}'] = self.__create_binary_fold(
                selected_target=selected_target, 
                filtered_label_data=filtered_label_data
            )