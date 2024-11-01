import os
import warnings

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings(action='ignore', category=FutureWarning)

def line_plot(data: pl.DataFrame, col: str, dtype_col: str) -> plt.figure:
    suffix: str = '  ' + dtype_col
            
    plt.figure(figsize=(12,8))
    fig = sns.barplot(
        data=data, 
        x='building_stock_type',  y=col
    )
    plt.title(col + suffix)
    return fig

def save_multiple_line_plot(
        dataset: pl.DataFrame, col_to_eda: list[str], 
        save_path: str, file_name: str
    ) -> None:
    assert isinstance(dataset, pl.DataFrame)
    
    save_path_file = os.path.join(
        save_path, f'{file_name}.pdf'
    )
    
    dataset_selected = dataset.select(col_to_eda)

    dtype_list = [
        str(pl_dtype)
        for pl_dtype in dataset_selected.dtypes
    ]

    mapping_col_dtype = {
        col_to_eda[i]: dtype_list[i]
        for i in range(dataset_selected.width)
    }
    
    data_selected = dataset.group_by('building_stock_type').agg(
        [
            pl.col(col).mean()
            for col in col_to_eda
        ]
    )
    with PdfPages(save_path_file) as export_pdf:
        
        for col in tqdm(col_to_eda):
            fig = line_plot(data_selected, col=col, dtype_col=mapping_col_dtype[col])
            export_pdf.savefig()  # saves the current figure into a pdf page
            plt.close()