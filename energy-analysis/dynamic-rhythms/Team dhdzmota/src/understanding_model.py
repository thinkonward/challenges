import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shap
import warnings

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.utils import resample

import src.utils as utils
import src.training_model as training_model

from src.model_metrics import (
    compute_brierscoreloss,
    compute_aucroc,
    compute_aucpr,
    bootstrap_func,
)

warnings.filterwarnings('ignore')

GENERAL_PATH = utils.get_general_path()
FINAL_DATA_PATH = utils.get_data_path('final')
TEMP_RESULTS_DATA_PATH = utils.get_data_path('temp_results')
GROUPED_INFO_METRICS_PATH = utils.join_paths(TEMP_RESULTS_DATA_PATH, 'grouped_info_metrics.parquet')
GENERAL_METRICS_PATH = utils.join_paths(TEMP_RESULTS_DATA_PATH, 'general_metrics.parquet')

RANDOM_SEED = utils.RANDOM_SEED
SET_COLORS_DICT = utils.SET_COLOR_DICT
SHAP_COLOR_PALLETE = utils.SHAP_COLOR_PALLETE

# Matplotlib configurations
main_color = '#41596a'
mpl.rcParams['text.color'] = main_color
mpl.rcParams['axes.labelcolor'] = main_color
mpl.rcParams['axes.edgecolor'] = main_color
mpl.rcParams['axes.edgecolor'] = main_color
mpl.rcParams['xtick.color'] = main_color
mpl.rcParams['xtick.labelcolor'] = main_color
mpl.rcParams['ytick.color'] = main_color
mpl.rcParams['ytick.labelcolor'] = main_color

DIFFERENCE_THRESHOLD_QUANTILE = 0.96
WRITTEN_FEATURES_NB = 6

def shap_top_N_features(x, shap_explainer, N=10):
    """
    Identify the top N most important features based on SHAP values.

    :param x: pandas DataFrame
        Feature dataset for which SHAP values will be computed.
    :param shap_explainer: shap.Explainer
        A fitted SHAP explainer object.
    :param N: int, optional
        Number of top features to return. Default is 10.
    :return: list
        List of top N feature names ranked by total absolute SHAP value.
    """
    shap_values_x_df = get_shap_values_df(x, shap_explainer)
    importance = shap_values_x_df.abs().sum().sort_values(ascending=False).head(N)
    return importance.index.to_list()


def get_shap_values_df(x, shap_explainer):
    """
    Compute SHAP values for the input features and return them as a DataFrame.

    :param x: pandas DataFrame
        Feature dataset for which SHAP values will be computed.
    :param shap_explainer: shap.Explainer
        A fitted SHAP explainer object.
    :return: shap_values_x_df: pandas DataFrame
        DataFrame containing SHAP values with the same shape and index as the input features.
    """
    shap_values_x = shap_explainer.shap_values(x)
    shap_values_x_df = pd.DataFrame(shap_values_x, columns=x.columns, index=x.index)
    return shap_values_x_df


def get_final_datasets():
    """
    Load final feature and metadata datasets from disk.

    This function reads all Parquet files in the FINAL_DATA_PATH directory and separates them
    into feature sets (`x`) and metadata/info sets (`info`) based on filename prefixes.

    :return: x, info: tuple of dicts
        x: dict
            Dictionary containing feature DataFrames keyed by dataset name (e.g., 'train', 'test').
        info: dict
            Dictionary containing corresponding metadata/info DataFrames.
    """
    x = {}
    info = {}
    for file in os.listdir(FINAL_DATA_PATH):
        if not file.startswith('.'):
            start_letter, filename_string = file.split('_')
            dataset_name = filename_string.split('.')[0]
            path = utils.join_paths(FINAL_DATA_PATH, file)
            if start_letter=='i':
                info[dataset_name] = pd.read_parquet(path)
            elif start_letter=='x':
                x[dataset_name] = pd.read_parquet(path)
    return x, info


def generate_grouped_info_metrics(info, months=6, save=True):
    """
    Compute grouped performance metrics (AUCPR, AUCROC, Brier score) over time intervals and optionally save the result.

    If the grouped metrics file already exists, it is loaded from disk. Otherwise, metrics are computed by grouping
    each dataset in `info` by date and applying bootstrapped scoring functions.

    :param info: dict
        Dictionary of DataFrames, each containing model predictions (`pred`) and true labels (`y`)
        along with a datetime column (`meteorological_current_datetime_val`) for temporal grouping.
    :param months: int, optional
        Size of the time window (in months) for grouping. Default is 6.
    :param save: bool, optional
        Whether to save the computed metrics to disk. Default is True.
    :return: grouped_info_metrics: dict
        Nested dictionary containing grouped metrics (AUCPR, AUCROC, Brier score) for each dataset key.
    """
    if utils.check_if_filepath_exists(GROUPED_INFO_METRICS_PATH):
        print(f'Loading file from {GROUPED_INFO_METRICS_PATH}')
        grouped_info_metrics = utils.read_pickle(GROUPED_INFO_METRICS_PATH)
        return grouped_info_metrics
    date_col = 'meteorological_current_datetime_val'
    grouper = pd.Grouper(key=date_col, freq=f'{months}M')
    grouped_info_metrics = {}
    grouped_info_metrics['aucpr'] = {}
    grouped_info_metrics['aucroc'] = {}
    grouped_info_metrics['brier'] = {}
    for key in info.keys():
        grouped_info_metrics['aucpr'][key] = info[key].groupby(
            grouper
        ).apply(
            bootstrap_func, function=compute_aucpr
        )
        grouped_info_metrics['aucroc'][key] = info[key].groupby(
            grouper
        ).apply(
            bootstrap_func, function=compute_aucroc
        )
        grouped_info_metrics['brier'][key] = info[key].groupby(
            grouper
        ).apply(
            bootstrap_func, function=compute_brierscoreloss
        )
    if save:
        print(f'Saving info at {GROUPED_INFO_METRICS_PATH}')
        utils.save_as_pickle(what=grouped_info_metrics, where=GROUPED_INFO_METRICS_PATH)
    return grouped_info_metrics


def plotting__brier_loss_score(grouped_info_metrics, sets=['train', 'test', 'OOT']):
    """
    Plot the Brier loss score over time for multiple dataset splits using error bars.

    :param grouped_info_metrics: dict
        Dictionary containing bootstrapped Brier loss metrics grouped by time intervals for each dataset split.
    :param sets: list of str, optional
        List of dataset splits to include in the plot. Default is ['train', 'test', 'OOT'].
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in sets:
        yerr = (
            grouped_info_metrics['brier'][key]['err_lo'],
            grouped_info_metrics['brier'][key]['err_hi']
        )
        ax.errorbar(
            grouped_info_metrics['brier'][key].index,
            grouped_info_metrics['brier'][key]['mean'],
            yerr=yerr,
            capsize=5,
            marker='o',
            alpha=0.8,
            label=key,
            color=SET_COLORS_DICT[key]
        )
    date_min_max = [
        grouped_info_metrics['brier']['train'].index.min(),
        grouped_info_metrics['brier']['OOT'].index.max()
    ]
    ax.plot(
        date_min_max,
        [0, 0],
        label='Max. score value',
        linestyle='--',
        color=main_color,
        alpha=0.5
    )
    plt.legend()
    plt.title('Brier loss over time for different datasets.')
    plt.xlabel('Dates')
    plt.ylabel('Brier loss score')
    plt.show()


def plotting__auc_roc(grouped_info_metrics, sets=['train', 'test', 'OOT'], baseline=0.5):
    """
    Plot the AUC ROC scores over time for different dataset splits using error bars.

    :param grouped_info_metrics: dict
        Dictionary containing bootstrapped AUC ROC metrics grouped by time intervals for each dataset split.
    :param sets: list of str, optional
        List of dataset splits to include in the plot. Default is ['train', 'test', 'OOT'].
    :param baseline: float, optional
        Baseline score to display as a reference line. Default is 0.5.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in sets:
        yerr = (
            grouped_info_metrics['aucroc'][key]['err_lo'],
            grouped_info_metrics['aucroc'][key]['err_hi']
        )
        ax.errorbar(
            grouped_info_metrics['aucroc'][key].index,
            grouped_info_metrics['aucroc'][key]['mean'],
            yerr=yerr,
            capsize=5,
            marker='o',
            alpha=0.8,
            label=key,
            color=SET_COLORS_DICT[key]
        )
    date_min_max = [
        grouped_info_metrics['aucroc']['train'].index.min(),
        grouped_info_metrics['aucroc']['OOT'].index.max()
    ]
    ax.plot(
        date_min_max,
        [1, 1],
        label='Max. score value',
        linestyle='--',
        color=main_color,
        alpha=0.5

    )

    ax.plot(
        date_min_max,
        [baseline, baseline],
        label='Baseline',
        linestyle='-.',
        color=main_color,
        alpha=0.5
    )

    plt.ylim(0.45, 1.05)
    plt.legend()
    plt.title('AUC ROC over time for different datasets.')
    plt.xlabel('Dates')
    plt.ylabel('AUC ROC')
    plt.show()


def plotting_score_hours_before_outage(info, sets=['train', 'OOT'], elements=150):
    """
    Plot the average model score as a function of hours before an outage, for different dataset splits.

    :param info: dict
        Dictionary of DataFrames containing the `pred` (model score) and `hours_to_outage` columns.
    :param sets: list of str, optional
        Dataset splits to include in the plot. Default is ['train', 'OOT'].
    :param elements: int, optional
        Number of earliest time bins (hours before outage) to include. Default is 150.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10,5))
    for key in sets:
        hours_to_outage_scores = info[key].groupby('hours_to_outage').agg(
            pred_mean=('pred', 'mean')
        ).sort_index().head(elements)
        ax.scatter(
            hours_to_outage_scores.index,
            hours_to_outage_scores.pred_mean,
            c=SET_COLORS_DICT[key],
            alpha=0.5,
            label=key)
        ax.plot(hours_to_outage_scores.pred_mean, color=SET_COLORS_DICT[key])
    plt.xlabel('Hours before an outage (generated by a storm)')
    plt.ylabel('Average model score')
    plt.title('Score mean value depending on the hours left so that an outage occurs.')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


def plotting__auc_pr(grouped_info_metrics, sets=['train', 'test', 'OOT'], baseline=None):
    """
    Plot the AUC PR (Precision-Recall) scores over time for different dataset splits using error bars.

    :param grouped_info_metrics: dict
        Dictionary containing bootstrapped AUC PR metrics grouped by time intervals for each dataset split.
    :param sets: list of str, optional
        List of dataset splits to include in the plot. Default is ['train', 'test', 'OOT'].
    :param baseline: float or None, optional
        Baseline score to display as a reference line. If None, no baseline is plotted. Default is None.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in sets:
        yerr = (
            grouped_info_metrics['aucpr'][key]['err_lo'],
            grouped_info_metrics['aucpr'][key]['err_hi']
        )
        ax.errorbar(
            grouped_info_metrics['aucpr'][key].index,
            grouped_info_metrics['aucpr'][key]['mean'],
            yerr=yerr,
            capsize=5,
            marker='o',
            alpha=0.8,
            label=key,
            color=SET_COLORS_DICT[key]
        )
    date_min_max = [
        grouped_info_metrics['aucpr']['train'].index.min(),
        grouped_info_metrics['aucpr']['OOT'].index.max()
    ]

    ax.plot(
        date_min_max,
        [1, 1],
        label='Max. score value',
        linestyle='--',
        color=main_color,
        alpha=0.5

    )

    ax.plot(
        date_min_max,
        [baseline, baseline],
        label='Baseline',
        linestyle='-.',
        color=main_color,
        alpha=0.5
    )

    plt.ylim(0, 1.05)
    plt.legend()
    plt.title('AUC PR over time for different datasets.')
    plt.xlabel('Dates')
    plt.ylabel('AUC PR')
    plt.show()


def generate_general_metrics(info, save=True):
    """
    Compute general performance metrics (AUC ROC, AUC PR, Brier Score) across dataset splits.

    If metrics have already been computed and saved, they are loaded from disk.
    Otherwise, bootstrapped estimates are computed and optionally saved.

    :param info: dict
        Dictionary of DataFrames, each containing model predictions (`pred`) and true labels (`y`).
    :param save: bool, optional
        Whether to save the computed metrics to disk. Default is True.
    :return: metrics: dict
        Nested dictionary containing bootstrapped metrics ('aucroc', 'aucpr', 'brier') for each dataset key.
    """
    if utils.check_if_filepath_exists(GENERAL_METRICS_PATH):
        print(f'Loading file from {GENERAL_METRICS_PATH}')
        metrics = utils.read_pickle(GENERAL_METRICS_PATH)
        return metrics
    metrics = {}
    metrics['aucroc'] = {}
    metrics['aucpr'] = {}
    metrics['brier'] = {}
    for key in info.keys():
        metrics['aucroc'][key] = bootstrap_func(info[key], compute_aucroc)
        metrics['aucpr'][key] = bootstrap_func(info[key], compute_aucpr)
        metrics['brier'][key] = bootstrap_func(info[key], compute_brierscoreloss)
    if save:
        print(f'Saving info at {GENERAL_METRICS_PATH}')
        utils.save_as_pickle(what=metrics, where=GENERAL_METRICS_PATH)
    return metrics


def plot_general_metrics(metrics_dict, metric, baseline=None):
    """
    Plot a bar chart with error bars for a given evaluation metric across dataset splits.

    :param metrics_dict: dict
        Dictionary containing bootstrapped metrics (e.g., 'aucroc', 'aucpr', 'brier') for each dataset split.
    :param metric: str
        The metric to plot (must be a key in `metrics_dict`).
    :param baseline: float, optional
        Optional baseline value to plot as a reference line. Default is None.
    :return: metrics_defined_df: pandas DataFrame
        DataFrame of the metric values and error bars used in the plot.
    """
    metrics_defined = metrics_dict[metric]
    metrics_defined_df = pd.DataFrame(metrics_defined).T
    metrics_defined_df['color'] = pd.Series(SET_COLORS_DICT)
    plt.bar(
        metrics_defined_df.index,
        metrics_defined_df['mean'],
        yerr=(metrics_defined_df['err_lo'], metrics_defined_df['err_hi']),
        color=metrics_defined_df.color
    )
    for i, row in enumerate(metrics_defined_df.iterrows()):
        set_name, traits = row
        plt.text(i, traits['mean'] + 0.08*traits['mean'], s=round(traits['mean'], 3), ha='center', va='bottom')
    metrics_defined_df_max_mean = (metrics_defined_df['mean'] + 0.2*metrics_defined_df['mean']).max()
    if baseline is not None:
        metrics_defined_df['baseline'] = baseline
        plt.plot(metrics_defined_df.index, metrics_defined_df['baseline'], color='k', linestyle='--', label='Baseline')
        plt.legend()
    plt.ylim(0, metrics_defined_df_max_mean)
    plt.ylabel(f'{metric}')
    plt.xlabel(f'datasets')
    plt.title(f'General {metric.upper()} for each dataset ')
    plt.show()
    return metrics_defined_df


def get_sample(info, x, sets=['OOT']):
    """
    Select a representative sample from each dataset split based on prediction change dynamics.

    This function:
      - Aggregates prediction metrics at the `episode_fips_id` level.
      - Computes the difference between last and first predictions for each episode.
      - Selects one sample with a high prediction shift (above a quantile threshold).
      - Returns the corresponding info and feature data for that sample.

    :param info: dict
        Dictionary containing metadata/info DataFrames with 'pred' and 'episode_fips_id' columns.
    :param x: dict
        Dictionary containing feature DataFrames indexed identically to the corresponding info DataFrames.
    :param sets: list of str, optional
        Dataset splits to include in sampling. Default is ['OOT'].
    :return: samples: dict
        Dictionary with selected sample information and features for each dataset split.
        Format: samples[set]['info'], samples[set]['x']
    """
    samples = {}
    for key in sets:
        samples[key] = {}
        info_key = info[key]
        episode_fips_metrics = info_key.groupby('episode_fips_id').agg(
            pred_first=('pred', 'first'),
            pred_last=('pred', 'last'),
            pred_min=('pred', 'min'),
            pred_max=('pred', 'max'),
            element_nb=('time', 'count')
        )
        episode_fips_metrics['difference__last_first'] = (
                episode_fips_metrics['pred_last'] -
                episode_fips_metrics['pred_first']
        )
        quantile = episode_fips_metrics.difference__last_first.quantile(
            DIFFERENCE_THRESHOLD_QUANTILE
        )
        sample_index = episode_fips_metrics[
            episode_fips_metrics.difference__last_first > quantile
        ].sample(
            1, random_state=RANDOM_SEED
        ).index.to_list()
        sample_info_key = info_key[info_key.episode_fips_id.isin(sample_index)]
        sample_x_key = x[key].loc[sample_info_key.index]

        samples[key]['info'] = sample_info_key
        samples[key]['x'] = sample_x_key
        return samples


def get_model_explainer():
    """
    Load the trained model and generate a SHAP TreeExplainer for it.

    :return: model, explainer: tuple
        model: Trained model object.
        explainer: SHAP TreeExplainer object for the model.
    """
    model = training_model.get_model()
    explainer = shap.TreeExplainer(model)
    return model, explainer


def plot_shap_importance_for_each_sample(samples):
    """
    Generate visualizations of SHAP feature importance over time and decision contributions for selected samples.

    For each dataset split in `samples`, this function:
      - Computes SHAP values and identifies the top contributing features.
      - Normalizes and visualizes feature importance over time via a stacked area plot.
      - Annotates top features at the start and end of the time window.
      - Overlays model score predictions.
      - Displays a SHAP decision plot showing cumulative feature contributions.

    :param samples: dict
        Dictionary containing sampled data for each split, with keys:
            - samples[split]['x']: feature DataFrame
            - samples[split]['info']: metadata/info DataFrame with 'pred', 'y', and datetime values
    :return: None
    """
    model, explainer = get_model_explainer()
    expected_value = explainer.expected_value

    for key in samples.keys():
        sample_x_key = samples[key]['x']
        sample_info_key = samples[key]['info']

        shap_values_sample_x_key_df = get_shap_values_df(
            x=sample_x_key,
            shap_explainer=explainer
        )
        shap_values_sample_x_key_df_abs = shap_values_sample_x_key_df.abs()
        important_features = shap_top_N_features(
            x=sample_x_key,
            shap_explainer=explainer,
            N=500
        )
        all_other_importances = shap_values_sample_x_key_df_abs.drop(
            important_features, axis=1
        ).sum(axis=1)
        shap_values_sample_x_key_df_abs_relevant = shap_values_sample_x_key_df_abs[
            important_features
        ]
        shap_values_sample_x_key_df_abs_relevant['all_other_features'] = all_other_importances
        total_sum = shap_values_sample_x_key_df_abs_relevant.T.sum()
        relevant_cols = shap_values_sample_x_key_df_abs_relevant.columns
        for col in relevant_cols:
            shap_values_sample_x_key_df_abs_relevant[col] = (
                    shap_values_sample_x_key_df_abs_relevant[col] / total_sum
            )

        last_record = utils.get_record_from_df(df=shap_values_sample_x_key_df_abs_relevant, pos=-1)
        shap_values_sample_x_key_df_abs_relevant_top5_index = last_record.sort_values(ascending=False).iloc[: WRITTEN_FEATURES_NB].index
        positions_top5_text = (last_record.cumsum() + last_record.shift(1).fillna(0).cumsum()) / 2
        top5_text = positions_top5_text.loc[shap_values_sample_x_key_df_abs_relevant_top5_index]

        first_record = utils.get_record_from_df(df=shap_values_sample_x_key_df_abs_relevant, pos=0)
        shap_values_sample_x_key_df_abs_relevant_low5_index = first_record.sort_values(ascending=False).iloc[: WRITTEN_FEATURES_NB].index
        positions_low5_text = (first_record.cumsum() + first_record.shift(1).fillna(0).cumsum()) / 2
        low5_text = positions_low5_text.loc[shap_values_sample_x_key_df_abs_relevant_low5_index]

        all_records_info = [
            shap_values_sample_x_key_df_abs_relevant[col]
            for col in shap_values_sample_x_key_df_abs_relevant.columns
        ]
        fig, ax = plt.subplots(figsize=(20,10))
        # Plot Stack, which is already normalized. These are normalized feature importances.
        ax.stackplot(
            sample_info_key.meteorological_current_datetime_val,
            all_records_info,
            colors=SHAP_COLOR_PALLETE,
            alpha=0.7,
        )
        pos_x_top = sample_info_key.meteorological_current_datetime_val.max()
        pos_x_low = sample_info_key.meteorological_current_datetime_val.min()
        tot_seconds = (pos_x_top - pos_x_low).total_seconds()
        additional_space = tot_seconds * 0.5
        additional_pos_x_top = pos_x_top + pd.Timedelta(f"{additional_space} seconds")
        additional_pos_x_low = pos_x_low - pd.Timedelta(f"{additional_space} seconds")

        # Plot text
        for element_top, element_low in zip(top5_text.index, low5_text.index):
            val_top = round(sample_x_key.iloc[-1][element_top], 2)
            pos_y = top5_text.loc[element_top]
            text = element_top
            ax.text(pos_x_top, pos_y, f' <-- {text}: ({val_top})', ha='left', va='center')
            val_low = round(sample_x_key.iloc[0][element_low], 2)
            pos_y = low5_text.loc[element_low]
            text = element_low
            ax.text(pos_x_low, pos_y, f'{text}: ({val_low}) --> ', ha='right', va='center')
        ax.plot(
            sample_info_key.meteorological_current_datetime_val,
            sample_info_key.pred,
            marker='o',
            label='Model score',
            color='k'
        )
        ax.vlines(
            sample_info_key.meteorological_current_datetime_val,
            0,
            1,
            color='k',
            linestyle='--',
            label='Temporal markers',
            alpha=0.7
        )
        plt.xlim(additional_pos_x_low, additional_pos_x_top)
        plt.legend()
        plt.title(
            'Behaviour of the model score over time for a specific storm.'
            ' With a representation of normalized feature importance changing over time.'
        )
        plt.xlabel('Date')
        plt.ylabel('Model Score')
        plt.show()

        shap.decision_plot(
            expected_value,
            shap_values_sample_x_key_df.to_numpy(),
            sample_x_key,
            link='logit',
            feature_order='importance',
            feature_display_range=slice(-1, -20, -1),
            highlight=sample_info_key['y'].astype('bool'),
        )
        plt.show()


def shap_importance_for_each_sample(x, info):
    """
    Generate SHAP importance plots for selected samples from the 'OOT' and 'test' datasets.

    This function:
      - Selects one representative sample from each dataset using `get_sample`.
      - Plots SHAP feature importance over time and the SHAP decision plot using `plot_shap_importance_for_each_sample`.

    :param x: dict
        Dictionary containing feature DataFrames for each dataset split.
    :param info: dict
        Dictionary containing metadata/info DataFrames including predictions and datetime values.
    :return: None
    """
    samples = get_sample(info, x, sets=['OOT','test'])
    plot_shap_importance_for_each_sample(samples)


def general_shap_summary(x, sets=['OOT', 'test']):
    """
    Generate SHAP summary plots for the specified dataset splits.

    This function:
      - Loads the trained model and creates a SHAP TreeExplainer.
      - Computes SHAP values for each specified dataset.
      - Plots summary plots showing the distribution and magnitude of SHAP values across features.

    :param x: dict
        Dictionary containing feature DataFrames for each dataset split.
    :param sets: list of str, optional
        List of dataset split names to visualize. Default is ['OOT', 'test'].
    :return: None
    """
    model = training_model.get_model()
    explainer = shap.TreeExplainer(model)
    for key in sets:
        x_key = x[key]
        shap_values = explainer.shap_values(x_key)
        plt.title(f'Summary Shap Values for {key} set')
        shap.summary_plot(shap_values, x_key)


def plot_shap_dependence_plots(x, sets=['OOT', 'test'], nb_important_features=5):
    """
    Generate SHAP dependence plots for the top features in the specified dataset splits.

    This function:
      - Loads the trained model and computes SHAP values for each dataset.
      - Identifies the top N most important features by total SHAP magnitude.
      - Plots SHAP dependence plots for those features, showing interaction with feature values.

    :param x: dict
        Dictionary containing feature DataFrames for each dataset split.
    :param sets: list of str, optional
        List of dataset split names to generate plots for. Default is ['OOT', 'test'].
    :param nb_important_features: int, optional
        Number of top features to plot. Default is 5.
    :return: None
    """
    model = training_model.get_model()
    explainer = shap.TreeExplainer(model)
    for key in sets:
        x_key = x[key]
        shap_values = explainer.shap_values(x_key)
        top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))
        for i in range(nb_important_features):
            shap.dependence_plot(top_inds[i], shap_values, x_key)


def plot_score_distribution(info, sets=['OOT']):
    """
    Plot the distribution of model scores for positive and negative classes in specified dataset splits.

    This function:
      - Creates overlaid histograms of predicted scores (`pred`) for both classes (`y` == 0 and `y` == 1).
      - Uses dual y-axes to represent each class's count independently for clearer comparison.

    :param info: dict
        Dictionary containing DataFrames with 'pred' (model score) and 'y' (true label) columns.
    :param sets: list of str, optional
        List of dataset split names to generate plots for. Default is ['OOT'].
    :return: None
    """
    for key in sets:
        info_key = info[key]
        fig, ax = plt.subplots(figsize=(20,10))
        alpha_val = 0.4
        ax.hist(info_key['pred'][info_key['y']==0], bins=100, color='blue', alpha=alpha_val)
        ax2 = ax.twinx()
        ax2.hist(info_key['pred'][info_key['y']==1], bins=100, color='orange', alpha=alpha_val)
        plt.title(f'Score distribution for the positive class on the {key} set.')
        plt.xlabel('Model score (bins)')
        ax.set_ylabel('Count (negative class)')
        ax2.set_ylabel('Count (positive class)')


def plot_general_score_distribution_w_qs(info, sets=['OOT'], q_min=0, q_max=0):
    """
    Plot the score distribution with highlighted uncertainty area based on quantile thresholds.

    This function:
      - Plots the density-normalized histogram of model scores (`pred`) for each dataset split.
      - Marks vertical lines at `q_min` and `q_max` to define the uncertainty zone.
      - Shades the area between these quantiles to visually represent model uncertainty.

    :param info: dict
        Dictionary containing DataFrames with a 'pred' column representing model scores.
    :param sets: list of str, optional
        List of dataset splits to include in the plot. Default is ['OOT'].
    :param q_min: float, optional
        Lower quantile threshold for uncertainty region. Default is 0.
    :param q_max: float, optional
        Upper quantile threshold for uncertainty region. Default is 0.
    :return: None
    """
    for key in sets:
        info_key = info[key]
        fig, ax = plt.subplots(figsize=(20,10))
        alpha_val = 0.2
        y_hist, x_hist, plot_hist = ax.hist(info_key['pred'], bins=1000, density=True)
        ax.vlines([q_min, q_max], 0, y_hist.max(), color='k', linestyle='--')
        plt.fill_between(
            [q_min, q_max],
            y_hist.max(),
            alpha=alpha_val,
            color='r',
            hatch='//',
            label='Uncertainty area'
        )
        plt.title('Score distribution with the corresponding Uncertainty Area.')
        plt.ylabel('Histogram Density')
        plt.xlabel('Model score')
        plt.legend()
        plt.show()
