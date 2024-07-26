# visualizations 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import networkx as nx
#from networkx.drawing.nx_agraph import graphviz_layout
from sklearn.cluster import KMeans
from matplotlib.ticker import FormatStrFormatter
from onward.utils import centered_average, ecdf


def plot_size_histogram(total_energy_consumption):
    """
    Plots a histogram of the total energy consumption across buildings.

    Parameters
    ----------
    total_energy_consumption : array_like
        An array of total energy consumption values, one per building.

    Returns
    -------
    None
        Displays a histogram plot.
    """    
    plt.figure(figsize=(9, 3.5))
    plt.hist(total_energy_consumption, 25)
    plt.xlabel("Total Energy Consumption")
    plt.ylabel("Number of Buildings")
    plt.grid()

    plt.tight_layout()
    plt.show()    

    
def plot_size_ecdf(total_energy_consumption):
    """
    Plots the Empirical Cumulative Distribution Function (ECDF) of total energy consumption.

    Parameters
    ----------
    total_energy_consumption : array_like
        An array of total energy consumption values, one per building, sorted in descending order.

    Returns
    -------
    None
        Displays the ECDF plot with marked percentiles.
    """    
    total_energy_consumption = np.sort(total_energy_consumption)[::-1]
    fraction_of_total = 100*np.cumsum(total_energy_consumption / total_energy_consumption.sum())
    P_fraction_of_total = 100*ecdf(fraction_of_total)[1]

    plt.figure(figsize=(9, 3.5))
    plt.plot(np.append(0, P_fraction_of_total), np.append(0, fraction_of_total), "k")
    for i, cutoff in enumerate([0.05, 0.1, 0.25, 0.5]):
        j = np.where(P_fraction_of_total < 100*cutoff)[0][-1]
        x, y = P_fraction_of_total[j], fraction_of_total[j]
        plt.plot([-10, x], [y, y], "--", color="C{0}".format(i), label="({0:0.0f}%, {1:0.0f}%)".format(x, y))
        plt.plot([x, x], [-10, y], "--", color="C{0}".format(i))
        plt.yticks(np.linspace(0, 100, 6), ["{0:0.0f}%".format(x) for x in np.linspace(0, 100, 6)])
        plt.xticks(np.linspace(0, 100, 11), ["{0:0.0f}%".format(x) for x in np.linspace(0, 100, 11)])

    plt.xlabel("Percent of Buildings (ordered by energy consumption)")
    plt.ylabel("Percent of Total Energy Consumption")
    plt.legend(title="(Buildings, Energy)")
    plt.xlim(-1, 101)
    plt.ylim(-1, 105)
    plt.tight_layout()
    plt.show()

    
def plot_time_dependent_components(df):
    """
    Plots time-dependent components and forecasts for energy consumption.

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame with at least 'energy_consumption', 'timestamp', and 'hour_of_week' columns.

    Returns
    -------
    None
        Displays multiple line plots of actual energy consumption, model predictions, and components.
    """
    # calculate forecast
    df["seasonal_component"] = centered_average(df.energy_consumption)
    df["seasonal_adjusted_energy_consumption"] = df.energy_consumption - df.seasonal_component
    weekly_component = df.groupby("hour_of_week")["seasonal_adjusted_energy_consumption"].mean()
    df["forecast"] = weekly_component.values[df.hour_of_week] + df["seasonal_component"]
    
    # plot
    plt.figure(figsize=(9, 4.5))
    plt.subplot(2, 1, 1)
    plt.plot(df.timestamp, df.energy_consumption/1e3, alpha=0.5, label="Actual")
    plt.plot(df.timestamp, df.forecast/1e3, alpha=0.5, label="Model Prediction")
    plt.plot(df.timestamp, df.seasonal_component/1e3, label="Seasonal Component")
    plt.ylabel("Total Energy\nConsumption (MWh)")
    plt.legend()
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.plot(weekly_component/1e3, "k", label="Weekly Component")
    plt.xticks(np.arange(0,169,24), ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"])
    plt.axhline(0, linestyle="--", color="gray", alpha=0.5)
    plt.ylabel("Baseline\nAdjustment (MWh)")
    plt.legend()
    plt.tight_layout()

    plt.show()

    
def plot_seasonal_ecdf(probs, q_winter, q_summer):
    """
    Plots the seasonal ECDF highlighting different probability thresholds for winter and summer.

    Parameters
    ----------
    probs : array_like
        Array of probabilities for classifying a winter condition.
    q_winter : float
        The winter probability threshold.
    q_summer : float
        The summer probability threshold.

    Returns
    -------
    None
        Displays the ECDF plot with areas highlighted for winter and summer.
    """    
    plt.figure(figsize=(9, 3.5))
    plt.plot(*ecdf(probs), "k.")
    xlim = plt.xlim()

    plt.fill_between(xlim, [q_winter, q_winter], [1.0, 1.0], color="C0", alpha=0.5)
    plt.fill_between(xlim, [0.0, 0.0], [q_summer, q_summer], color="C1", alpha=0.5)
    plt.fill_between(xlim, [q_summer, q_summer], [q_winter, q_winter], color="k", alpha=0.15)
    plt.text(0.5, (1+q_winter)/2, "Winter Cluster", va="center", ha="center")
    plt.text(0.5, (q_winter + q_summer)/2, "Winter and Summer", va="bottom", ha="center")
    plt.text(0.5, q_summer/2, "Summer Cluster", va="center", ha="center")
    plt.ylabel("Empirical CDF")
    plt.xlabel("Probability Class 1 (Winter)")
    plt.ylim(0, 1)
    plt.xlim(xlim)

    plt.tight_layout()
    plt.show()


def plot_seasonal_top3(probs, seasonal_features, q_winter, q_summer):
    """
    Plots the top 3 seasonal patterns for peak energy consumption times across different categories.

    Parameters
    ----------
    probs : array_like
        Array of probabilities associated with each pattern.
    seasonal_features : array_like
        Array containing normalized consumption patterns over the year.
    q_winter : float
        Threshold probability for winter peaking.
    q_summer : float
        Threshold probability for summer peaking.

    Returns
    -------
    None
        Displays line plots for the top 3 seasonal consumption patterns.
    """    
    plt.figure(figsize=(9, 6))

    for plot_num, i in enumerate(np.argsort(-probs)[:3]):
        plt.subplot(3, 3, 3*plot_num+1)
        plt.plot(range(14,365-14), seasonal_features[i,14:-14], color="C0")    
        plt.xticks(np.linspace(0, 365, 5), ["Jan", "Apr", "Jul", "Oct", "Jan"])
        plt.ylabel("Normalized\nConsumption")
        plt.gca().yaxis.set_major_formatter("{x:.1f}")
        if plot_num==0: plt.title("Mainly Winter Peak")

    for plot_num, i in enumerate(np.argsort(np.abs(probs - (q_winter + q_summer)/2))[:3]):
        plt.subplot(3, 3, 3*plot_num+2)
        plt.plot(range(14,365-14), seasonal_features[i,14:-14], color="gray")    
        plt.xticks(np.linspace(0, 365, 5), ["Jan", "Apr", "Jul", "Oct", "Jan"])
        if plot_num==0: plt.title("Winter and Summer Peak")

    for plot_num, i in enumerate(np.argsort(probs)[:3]):
        plt.subplot(3, 3, 3*plot_num+3)
        plt.plot(range(14,365-14), seasonal_features[i,14:-14], color="C1")    
        plt.xticks(np.linspace(0, 365, 5), ["Jan", "Apr", "Jul", "Oct", "Jan"])
        if plot_num==0: plt.title("Mainly Summer Peak")

    plt.tight_layout()
    plt.show()

    
def plot_seasonal_cluster_distribution(raw_seasonal_features, seasonal_features, is_winter_peak, is_both_peak, is_summer_peak):
    """
    Plots normalized consumption profile distribution for seasonal cluster categories.

    Parameters
    ----------
    raw_seasonal_features : ndarray
        Array of raw consumption values over a year for multiple buildings.
    seasonal_features : ndarray
        Array of normalized consumption values over a year for multiple buildings.
    is_winter_peak : ndarray of bool
        Boolean array indicating buildings that peak mainly in winter.
    is_both_peak : ndarray of bool
        Boolean array indicating buildings that peak in both winter and summer.
    is_summer_peak : ndarray of bool
        Boolean array indicating buildings that peak mainly in summer.

    Returns
    -------
    None
        Displays a plot of seasonal consumption profiles for each category.
    """
    plt.figure(figsize=(10, 3))
    for plot_num, (x, membership) in enumerate(zip(
        ["Mainly Winter Peak", "Winter and Summer Peak", "Mainly Summer Peak"],
        [is_winter_peak, is_both_peak, is_summer_peak
    ])):
        plt.subplot(1, 3, plot_num+1)
        for seasonal_profile in seasonal_features[membership, 14:-14]:
            plt.plot(range(14, 365-14), seasonal_profile, "C0", alpha=20/np.sum(membership))
        plt.plot(range(14, 365-14), seasonal_features[membership, 14:-14].mean(0), "C1")    
        plt.title("{0}\n{1:0.1f} kWh ({2:0.1f}%)\n{3} Buildings ({4:0.1f}%)".format(
            x,
            raw_seasonal_features[membership].mean(),
            100*raw_seasonal_features[membership].sum()/raw_seasonal_features.sum(),
            np.sum(membership),
            100*np.mean(membership)
        ))
        plt.xticks(np.linspace(0, 365, 5), ["Jan", "Apr", "Jul", "Oct", "Jan"])
        plt.axhline(0, linestyle=":", color="k", alpha=0.5)
        if plot_num==0: plt.ylabel("Normalized Consumption")
        plt.ylim(-1, 2)
    plt.tight_layout()
    plt.show()    


def plot_tod_n_clusters(daily_features):
    """
    Plots cluster profiles for time of day (TOD) based on daily features, using K-means clustering.

    Parameters
    ----------
    daily_features : ndarray
        Array of daily consumption patterns for multiple buildings.

    Returns
    -------
    None
        Displays cluster profiles for 2, 3, and 4 clusters.
    """    
    plt.figure(figsize=(10, 2.5))
    for plot_num, n_clusters in enumerate([2, 3, 4]):

        plt.subplot(1, 3, plot_num+1)    
        cluster_model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        cluster_labels = cluster_model.fit_predict(daily_features)
        for cluster_num, cluster_count in zip(*np.unique(cluster_labels, return_counts=True)):        
            plt.plot(daily_features[cluster_labels == cluster_num].mean(0))

        plt.title("n_clusters={0}".format(n_clusters))
        plt.xticks(np.arange(0, 30, 6), ["12am", "6am", "12pm", "6am", "12am"])
        if plot_num == 0:
            plt.ylabel("Normalized Consumption")

    plt.tight_layout()
    plt.show()
        
    
def plot_tod_cluster_distribution(raw_daily_features, daily_features, tod_cluster_labels):
    """
    Plots the daily time of day (TOD) cluster profile distribution.

    Parameters
    ----------
    daily_features : ndarray
        Array containing daily profiles of energy consumption for multiple buildings.
    tod_cluster_labels : ndarray
        Array of cluster labels for each daily profile, indicating the cluster it belongs to.

    Returns
    -------
    None
        Displays all the profiles in each TOD cluster.
    """    
    plt.figure(figsize=(10, 3))
    for cluster_num, desc in enumerate(["Midday Peak", "Midday Trough", "Evening Peak"]):
        membership = tod_cluster_labels == cluster_num
        plt.subplot(1, 3, cluster_num+1)
        for daily_profile in daily_features[membership]:
            plt.plot(daily_profile, "C0", alpha=20/np.sum(membership))
        plt.plot(daily_features[membership].mean(0), "C1")    
        plt.title("{0}\n{1:0.2f} GWh ({2:0.1f}%)\n{3} Buildings ({4:0.1f}%)".format(
            desc,
            raw_daily_features[membership].sum() / 1e6,
            100*raw_daily_features[membership].sum()/raw_daily_features.sum(),
            np.sum(membership),
            100*np.mean(membership)
        ))
        plt.xticks(np.arange(0, 30, 6), ["12am", "6am", "12pm", "6am", "12am"])
        plt.axhline(0, linestyle=":", color="k", alpha=0.5)
        if cluster_num == 0:
            plt.ylabel("Normalized Consumption")
        plt.ylim(-2.3, 2)
    plt.tight_layout()
    plt.show()


def plot_first_level_energy_fractions(cluster_data):
    """
    Plots the energy fraction of buildings in each first-level group.

    Parameters
    ----------
    cluster_data : DataFrame
        DataFrame containing the cluster groups and total energy consumption for each building.

    Returns
    -------
    None
        Displays a pie chart of the fraction of buildings in each first-level group.
    """
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))



    for i, group_name in enumerate(["small", "large"]):

        df = cluster_data \
            .loc[cluster_data.total_consumption_group == group_name].groupby("time_of_day_consumption_group") \
            .agg(total_energy_consumption=("total_energy_consumption", "sum"), count=("total_energy_consumption", "count")) \
            .reset_index().sort_values("time_of_day_consumption_group")

        ax[i].set_title("{} Building Consumption".format(group_name.title()), y=0.925)

        group_total = df.total_energy_consumption.sum()
        wedges, texts = ax[i].pie(df.total_energy_consumption, labels=[
            "{0}\n * {1:0.2f} GWh\n * {2:0.2f}%".format(name.replace("_", " ").title(), total/1e6, 100*total/group_total)
            for _, (name, total, count) in df.iterrows()
        ])

        for i, text in enumerate(texts):
            if ((group_name == "small") and (i == 0)) or ((group_name == "large") and (i == 1)):
                text.set_position((-0.38, 0.55))
            text.set_horizontalalignment("left")
            text.set_bbox(dict(facecolor="white", boxstyle="round", alpha=1.0, edgecolor="lightgrey"))
            text.set_color("C{0}".format(i))

    plt.tight_layout()
    plt.show()


def plot_first_level_count_fractions(cluster_data):
    """
    Plots the energy fraction of buildings in each first-level group.

    Parameters
    ----------
    cluster_data : DataFrame
        DataFrame containing the cluster groups and total energy consumption for each building.

    Returns
    -------
    None
        Displays a pie chart of the fraction of buildings in each first-level group.
    """
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    for i, group_name in enumerate(["small", "large"]):

        df = cluster_data \
            .loc[cluster_data.total_consumption_group == group_name].groupby("time_of_day_consumption_group") \
            .agg(count=("total_energy_consumption", "count")) \
            .reset_index().sort_values("time_of_day_consumption_group")

        ax[i].set_title("{} Building Count".format(group_name.title()), y=0.925)

        group_count = df["count"].sum()
        wedges, texts = ax[i].pie(df["count"], labels=[
            "{0}\n * {1} Buildings\n * {2:0.2f}% ".format(name.replace("_", " ").title(), count, 100*count/group_count)
            for _, (name, count) in df.iterrows()
        ])

        for i, text in enumerate(texts):
            if ((group_name == "small") and (i == 0)):
                text.set_position((-0.38, 0.55))
            if ((group_name == "small") and (i == 1)):
                text.set_position((1.05, -0.4))
            if ((group_name == "small") and (i == 2)):
                text.set_position((1.1, 0.15))

            if ((group_name == "large") and (i == 1)):
                text.set_position((-0.38, 0.55))
            text.set_horizontalalignment("left")
            text.set_bbox(dict(facecolor="white", boxstyle="round", alpha=1.0, edgecolor="lightgrey"))
            text.set_color("C{0}".format(i))

    plt.tight_layout()
    plt.show()


def plot_group_hierarchy(cluster_data):
    """
    Plots a hierarchical graph of building clusters.

    Parameters
    ----------
    cluster_data : DataFrame
        DataFrame containing the cluster groups and total energy consumption for each building.

    Returns
    -------
    None
        Displays a hierarchical graph of building clusters.
    """
    # root
    edge_list = []
    labels = {"root": "All\n{0:0.2f} GWh\n{1} Buildings".format(
        cluster_data.total_energy_consumption.sum() / 1e6, len(cluster_data))}

    # level 1
    df1 = cluster_data.groupby(["total_consumption_group"]) \
        .agg(total_energy_consumption=("total_energy_consumption", "sum"),
             count=("total_energy_consumption", "count")) \
        .sort_values("total_energy_consumption", ascending=False).reset_index()

    for _, (node_name, total, count) in df1.iterrows():
        edge_list.append(("root", node_name))
        labels[node_name] = "{0}\n{1:0.2f} GWh\n {2} Buildings".format(node_name.title(), total / 1e6, count)

    # level 2
    df2 = cluster_data.groupby(["total_consumption_group", "time_of_day_consumption_group"]) \
        .agg(total_energy_consumption=("total_energy_consumption", "sum"),
             count=("total_energy_consumption", "count")) \
        .sort_values("total_energy_consumption", ascending=False).reset_index()

    for _, (name1, name2, total, count) in df2.iterrows():
        node_name = "{0}_{1}".format(name1, name2)
        edge_list.append((name1, node_name))
        labels[node_name] = "{0}\n{1}\n{2:0.2f} GWh\n{3} Buildings".format(
            name1.title(), name2.replace("_", " ").title(), total / 1e6, count)

    # level 3
    # small builings
    df = cluster_data.loc[
        (cluster_data.total_consumption_group == "small") &
        (cluster_data.time_of_day_consumption_group == "evening_peak") &
        (cluster_data.summer_consumption_group == "summer_peak")]
    edge_list.append(("small_evening_peak", "small_evening_peak_summer"))
    labels["small_evening_peak_summer"] = "Small Summer\nEvening Peak\n{0:0.2f} GWh\n{1} Buildings".format(
        df.total_energy_consumption.sum() / 1e6, len(df))

    df = cluster_data.loc[
        (cluster_data.total_consumption_group == "small") &
        (cluster_data.time_of_day_consumption_group == "evening_peak") &
        (cluster_data.winter_consumption_group == "winter_peak")]
    edge_list.append(("small_evening_peak", "small_evening_peak_winter"))
    labels["small_evening_peak_winter"] = "Small Winter\nEvening Peak \n{0:0.2f} GWh\n{1} Buildings".format(
        df.total_energy_consumption.sum() / 1e6, len(df))

    # large buildings
    df = cluster_data.loc[
        (cluster_data.total_consumption_group == "large") &
        (cluster_data.time_of_day_consumption_group == "midday_peak") &
        (cluster_data.summer_consumption_group == "summer_peak")]
    edge_list.append(("large_midday_peak", "large_midday_peak_summer"))
    labels["large_midday_peak_summer"] = "Large Summer\nMidday Peak\n{0:0.2f} GWh\n{1} Buildings".format(
        df.total_energy_consumption.sum() / 1e6, len(df))

    df = cluster_data.loc[
        (cluster_data.total_consumption_group == "large") &
        (cluster_data.time_of_day_consumption_group == "midday_peak") &
        (cluster_data.winter_consumption_group == "winter_peak")]
    edge_list.append(("large_midday_peak", "large_midday_peak_winter"))
    labels["large_midday_peak_winter"] = "Large Winter\nMidday Peak\n{0:0.2f} GWh\n{1} Buildings".format(
        df.total_energy_consumption.sum() / 1e6, len(df))
    df = cluster_data.loc[
        (cluster_data.total_consumption_group == "large") &
        (cluster_data.time_of_day_consumption_group == "evening_peak") &
        (cluster_data.summer_consumption_group == "summer_peak")]
    edge_list.append(("large_evening_peak", "large_evening_peak_summer"))
    labels["large_evening_peak_summer"] = "Large Summer\nEvening Peak\n{0:0.2f} GWh\n{1} Buildings".format(
        df.total_energy_consumption.sum() / 1e6, len(df))

    df = cluster_data.loc[
        (cluster_data.total_consumption_group == "large") &
        (cluster_data.time_of_day_consumption_group == "evening_peak") &
        (cluster_data.winter_consumption_group == "winter_peak")]
    edge_list.append(("large_evening_peak", "large_evening_peak_winter"))
    labels["large_evening_peak_winter"] = "Large Winter\nEvening Peak\n{0:0.2f} GWh\n{1} Buildings".format(
        df.total_energy_consumption.sum() / 1e6, len(df))

    # plot
    plt.figure(figsize=(9, 7))
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    pos = graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    for name, (x, y) in pos.items():
        if name in {"small", "large"}:
            pos[name] = (1.1 * x, y)
        elif name.count("_") == 2:
            pos[name] = (0.9 * x, y)
        elif name.count("_") == 3:
            pos[name] = (0.7 * x, y)

    nx.draw(G, pos, labels={x: "" for x in labels}, node_color="white", node_size=15000)

    for node, label in labels.items():

        if node in {"large_midday_trough"}:
            bbox = dict(facecolor="C6", alpha=0.25, boxstyle="round,pad=0.3")
        elif node in {"large_midday_peak_summer"}:
            bbox = dict(facecolor="C0", alpha=0.25, boxstyle="round,pad=0.3")
        elif node in {"large_midday_peak_winter"}:
            bbox = dict(facecolor="C1", alpha=0.25, boxstyle="round,pad=0.3")

        elif node in {"large_evening_peak_summer", "small_evening_peak_summer"}:
            bbox = dict(facecolor="C2", alpha=0.25, boxstyle="round,pad=0.3")

        elif node in {"large_evening_peak_winter", "small_evening_peak_winter"}:
            bbox = dict(facecolor="C3", alpha=0.25, boxstyle="round,pad=0.3")
        else:
            bbox = None

        nx.draw_networkx_labels(G, pos, font_size=14 if node == "root" else 12,
                                labels={node: label}, font_color="k", bbox=bbox)

    plt.text(144.694, 220, "Size", ha="center", fontsize=14, fontweight="bold")
    plt.text(281.817, 320, "Time-of-Day", ha="center", fontsize=14, fontweight="bold")
    plt.text(423.192, 375, "Summer/Winter", ha="center", fontsize=14, fontweight="bold")
    plt.show()


def plot_final_clusters(final_cluster_info, raw_seasonal_features, raw_daily_features, seasonal_features, daily_features):
    """
    Plots the seasonal and daily consumption profiles for each final cluster.
    
    Parameters
    ----------
    final_cluster_info : DataFrame
        DataFrame containing the list of buildings in the final clusters.
    raw_seasonal_features : ndarray
        Array of raw seasonal consumption profiles for each building.
    raw_daily_features : ndarray
        Array of raw daily consumption profiles for each building.        
    seasonal_features : ndarray
        Array of normalized seasonal consumption profiles for each building.
    daily_features : ndarray
        Array of normalized daily consumption profiles for each building.

    Returns
    -------
    None
        Displays line plots for the seasonal and daily consumption profiles of each final cluster.
    """

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(9, 16))

    colors = ["C0", "C1", "C6", "C2", "C3"]
    for row_num, row in final_cluster_info.iterrows():

        ax = axes[row_num]
        color = colors[row_num]
        indices = np.array(row["ids"]) - 1
        range(14, 365 - 14)

        # # raw seasonal
        # ax[0].plot(range(14, 365 - 14), raw_seasonal_features[indices, 14:-14].mean(0), color=color)
        # ax[0].set_xticks(np.linspace(0, 365, 4), ["Jan", "May", "Sep", "Jan"])
        # ax[0].set_ylabel(row["name"].replace(" Evening", "\nEvening").replace(" Midday ", "\nMidday"))
        #
        # # raw daily
        # ax[1].plot(raw_daily_features[indices].mean(0)[np.arange(25) % 24], color=color)
        # ax[1].set_xticks([0, 8, 16, 24], ["12am", "8am", "4pm", "12am"])
        # ax[1].set_xlim(0, 24)

        # normalized seasonal
        for profile in seasonal_features[indices, 14:-14]:
            ax[0].plot(range(14, 365 - 14), profile, color=color, alpha=15 / len(indices))
        ax[0].plot(range(14, 365 - 14), seasonal_features[indices, 14:-14].mean(0), color="k")
        ax[0].set_xticks(np.linspace(0, 365, 7), ["Jan", "Mar", "May", "Jul", "Sep", "Nov", "Jan"])
        ax[0].set_ylabel(row["name"].replace(" Evening", "\nEvening").replace(" Midday ", "\nMidday"))

        # normalized daily
        for profile in daily_features[indices]:
            ax[1].plot(profile[np.arange(25) % 24], color=color, alpha=15 / len(indices))
        ax[1].plot(daily_features[indices].mean(0)[np.arange(25) % 24], color="k")
        ax[1].set_xticks([0, 4, 8, 12, 16, 20, 24], ["12am", "4am", "8am", "12pm", "4pm", "8pm", "12am"])
        ax[1].set_xlim(0, 24)

        # y_lim = (
        #     min(ax[0].get_ylim()[0], ax[1].get_ylim()[0]),
        #     max(ax[0].get_ylim()[1], ax[1].get_ylim()[1])
        # )
        # ax[0].set_ylim(y_lim)
        # ax[1].set_ylim(y_lim)

        # title
        if row_num == 0:
            # ax[0].set_title("Average Raw\nSeasonal Profile")
            # ax[1].set_title("Average Raw\nDaily Profile")
            ax[0].set_title("Seasonal Profiles")
            ax[1].set_title("Daily Profiles")

    plt.tight_layout()
    plt.show()