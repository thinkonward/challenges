import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as jaccard_score
from matplotlib.ticker import MultipleLocator

def plot_report(history: dict, x_locator_tick: int = 10):
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax1.plot(
        [*history.keys()],
        [x.get("loss") for x in history.values()],
        color="#6ab3a2",
        lw=2,
    )
    ax2.plot(
        [*history.keys()],
        [x.get("base_lr") for x in history.values()],
        color="#3399e6",
        lw=1,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#6ab3a2", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="#6ab3a2")

    ax2.set_ylabel("Base learning rate", color="#3399e6", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="#3399e6")
    fig.suptitle("Training report", fontsize=18)
    fig.gca().xaxis.set_major_locator(MultipleLocator(x_locator_tick))