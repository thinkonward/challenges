import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from typing import Sequence, Optional, List, Dict, Tuple, Any
import copy
import random

from simulator import *


def generate_sensors_random(
    Lx: float, Ly: float, m: int, seed: Optional[int] = 32, margin: float = 0.05
) -> np.ndarray:
    """
    Randomly generate m sensor locations within the rectangular domain,
    keeping a margin away from the boundaries.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(margin * Lx, (1 - margin) * Lx, size=m)
    ys = rng.uniform(margin * Ly, (1 - margin) * Ly, size=m)
    return np.column_stack([xs, ys])


def add_noise(
    Y: np.ndarray, sigma: float = 0.01, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Add i.i.d. Gaussian noise to a time-series matrix.
    """
    rng = np.random.default_rng() if rng is None else rng
    return Y + rng.normal(0.0, sigma, size=Y.shape)


def simulate_measurements(
    solver,
    dt: float,
    nt: int,
    sensors_xy: np.ndarray,
    sources: list,
    T0: Optional[float] = None,
    noise_std: float = 0.0,
    store_every: int = 1,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Run the heat diffusion simulation and sample temperature at given sensors.

    This function calls the solver, relying on it to handle grid-dependent
    parameters like sigma and the default 'on' window.
    """

    # The solver (Heat2D.solve) now handles sigma calculation internally.

    # Run solver with the sources list
    times, fields = solver.solve(
        dt=dt, nt=nt, T0=T0, sources=sources, store_every=store_every
    )

    # Build sensor time series
    m = sensors_xy.shape[0]
    S = len(fields)
    Y = np.zeros((S, m), dtype=float)
    for k, U in enumerate(fields):
        Y[k, :] = solver.sample_sensors(U, sensors_xy)

    if noise_std > 0:
        Y_noisy = add_noise(Y, sigma=noise_std)
    else:
        Y_noisy = Y

    return times, Y, Y_noisy, fields


def random_sources(
    n_sources: int,
    Lx: float,
    Ly: float,
    q_range: Tuple[float, float],
    on_window: Tuple[float, float],
    min_sep: float = 0.05,
    seed: Optional[int] = None,
) -> list:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    attempts = 0
    while len(xs) < n_sources and attempts < 10000:
        attempts += 1
        x = rng.uniform(0.05 * Lx, 0.95 * Lx)
        y = rng.uniform(0.05 * Ly, 0.95 * Ly)
        ok = True
        for xp, yp in zip(xs, ys):
            if np.hypot(x - xp, y - yp) < min_sep:
                ok = False
                break
        if ok:
            xs.append(x)
            ys.append(y)
    if len(xs) < n_sources:
        raise RuntimeError(
            "Could not place sources with the requested minimum separation."
        )
    qs = rng.uniform(q_range[0], q_range[1], size=n_sources)

    return [
        {"x": float(x), "y": float(y), "q": float(q), "on": on_window}
        for x, y, q in zip(xs, ys, qs)
    ]


def sample_from_distribution(
    distribution: Dict, num_samples: int, rng: np.random.Generator
) -> list:
    values = list(distribution.keys())
    probs = list(distribution.values())
    if not np.isclose(sum(probs), 1.0):
        raise ValueError("Distribution values must sum to 1.0")
    return list(rng.choice(values, size=num_samples, p=probs))


def make_training_dataset(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    num_samples: int,
    dt: float,
    num_sources_sensors: Dict[Tuple[int, int], float],
    q_range: Tuple[float, float],
    noise_std: Dict[float, float],
    nt: Dict[int, float],
    kappa: Dict[float, float],
    bc: str = "dirichlet",
    T0: Optional[float or Tuple[float, float]] = None,
    seed: int = 32,
) -> Dict:
    rng = np.random.default_rng(seed)

    # Validate distributions
    for dist in [num_sources_sensors, noise_std, nt, kappa]:
        if not np.isclose(sum(dist.values()), 1.0):
            raise ValueError("All distribution values must sum to 1.0")

    # Determine sample counts per (n_sources, n_sensors)
    config_counts = {
        k: int(round(v * num_samples)) for k, v in num_sources_sensors.items()
    }
    total_allocated = sum(config_counts.values())
    if total_allocated != num_samples:
        diff = num_samples - total_allocated
        largest_key = max(config_counts, key=config_counts.get)
        config_counts[largest_key] += diff

    # Sample parameter values
    noise_list = sample_from_distribution(noise_std, num_samples, rng)
    nt_list = sample_from_distribution(nt, num_samples, rng)
    kappa_list = sample_from_distribution(kappa, num_samples, rng)

    # Sample boundary conditions
    if bc == "random":
        bc_list = list(rng.choice(["dirichlet", "neumann"], size=num_samples))
    elif bc in ["dirichlet", "neumann"]:
        bc_list = [bc] * num_samples
    else:
        raise ValueError("bc must be 'dirichlet', 'neumann', or 'random'")

    # Sample initial conditions
    if T0 is None:
        T0_list = [0.0] * num_samples
    elif isinstance(T0, (int, float)):
        T0_list = [float(T0)] * num_samples
    elif isinstance(T0, tuple) and len(T0) == 2:
        T0_list = list(rng.uniform(T0[0], T0[1], size=num_samples))
    else:
        raise ValueError("T0 must be None, a scalar, or a tuple (min, max)")

    samples = []
    sample_id_counter = 0
    sample_index = 0

    for (n_sources, m_sensors), count in config_counts.items():
        for _ in range(count):
            noise_std_i = noise_list[sample_index]
            nt_i = nt_list[sample_index]
            kappa_i = kappa_list[sample_index]
            bc_i = bc_list[sample_index]
            T0_i = T0_list[sample_index]
            on_window_i = (0.0, float(nt_i * dt))

            s_seed = int(rng.integers(0, 1_000_000))
            solver = Heat2D(Lx, Ly, nx, ny, kappa=kappa_i, bc=bc_i)

            sources = random_sources(
                n_sources,
                Lx,
                Ly,
                q_range=q_range,
                on_window=on_window_i,
                min_sep=0.07 * min(Lx, Ly),
                seed=s_seed,
            )
            sensors_xy = generate_sensors_random(
                Lx, Ly, m=m_sensors, seed=s_seed, margin=0.05
            )
            times, Y, Y_noisy, _ = simulate_measurements(
                solver,
                dt,
                nt_i,
                sensors_xy,
                sources,
                T0=T0_i,
                noise_std=noise_std_i,
                store_every=1,
            )

            samples.append(
                {
                    "sample_id": f"sample_{sample_id_counter}",
                    "n_sources": n_sources,
                    "true_sources": sources,
                    "sensors_xy": sensors_xy,
                    "Y": Y,
                    "Y_noisy": Y_noisy,
                    "sample_metadata": {
                        "noise_std": float(noise_std_i),
                        "nt": int(nt_i),
                        "kappa": float(kappa_i),
                        "bc": str(bc_i),
                        "T0": float(T0_i),
                    },
                }
            )
            sample_id_counter += 1
            sample_index += 1

    return {"samples": samples, "meta": {"q_range": q_range, "dt": dt}}


def remove_source_parameters(dataset):
    """
    Removes the 'true_sources' and 'Y' keys from each sample in the dataset.

    Parameters:
        dataset (dict): A dictionary containing a 'samples' key with a list of sample dictionaries.

    Returns:
        dict: A new dataset dictionary with 'true_sources' and 'Y' removed from each sample.
    """
    # Make a deep copy to avoid modifying the original dataset
    new_dataset = copy.deepcopy(dataset)

    for sample in new_dataset.get("samples", []):
        sample.pop("true_sources", None)  # Remove 'true_sources' if it exists
        sample.pop("Y", None)

    return new_dataset


def dummy_source_estimator(
    dataset: Dict,
    Lx: float,
    Ly: float,
    q_range: Tuple[float, float],
    N_max: int,
    number_candidates: str,
) -> List[Dict]:
    """
    Generate dummy estimated source candidates for each sample in the dataset.

    Parameters:
    - dataset (Dict): A dictionary containing sample data. Each sample has keys like 'sample_id', 'n_sources', etc.
    - Lx (float): Maximum value for x-coordinate of sources.
    - Ly (float): Maximum value for y-coordinate of sources.
    - q_range (Tuple[float, float]): Range (min, max) for source intensity q.
    - Nmax (int): Maximum number of estimated source candidates per sample.

    Returns:
    - List[Dict]: A list of dictionaries, each containing:
        - 'sample_id': ID of the sample.
        - 'estimated_sources': A list of estimated source candidates. Each candidate is a list of (x, y, q) tuples.
    """
    output = []

    for sample in dataset["samples"]:
        sample_id = sample["sample_id"]
        n_sources = sample["n_sources"]
        if number_candidates == "random":
            n_candidates = random.randint(1, N_max)
        else:
            n_candidates = N_max

        estimated_sources = []
        for _ in range(n_candidates):
            candidate = [
                (
                    random.uniform(0, Lx),
                    random.uniform(0, Ly),
                    random.uniform(q_range[0], q_range[1]),
                )
                for _ in range(n_sources)
            ]
            estimated_sources.append(candidate)

        output.append({"sample_id": sample_id, "estimated_sources": estimated_sources})

    return output


def plot_sensor_temp(dataset: Dict, sample_ids: List[str]) -> None:
    """
    Plot sensor positions and temperature time histories for the given sample IDs.

    For each sample ID, this function creates a 1x2 subplot figure:

    Left subplot
    ------------
    - Draws the spatial domain as a rectangle with fixed size Lx = 2, Ly = 1.
    - Plots true sources as **red stars** at (x, y) from `true_sources` and annotates
      each source with its intensity: `q = value`.
    - Plots sensors as colored circles at (x, y) from `sensors_xy`. The sensor
      colors are consistent with the right subplot line colors for that sensor index.
    - Displays metadata above the rectangle: boundary condition (bc),
      thermal diffusivity (κ), and noise standard deviation (σ).

    Right subplot
    -------------
    - Plots time histories for each sensor:
        * 'Y_noisy' as solid **slightly transparent** lines (thicker).
        * 'Y' as **dashed** lines (**thicker** than before), using the **same color**
          as the corresponding noisy line so pairs are visually linked.
    - Time axis is computed as t = np.arange(n_steps) * dt, where dt is read
      from dataset['meta']['dt'].

    Parameters
    ----------
    dataset : dict
        A dictionary with:
          - 'samples': list of sample dicts; each sample must contain:
              * 'sample_id': str
              * 'true_sources': list of dicts with 'x', 'y', 'q'
              * 'sensors_xy': np.ndarray of shape (n_sensors, 2)
              * 'Y': np.ndarray of shape (n_steps, n_sensors)
              * 'Y_noisy': np.ndarray of the same shape as 'Y'
              * 'sample_metadata': dict with 'bc', 'kappa', 'noise_std'
          - 'meta': dict with 'dt' (float), used for the time axis.

    sample_ids : list of str
        Sample IDs to plot. One figure (1x2) is created per sample ID.

    Returns
    -------
    None
        Displays the figures with matplotlib.

    Notes
    -----
    - Coordinates are relative to the bottom-left corner of the domain rectangle.
    - If a provided sample_id is not found, a warning is printed and that ID
      is skipped.
    - Legend on the right shows *noisy* series only (solid lines), to avoid
      duplication; dashed lines (noise-free) have no separate legend entries.

    Examples
    --------
    >>> plot_sensor_temp(dataset, ['sample_0', 'sample_2'])
    """
    # Fixed domain size per requirement
    Lx, Ly = 2.0, 1.0

    # Global dt (same across samples), default defensively to 0.004 if missing
    dt = float(dataset.get("meta", {}).get("dt", 0.004))

    # Build lookup by sample_id
    samples = dataset.get("samples", [])
    sample_lookup = {s.get("sample_id"): s for s in samples if isinstance(s, dict)}

    # Color palette for sensors; tab10 repeats every 10 sensors
    cmap = plt.get_cmap("tab10")

    for sid in sample_ids:
        if sid not in sample_lookup:
            print(f"Warning: sample_id '{sid}' not found in dataset.")
            continue

        sample = sample_lookup[sid]

        # Extract required fields
        true_sources = sample.get("true_sources", [])
        sensors_xy = np.asarray(sample.get("sensors_xy"))
        Y = np.asarray(sample.get("Y"))
        Y_noisy = np.asarray(sample.get("Y_noisy"))
        meta = sample.get("sample_metadata", {})

        # Basic shape sanity (explicit checks)
        if sensors_xy.ndim != 2 or sensors_xy.shape[1] != 2:
            raise ValueError(
                f"Sample '{sid}': 'sensors_xy' must be of shape (n_sensors, 2); got {sensors_xy.shape}."
            )
        if Y.shape != Y_noisy.shape:
            raise ValueError(
                f"Sample '{sid}': 'Y' and 'Y_noisy' must have the same shape; "
                f"got {Y.shape} vs {Y_noisy.shape}."
            )
        n_steps, n_sensors = Y.shape
        if sensors_xy.shape[0] != n_sensors:
            raise ValueError(
                f"Sample '{sid}': number of sensors in 'sensors_xy' ({sensors_xy.shape[0]}) "
                f"does not match 'Y' columns ({n_sensors})."
            )

        # Prepare figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)
        ax_left, ax_right = axes

        # ---------------- Left subplot: domain + annotations ----------------
        ax_left.set_xlim(0, Lx)
        ax_left.set_ylim(0, Ly)
        ax_left.set_aspect("equal", adjustable="box")
        # Move the title higher (avoid overlap with metadata text)
        ax_left.set_title(f"Sample ID: {sid}", pad=28, fontsize=12, fontweight="bold")

        # Draw domain rectangle
        rect = plt.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor="black", linewidth=2)
        ax_left.add_patch(rect)

        # Plot sources as red stars, annotate with q near the star
        for src in true_sources:
            x, y = src.get("x"), src.get("y")
            q = src.get("q")
            ax_left.plot(
                x,
                y,
                marker="*",
                color="red",
                markersize=12,
                linestyle="None",
                label="Source",
            )
            # Offset the text slightly to avoid glyph overlap
            ax_left.annotate(
                f"q = {q:.3f}" if isinstance(q, (float, int)) else f"q = {q}",
                xy=(x, y),
                xytext=(6, 6),
                textcoords="offset points",
                color="red",
                fontsize=9,
                ha="left",
                va="bottom",
            )

        # Plot sensors as colored circles (consistent with right subplot colors)
        for i in range(n_sensors):
            color = cmap(i % 10)
            ax_left.plot(
                sensors_xy[i, 0],
                sensors_xy[i, 1],
                marker="o",
                color=color,
                markersize=7,
                linestyle="None",
            )

        # Metadata text just above the rectangle in axes coordinates
        bc = meta.get("bc", "unknown")
        kappa = meta.get("kappa", "unknown")
        noise_std = meta.get("noise_std", "unknown")
        # Use math symbol for sigma
        meta_text = rf"BC: {bc},  $\kappa$ = {kappa},  $\sigma$_sensor = {noise_std}"
        # Place text above the top edge (in axes coords), centered
        ax_left.text(
            0.5,
            1.02,
            meta_text,
            transform=ax_left.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
        )

        # Custom legend: one entry for Source + one for Sensor (generic)
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="red",
                linestyle="None",
                markersize=12,
                label="Source",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                markersize=7,
                label="Sensor",
            ),
        ]
        ax_left.legend(handles=legend_handles, loc="upper right", frameon=True)

        ax_left.set_xlabel("x direction", fontsize=12)
        ax_left.set_ylabel("y direction", fontsize=12)

        # ---------------- Right subplot: time histories ----------------
        time = np.arange(n_steps) * dt

        # Styling to improve dashed visibility vs solid noisy lines
        noisy_alpha = 0.6  # a bit transparent for noisy
        noisy_lw = 1.8  # thicker solid for noisy
        clean_alpha = 0.95  # nearly opaque
        clean_lw = 2.4  # THICKER dashed for clean (your request)

        for i in range(n_sensors):
            color = cmap(i % 10)
            # Noisy: solid, slightly transparent, thicker
            ax_right.plot(
                time,
                Y_noisy[:, i],
                color=color,
                linestyle="-",
                linewidth=noisy_lw,
                alpha=noisy_alpha,
                label=f"Sensor {i} (noisy)",
            )
            # Clean: dashed, thicker than before
            ax_right.plot(
                time,
                Y[:, i],
                color=color,
                linestyle="--",
                linewidth=clean_lw,
                alpha=clean_alpha,
            )

        ax_right.set_xlabel("Time", fontsize=12)
        ax_right.set_ylabel("Temperature", fontsize=12)
        ax_right.set_title("Sensor Readings")
        ax_right.grid(True, alpha=0.3)

        # Use a slightly tighter layout; leave room for left-title pad and metadata
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    return


# Types for clarity
# Define a source as a tuple (x, y, q)
Source = Tuple[float, float, float]
SourcesCandidate = List[Source]


def plot_gt_vs_est(
    gt_dataset: Dict[str, Any],
    estimated_dataset: List[Dict[str, Any]],
    sample_ids: List[str],
    solver_kwargs: Optional[Dict[str, Any]] = None,
    candidates_to_plot: Optional[List[int]] = None,
) -> None:
    """
    Plot ground-truth vs. estimated sources and corresponding sensor time histories.

    For each sample_id in `sample_ids`, this function creates a 1x2 figure:

    LEFT subplot
    ------------
    - Domain rectangle (Lx, Ly from solver_kwargs defaults to 2.0 x 1.0).
    - Sensors as colored circles (one color per sensor index; matches right subplot lines).
    - Ground-truth sources: **red stars** with `q = value` labels (3 decimals).
    - Estimated candidates: stars for each candidate; **all stars in a candidate share
      the same color** (from a qualitative colormap). Each candidate star also shows
      `q = value` near the marker in the candidate's color.
    - Metadata above the rectangle: `BC`, thermal diffusivity `κ`, and noise std `σ`.

    RIGHT subplot
    -------------
    - Clean ground-truth sensor readings `Y` as **solid** lines (sensor-consistent colors).
    - For selected candidates (via `candidates_to_plot`), simulate clean readings via Heat2D
      and overlay them with distinct **linestyles** per candidate (e.g., dashed, dash-dot,
      dotted, long-dash).  Colors per sensor are consistent with the left subplot sensor markers.
      The legend is shown in the **top-left** and enlarged.

    Parameters
    ----------
    gt_dataset : dict
        Ground-truth dataset; see previous versions for structure.

    estimated_dataset : list of dict
        Each element: {'sample_id': str, 'estimated_sources': [candidate_1, candidate_2, ...]},
        where each candidate_* is a list of (x, y, q) tuples. Assumed <= 4 candidates per sample.

    sample_ids : list of str
        Sample IDs to plot; one figure per ID.

    solver_kwargs : dict, optional
        Heat2D solver args: 'Lx' (float, default 2.0), 'Ly' (float, default 1.0),
        'nx' (int, default 100), 'ny' (int, default 50).

    candidates_to_plot : list[int], optional
        1-based indices of candidates to plot on the RIGHT subplot.  If None, all available
        (up to 4) candidates are plotted.  Out-of-range indices are ignored with a warning.

    Returns
    -------
    None
        Displays matplotlib figures.

    Notes
    -----
    - LEFT subplot still shows ALL candidates' stars and q labels (regardless of `candidates_to_plot`),
      to provide spatial context.
    - Candidates whose tuple count != ground-truth `n_sources` are skipped (warning).
    - PDE simulation assumes `simulator.Heat2D` is importable.
    """

    # --------- helper: simulate candidate readings (clean Y) ---------
    def _simulate_candidate(
        candidate: SourcesCandidate,
        sample_gt: Dict[str, Any],
        meta: Dict[str, Any],
        skw: Dict[str, Any],
    ) -> np.ndarray:
        try:
            from simulator import Heat2D  # import lazily
        except Exception as e:
            raise RuntimeError(f"Heat2D simulator is not available/importable: {e}")

        Lx = float(skw.get("Lx", 2.0))
        Ly = float(skw.get("Ly", 1.0))
        nx = int(skw.get("nx", 100))
        ny = int(skw.get("ny", 50))

        dt = float(meta["dt"])
        nt = int(sample_gt["sample_metadata"]["nt"])
        kappa = float(sample_gt["sample_metadata"]["kappa"])
        bc = sample_gt["sample_metadata"]["bc"]
        T0 = float(sample_gt["sample_metadata"]["T0"])
        sensors_xy = np.asarray(sample_gt["sensors_xy"])

        solver = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)

        sources = [
            {"x": s[0], "y": s[1], "q": s[2], "on": (0, nt * dt)} for s in candidate
        ]
        times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)

        Y_pred = np.array([solver.sample_sensors(U_t, sensors_xy) for U_t in Us])
        return Y_pred

    # --------- setup lookups ---------
    gt_samples = gt_dataset.get("samples", [])
    dt = float(gt_dataset.get("meta", {}).get("dt", 0.004))  # safe default

    gt_by_id = {s.get("sample_id"): s for s in gt_samples if isinstance(s, dict)}
    est_by_id = {
        d.get("sample_id"): d for d in estimated_dataset if isinstance(d, dict)
    }

    # --------- plotting styles ---------
    sensor_cmap = plt.get_cmap("tab10")  # per-sensor colors
    candidate_cmap = plt.get_cmap("Dark2")  # per-candidate colors (avoid red)
    candidate_styles = [
        "--",
        "-.",
        ":",
        (0, (5, 2)),
    ]  # dashed, dashdot, dotted, long-dash

    # Defaults for domain
    Lx_default, Ly_default = 2.0, 1.0

    for sid in sample_ids:
        if sid not in gt_by_id:
            print(f"Warning: sample_id '{sid}' not found in ground-truth dataset.")
            continue
        sample_gt = gt_by_id[sid]

        # Resolve estimated candidates
        est_entry = est_by_id.get(sid)
        est_candidates: List[SourcesCandidate] = []
        if est_entry is not None:
            est_candidates = est_entry.get("estimated_sources", []) or []
        else:
            print(f"Note: no estimated candidates provided for sample_id '{sid}'.")

        # Extract GT fields
        true_sources = sample_gt.get("true_sources", [])
        sensors_xy = np.asarray(sample_gt.get("sensors_xy"))
        Y_gt = np.asarray(sample_gt.get("Y"))
        meta = sample_gt.get("sample_metadata", {})

        # Sanity checks
        if sensors_xy.ndim != 2 or sensors_xy.shape[1] != 2:
            raise ValueError(
                f"Sample '{sid}': 'sensors_xy' must be (n_sensors, 2); got {sensors_xy.shape}."
            )
        n_steps_gt, n_sensors = Y_gt.shape
        if sensors_xy.shape[0] != n_sensors:
            raise ValueError(
                f"Sample '{sid}': sensors count ({sensors_xy.shape[0]}) != Y columns ({n_sensors})."
            )
        n_sources_gt = int(sample_gt.get("n_sources", len(true_sources)))

        # Effective solver_kwargs for this run
        skw = dict(solver_kwargs or {})
        skw.setdefault("Lx", Lx_default)
        skw.setdefault("Ly", Ly_default)
        skw.setdefault("nx", 100)
        skw.setdefault("ny", 50)

        # Prepare figure
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=False)
        ax_left, ax_right = axes

        # ---------------- LEFT subplot ----------------
        ax_left.set_xlim(0, skw["Lx"])
        ax_left.set_ylim(0, skw["Ly"])
        ax_left.set_aspect("equal", adjustable="box")
        ax_left.set_title(f"Sample: {sid}", pad=28, fontsize=12, fontweight="bold")

        # Domain rectangle
        ax_left.add_patch(
            plt.Rectangle(
                (0, 0), skw["Lx"], skw["Ly"], fill=False, edgecolor="black", linewidth=2
            )
        )

        # Sensors
        for i in range(n_sensors):
            color = sensor_cmap(i % 10)
            ax_left.plot(
                sensors_xy[i, 0],
                sensors_xy[i, 1],
                marker="o",
                color=color,
                markersize=7,
                linestyle="None",
            )

        # GT sources: red stars + q labels
        for src in true_sources:
            x, y, q = src["x"], src["y"], src["q"]
            ax_left.plot(x, y, marker="*", color="red", markersize=12, linestyle="None")
            ax_left.annotate(
                f"q = {q:.3f}",
                xy=(x, y),
                xytext=(6, 6),
                textcoords="offset points",
                color="red",
                fontsize=9,
                ha="left",
                va="bottom",
            )

        # Estimated candidates: draw ALL on the LEFT (spatial context)
        n_cands_total = len(est_candidates)
        n_cands_left = min(n_cands_total, 4)
        for c_idx in range(n_cands_left):
            cand = est_candidates[c_idx]
            if len(cand) != n_sources_gt:
                print(
                    f"Warning: sample '{sid}' candidate #{c_idx+1} length {len(cand)} "
                    f"!= n_sources ({n_sources_gt}); skipping on left."
                )
                continue
            cand_color = candidate_cmap(c_idx % candidate_cmap.N)
            for x, y, q in cand:
                ax_left.plot(
                    x, y, marker="*", color=cand_color, markersize=11, linestyle="None"
                )
                ax_left.annotate(
                    f"q = {q:.3f}",
                    xy=(x, y),
                    xytext=(6, 6),
                    textcoords="offset points",
                    color=cand_color,
                    fontsize=9,
                    ha="left",
                    va="bottom",
                )

        # Metadata text
        bc = meta.get("bc", "unknown")
        kappa = meta.get("kappa", "unknown")
        noise_std = meta.get("noise_std", "unknown")
        ax_left.text(
            0.5,
            1.02,
            rf"BC: {bc},  $\kappa$ = {kappa},  $\sigma$ = {noise_std}",
            transform=ax_left.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
        )

        # Legend (left)
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="red",
                linestyle="None",
                markersize=12,
                label="GT Source",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                markersize=7,
                label="Sensor",
            ),
        ]
        for c_idx in range(n_cands_left):
            cand_color = candidate_cmap(c_idx % candidate_cmap.N)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color=cand_color,
                    linestyle="None",
                    markersize=11,
                    label=f"Candidate {c_idx+1}",
                )
            )
        ax_left.legend(handles=legend_handles, loc="upper right", frameon=True)
        ax_left.set_xlabel("x")
        ax_left.set_ylabel("y")

        # ---------------- RIGHT subplot ----------------
        t = np.arange(n_steps_gt) * dt

        # Plot GT (solid)
        gt_lw = 2.0
        for i in range(n_sensors):
            color = sensor_cmap(i % 10)
            ax_right.plot(
                t,
                Y_gt[:, i],
                color=color,
                linestyle="-",
                linewidth=gt_lw,
                label=("GT" if i == 0 else None),
            )

        # Determine which candidate indices to draw on RIGHT (1-based -> 0-based)
        if candidates_to_plot is None:
            selected_indices = list(
                range(1, min(n_cands_total, 4) + 1)
            )  # 1..min(total,4)
        else:
            # Deduplicate and keep positive integers
            selected_indices = sorted(
                {i for i in candidates_to_plot if isinstance(i, int) and i >= 1}
            )
            if not selected_indices:
                print(
                    f"Note: candidates_to_plot is empty/invalid for sample '{sid}'; no candidates plotted on right."
                )

        candidate_labels_added = set()
        for cand_1b in selected_indices:
            c_idx = cand_1b - 1
            if c_idx < 0 or c_idx >= n_cands_total:
                print(
                    f"Warning: sample '{sid}': candidates_to_plot index {cand_1b} out of range (1..{n_cands_total}); skipping."
                )
                continue

            cand = est_candidates[c_idx]
            if len(cand) != n_sources_gt:
                print(
                    f"Warning: sample '{sid}': candidate #{cand_1b} length {len(cand)} "
                    f"!= n_sources ({n_sources_gt}); skipping on right."
                )
                continue

            try:
                Y_pred = _simulate_candidate(cand, sample_gt, gt_dataset["meta"], skw)
            except Exception as e:
                print(
                    f"Simulation failed for sample '{sid}', candidate #{cand_1b}: {e}"
                )
                continue

            if Y_pred.shape != Y_gt.shape:
                print(
                    f"Warning: sample '{sid}', candidate #{cand_1b} produced shape {Y_pred.shape}, "
                    f"expected {Y_gt.shape}; plotting common prefix."
                )
            n_steps_pred = min(Y_pred.shape[0], n_steps_gt)

            style = candidate_styles[c_idx % len(candidate_styles)]
            lw = 2.0
            label_once = f"Candidate {cand_1b}"

            for i in range(n_sensors):
                color = sensor_cmap(i % 10)
                label = (
                    label_once
                    if (i == 0 and label_once not in candidate_labels_added)
                    else None
                )
                ax_right.plot(
                    t[:n_steps_pred],
                    Y_pred[:n_steps_pred, i],
                    color=color,
                    linestyle=style,
                    linewidth=lw,
                    label=label,
                )
                if label is not None:
                    candidate_labels_added.add(label)

        ax_right.set_xlabel("Time")
        ax_right.set_ylabel("Temperature")
        ax_right.set_title("Sensor Readings (Clean)")
        ax_right.grid(True, alpha=0.3)
        # Legend: move to top-left and enlarge

        #         custom_handles = [Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='GT'),
        #                           Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Candidate')]
        #         ax_right.legend(handles=custom_handles, loc='upper left', fontsize=10, frameon=True)

        ax_right.legend(loc="upper left", fontsize=10, frameon=True)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    return


def scale_sources_dynamic(
    sources_candidates: List[SourcesCandidate],
) -> List[SourcesCandidate]:
    """
    Apply dynamic scaling to all source candidates using dataset-wide mean and standard deviation.

    Parameters:
    - sources_candidates (List[SourcesCandidate]): List of source candidates.

    Returns:
    - List[SourcesCandidate]: Scaled source candidates.
    """
    all_sources = np.array([s for candidate in sources_candidates for s in candidate])
    means = np.mean(all_sources, axis=0)
    stds = np.std(all_sources, axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero

    scaled_sources_candidates = []
    for candidate in sources_candidates:
        scaled_candidate = [tuple((np.array(s) - means) / stds) for s in candidate]
        scaled_sources_candidates.append(scaled_candidate)
    return scaled_sources_candidates


def scale_source(
    s: Source, scale_factors: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """Scale a single source using fixed scale factors."""
    x, y, q = s
    sx, sy, sq = scale_factors
    return (x / sx, y / sy, q / sq)


def scale_sources_fixed(
    sources_candidates: List[SourcesCandidate],
    scale_factors: Tuple[float, float, float],
) -> List[SourcesCandidate]:
    """Apply fixed scaling to all source tuples."""
    scaled_sources_candidates = []
    for candidate in sources_candidates:
        scaled_candidate = [scale_source(s, scale_factors) for s in candidate]
        scaled_sources_candidates.append(scaled_candidate)
    return scaled_sources_candidates


def sort_sources_candidate(sources_candidate: SourcesCandidate) -> SourcesCandidate:
    """Sort sources --in a source candidate list-- lexicographically."""
    return sorted(sources_candidate)


def dissimilarity(s1: SourcesCandidate, s2: SourcesCandidate) -> float:
    """Compute average Euclidean distance between corresponding sources in two source candidates of s1 and s2."""
    return np.mean([np.linalg.norm(np.array(a) - np.array(b)) for a, b in zip(s1, s2)])


def filter_distinct_candidates(
    sources_candidates: List[SourcesCandidate],
    tau: float,
    N_max: int,
    scale_factors: Tuple[float, float, float],
) -> List[SourcesCandidate]:
    """Filter out similar source candidates based on dissimilarity threshold."""
    scaled_candidates = scale_sources_fixed(sources_candidates, scale_factors)
    normalized_candidates = [sort_sources_candidate(c) for c in scaled_candidates]

    valid_indices = []
    for i, c1 in enumerate(normalized_candidates):
        is_distinct = True
        for j in valid_indices:
            c2 = normalized_candidates[j]
            if dissimilarity(c1, c2) <= tau:
                is_distinct = False
                break
        if is_distinct:
            valid_indices.append(i)
        if len(valid_indices) >= N_max:
            break

    return [sources_candidates[i] for i in valid_indices]


def forward_error(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    loss: str = "rmse",
    huber_delta: float = 0.01,
) -> float:
    """
    Compute forward data misfit over all time steps and sensors.

    Parameters:
    - Y_true: shape (nt+1, m)
    - Y_pred: shape (nt+1, m)
    - loss: "mae", "rmse", or "huber"
    - huber_delta: threshold for Huber loss

    Returns:
    - loss_value: raw error (float)
    """
    if Y_true.shape != Y_pred.shape:
        raise ValueError(
            f"Shape mismatch: Y_true {Y_true.shape} vs Y_pred {Y_pred.shape}"
        )

    R = Y_pred - Y_true
    if loss.lower() == "mae":
        return float(np.mean(np.abs(R)))
    elif loss.lower() == "rmse":
        return float(np.sqrt(np.mean(R**2)))
    elif loss.lower() == "huber":
        abs_r = np.abs(R)
        quad = 0.5 * (R**2)
        lin = huber_delta * (abs_r - 0.5 * huber_delta)
        return float(np.where(abs_r <= huber_delta, quad, lin).mean())
    else:
        return float(np.sqrt(np.mean(R**2)))  # default rmse


def simulate_candidate(
    candidate: SourcesCandidate, sample_gt: Dict, meta, solver_kwargs: Dict
) -> np.ndarray:
    """
    Simulate PDE for a given candidate and return predicted sensor readings.
    """
    Lx = solver_kwargs.get("Lx")
    Ly = solver_kwargs.get("Ly")
    nx = solver_kwargs.get("nx")
    ny = solver_kwargs.get("ny")
    dt = meta["dt"]
    nt = sample_gt["sample_metadata"]["nt"]
    kappa = sample_gt["sample_metadata"]["kappa"]
    bc = sample_gt["sample_metadata"]["bc"]
    T0 = sample_gt["sample_metadata"]["T0"]
    sensors_xy = sample_gt["sensors_xy"]

    # Build solver
    solver = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)

    # Convert candidate to solver-compatible sources
    sources = [{"x": s[0], "y": s[1], "q": s[2], "on": (0, nt * dt)} for s in candidate]

    # Solve PDE
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)

    Y_pred = np.array([solver.sample_sensors(U_t, sensors_xy) for U_t in Us])
    #     Y_pred = np.stack([solver.sample_sensors(U_t, sensors_xy) for U_t in Us], axis=0)
    return Y_pred  # Shape (S, num_sensors)


def score_sample(
    sample_gt: Dict,
    sample_pred: Dict,
    meta: Dict,
    N_max: int,
    lambda_: float,
    tau: float,
    scale_factors: Tuple[float, float, float],
    forward_loss: str,
    solver_kwargs: Dict,
) -> float:
    """
    Compute performance score for a single sample.
    """
    # Filter distinct candidates
    candidates = sample_pred["estimated_sources"]
    valid_candidates = filter_distinct_candidates(candidates, tau, N_max, scale_factors)
    N_valid = len(valid_candidates)
    print(f"\t - number of valid candidates for this sample: {N_valid}")

    if N_valid == 0:
        return 0.0

    # Compute forward loss for each valid candidate
    Y_true = sample_gt["Y"]
    losses = []
    for candidate in valid_candidates:
        Y_pred = simulate_candidate(candidate, sample_gt, meta, solver_kwargs)
        fe = forward_error(Y_true, Y_pred, loss=forward_loss)
        losses.append(fe)

    # Compute score for this sample
    score = (1 / N_valid) * sum(1 / (1 + L) for L in losses) + lambda_ * (
        N_valid / N_max
    )
    return score


def score_submission(
    gt_dataset: Dict,
    pred_dataset: List[Dict],
    N_max: int,
    lambda_: float,
    tau: float,
    scale_factors: Tuple[float, float, float],
    forward_loss: str,
    solver_kwargs: Dict,
) -> float:
    """
    Compute final score for a submission across all samples.
    """
    sample_scores = []
    gt_samples = {s["sample_id"]: s for s in gt_dataset["samples"]}
    meta = gt_dataset["meta"]

    for pred_sample in pred_dataset:
        sample_id = pred_sample["sample_id"]
        sample_gt = gt_samples[sample_id]
        print(f"Calculating score for sample ID: {sample_id}...")
        score = score_sample(
            sample_gt,
            pred_sample,
            meta,
            N_max,
            lambda_,
            tau,
            scale_factors,
            forward_loss,
            solver_kwargs,
        )
        sample_scores.append(score)

    return float(np.mean(sample_scores)) if sample_scores else 0.0
