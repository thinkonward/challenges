import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr
import scipy
from typing import Union, Tuple


def rescale_volume(seismic, low=0, high=100):
    """
    Rescaling 3D seismic volumes 0-255 range, clipping values between low and high percentiles
    """
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255

    return seismic


def dummy_prediction(input_volume):
    """
    Generates a dummy prediction for fault locations in a 3D seismic volume.

    This function creates a binary prediction array of the same shape as the input volume,
    where each element is randomly assigned a value of 0 or 1. The probability of an element
    being 1 (indicating a fault) is 3%, and the probability of being 0 (indicating no fault) is 97%.

    Parameters:
    input_volume (numpy.ndarray): A 3D numpy array representing the seismic volume.

    Returns:
    numpy.ndarray: A 3D numpy array of the same shape as input_volume, containing binary values
                   where 1 indicates a fault location and 0 indicates no fault.
    """
    prediction = np.random.choice([0, 1], size=input_volume.shape, p=[0.97, 0.03])
    return prediction


def create_submission(
    sample_id: str, prediction: np.ndarray, submission_path: str, append: bool = True
):
    """Function to create submission file out of one test prediction at time

    Parameters:
        sample_id: id of survey used for perdiction
        prediction: binary 3D np.ndarray of predicted faults
        submission_path: path to save submission
        append: whether to append prediction to existing .npz or create new one

    Returns:
        None
    """

    if append:
        try:
            submission = dict(np.load(submission_path))
        except:
            print("File not found, new submission will be created.")
            submission = dict({})
    else:
        submission = dict({})

    # Positive value coordinates
    coordinates = np.stack(np.where(prediction == 1)).T
    coordinates = coordinates.astype(np.uint16)

    submission.update(dict([[sample_id, coordinates]]))

    np.savez(submission_path, **submission)


def get_dice(gt_mask, pred_mask):
    # masks should be binary
    # DICE Score = (2 * Intersection) / (Area of Set A + Area of Set B)
    intersect = np.sum(pred_mask * gt_mask)
    total_sum = np.sum(pred_mask) + np.sum(gt_mask)
    if total_sum == 0:  # both samples are without positive masks
        dice = 1.0
    else:
        dice = (2 * intersect) / total_sum
    return dice


def get_submission_score(
    gt_submission_path, prediction_submission_path, mask_shape=(300, 300, 1259)
):
    # load submissions
    gt_submission = dict(np.load(gt_submission_path))
    prediction_submission = dict(np.load(prediction_submission_path))

    # prepare place to store per sample score
    global_scores = []
    for sample_id in gt_submission.keys():
        # reconstruct gt mask
        gt_mask = np.zeros(mask_shape)
        gt_coordinates = gt_submission[sample_id]
        if gt_coordinates.shape[0] > 0:
            gt_mask[
                gt_coordinates[:, 0], gt_coordinates[:, 1], gt_coordinates[:, 2]
            ] = 1

        # reconstruct prediction mask
        pred_mask = np.zeros(mask_shape)
        pred_coordinates = prediction_submission[sample_id]
        if pred_coordinates.shape[0] > 0:
            pred_mask[
                pred_coordinates[:, 0], pred_coordinates[:, 1], pred_coordinates[:, 2]
            ] = 1

        global_scores.append(get_dice(gt_mask, pred_mask))

    sub_score = sum(global_scores) / len(global_scores)

    return sub_score


def load_seismic_data(filepath: str) -> xr.DataArray:
    """Load seismic amplitude data from npy array.

    Args:
        filepath (str):
            Path to the npy file containing the 3D seismic
            amplitude data.

    Returns:
        seis_vol (xr.DataArray):
            Seismic amplitude volume xarray with shape
            (inline, xline, twt). twt is two-way-time.

    Typical Usage Example:
        Use this function to load numpy files containing 3D
        seismic amplitude data. It conveniently outputs
        an xarray, which is an array easily manipulated
        during plotting, for example.

        seismic = load_seismic_data('/root/home/stuff/things/seismic.npy')
    """
    # Load in npy file containing 3D seismic amplitudes
    data_in = np.load(filepath, allow_pickle=True)

    # i = inline, x = xline, t = two way time
    i, x, t = map(np.arange, data_in.shape)

    # Convert npy array to xarray for easier 3D manipulation
    # and plotting
    seis_vol = xr.DataArray(
        data_in,
        name="amplitude",
        coords=[i, x, t * 0.004],
        dims=["inline", "xline", "twt"],
    )

    return seis_vol


def get_sampling_freq_window_length(seis_vol: xr.DataArray) -> Tuple[float, float]:
    """Extract sampling frequency and window length
    from 3D seismic amplitude volume.

    Args:
        seis_vol (xr.DataArray):
            Seismic amplitude volume xarray with shape
            (inline, xline, twt). twt is two-way-time.

    Returns:
        fs_mean (float):
            Sampling frequency extracted from the 3D seismic
            amplitude volume

        window_length (float):
            The desired length of the window in seconds used in the
            STFT spectral decomposition function.

    Typical Usage Example:
        Typically one would obtain sampling frequency from the seismic
        metadata or header information. STFT window length can vary
        depending upon your data and objectives. One may use the following
        function as a starting point for fs and window length, if desired:

            fs, window = get_sampling_freq_window_length(seismic)

        Use this function with caution in conjunction with the spectral
        decomposition function. Tuning of fs and window length requires
        careful consideration.
    """
    # Extract two way travel times from the seismic volume
    twt = seis_vol.coords["twt"].values

    # Calculate the mean difference between consecutive times `dt`
    dt_mean = np.diff(twt).mean()

    # Calculate the mean sampling frequency `fs`
    fs_mean = 1 / dt_mean

    # Calculate the two way travel time range
    window_length = (twt.max() - twt.min()) / 2

    print(f"Mean sampling frequency is: {fs_mean}")
    print(f"Window length is: {window_length}")

    return fs_mean, window_length


def spectral_decomp(
    seis_vol: xr.DataArray,
    fs: int = 250,
    window: float = 0.2,
    hop: int = 1,
    fft_mode: str = "onesided",
    scale_to: str = "psd",
    win_func_np: callable = np.hanning,
) -> xr.DataArray:
    """Spectral Decomposition on a 3D seismic volume.

    Args:
        seis_vol (xr.DataArray):
            3D seismic amplitude data with shape (inline, xline, twt).
            inline and xline are map-view spatial dimensions.
            twt is two-way time in seconds.

        fs (int):
            Sampling frequency of seismic volume in Hz. Default is 250.

        window (float):
            Length of the STFT (short-time Fourier transform) window.
            Default is 0.2 seconds.

        hop (int):
            Step size for the STFT window. Default is 1.

        fft_mode (str):
            Mode of utilized FFT. ‘twosided’, ‘centered’, ‘onesided’,
            or ‘onesided2X’. Refer to scipy.signal.ShortTimeFFT.fft_mode
            Default is 'onesided'.

        scale_to (str):
            Scale the result to 'magnitude' or 'power'. Default is 'magnitude'.

        win_func_np:
            1D numpy window function used for sliding window sequential FFT
            calculations. Refer to numpy window functions. Options are
            np.bartlett, np.blackman, np.hamming, np.hanning, np.kaiser.
            The choice of window may significantly impact the results.

    Returns:
        spec_decomp (xr.DataArray):
            3D spectral decomposition data with shape
            (inline, xline, freq, time)

    Typical Usage Example:
        Finds the frequency content of a 3D seismic amplitude volume.
        Please refer to scipy.signal.ShortTimeFFT for more details.

        One of the many uses would be to allow interpreters to identify
        subtle features which might be missed by amplitude data, such
        as thin beds. Expanding upon this idea, one could use this
        information for reservoir characterization and complex structural
        analysis.

        Typically one would obtain sampling frequency from the seismic
        metadata or header information. STFT window length can vary
        depending upon your data and objectives. One may use the following
        function as a starting point for fs and window length, if desired:

            fs, window = get_sampling_freq_window_length(seismic)

        Use fs and window as inputs into the spectral decomposition function
        to obtain the spectral decomposition volume. Hint, there is a tradoff
        between time and frequency resolution which depends on the window
        length.

            spec_decomp = spectral_decomp(
                seismic,
                fs=fs,
                window=window
            )

       There are many more options for windowing functions and STFT. Please only
       use this as a starting point.
    """
    # "number of data points per segment", or samples in window
    # This value is crucial b/c it balances time and frequency resolution
    nperseg = int(fs * window)

    # Using num samples and window function, calculate window
    win = win_func_np(nperseg)

    twt_seis = seis_vol["twt"]

    # Initialize the ShortTimeFFT class instance
    SFT = scipy.signal.ShortTimeFFT(
        win=win, hop=hop, fs=fs, fft_mode=fft_mode, scale_to=scale_to
    )

    # convert xarray to np.array for next steps
    seis_vol = np.array(seis_vol)

    # Perform the STFT on seis_vol
    Sxx = SFT.stft(seis_vol)  # perform the STFT

    # Get frequencies
    f = SFT.f
    # Get times and adjust time axis to match seis_vol
    t = SFT.t(seis_vol.shape[-1])

    # t = SFT.t(n=seis_vol.shape[-1], p0=min_time)
    # Get inlines and xlines
    i, x, _ = map(np.arange, seis_vol.shape)

    # Construct Sectral Decomp volume using xarray
    spec_decomp = xr.DataArray(
        np.real(np.sqrt(Sxx)),  # grab only the real part of the STFT
        name="spec_decomp",
        coords=[x, i, f, t],
        dims=["inline", "xline", "freq", "time"],
    )

    # Trim the spec_decomp xarray using the seis_vol twt values
    # Removes padding from spec decomp process
    spec_decomp_trimmed = spec_decomp.sel(time=twt_seis)

    return spec_decomp_trimmed


def plot_inline_spec_decomp_amplitude(
    spec_decomp: xr.DataArray,
    seis_vol: xr.DataArray,
    inline: int = 10,
    freq_idx: int = 2,
) -> None:
    """
    Args:
        spec_decomp (xr.DataArray):
            3D spectral decomposition data with shape
            (inline, xline, freq, time)

        inline (int):
            The inline slice to plot, keeping all other xlines.
            Default is inline 10.

        freq_idx (int):
            The index of the frequency to be plotted. The idx vs
            frequency will change depending on the STFT parameters.
            Default is index 2.

    Returns:
        None

    Typical Usage Example:
        Used to plot a selected spectral decomposition frequency on top
        of a seismic amplitude slice.

        For example, if you wanted to plot a particular inline and plot
        all computed frequencies, do the following:

            for freq in range(1, len(spec_decomp['freq'])):
                plot_inline_spec_decomp_amplitude(
                    spec_decomp,
                    seismic,
                    inline=inline,
                    freq=freq
                )
    """
    # Init figure
    plt.figure(figsize=(15, 5))
    plt.suptitle("Seismic and Spectral Decomposition")

    # Plot the seismic amplitude slice
    seis_vol.isel({"inline": inline}).T.plot.imshow(
        ax=plt.gca(), cmap="Greys", add_colorbar=False, origin="upper"
    )

    # Overlay the spectral decomposition slice on top of the amplitude slice
    spec_decomp.isel({"inline": inline, "freq": freq_idx}).T.plot.imshow(
        ax=plt.gca(), cmap="viridis", alpha=0.4, add_colorbar=True, origin="upper"
    )

    plt.show()
