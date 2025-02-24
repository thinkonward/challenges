import pickle
from constants import SEISMIC_C_SUM, SEISMIC_C_DIV
import os
from constants import EXPAND_TILE, THRESHOLD_3D, XYZ_DIV, GAP_DIF
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision


def set_tensorflow_mixed_precision_and_jit(jit=True, mix=True):
    """
    Function to set the global policy for mixed precision and use jit optimizations.
    :param jit: Whether to use jit optimizations
    :type jit: bool
    :param mix: Whether to use mixed precision
    :type mix: bool
    :return: None
    """
    if mix:
        mixed_precision.set_global_policy('mixed_float16')
    if jit:
        tf.config.optimizer.set_jit(True)
    return None


def preprocess_raw_data(directory_path):
    """
    Function to preprocess raw data and save it as a .npy file.
    Use normalize_seismic function to normalize the seismic data.
    :param directory_path: Path to the directory containing the raw data.
    :type directory_path: str
    :return: None
    """
    file_ids = []
    seismic_data = []

    # Iterate over the items in the directory
    for item in os.listdir(directory_path):
        item_path_part = os.path.join(directory_path, item)
        if os.path.isdir(item_path_part):
            file_ids.append(item_path_part)

    for sample_id in file_ids:
        # Path to the subfolder/sample
        sample_path = os.path.join(directory_path, sample_id)

        # List and print all files in the sample subfolder
        files = os.listdir(sample_path)
        print("\n Files in the sample subfolder:", files)

        # Iterate over the files and load the .npy files.
        for file in files:
            if file.startswith("seismicCubes_") and file.endswith(".npy"):
                seismic = np.load(os.path.join(sample_path, file), allow_pickle=True)
                seismic_data.append(normalize_seismic(seismic))

    # unify the seismic data and save as uint8 to save space
    seismic_data = np.stack(seismic_data, axis=0, dtype='uint8')
    np.save(os.path.join(directory_path, 'seismic_processed.npy'), seismic_data)

    with open(os.path.join(directory_path, 'file_ids.pkl'), 'wb') as f:
        pickle.dump(file_ids, f)
    return None


def normalize_seismic(seismic_data):
    """
    Function to normalize seismic data.
    Key points:
    - Use log1p to normalize the high range values.
    - Use sign to keep the negative values.
    - Use a simple linear transformation to use almost all the 0-255 range.
    For a better performance, use int16. (The ram usage during training will be almost twice)
    :param seismic_data: Seismic data to normalize
    :type seismic_data: np.ndarray
    :return: Normalized seismic data
    :rtype: np.ndarray
    """
    seismic_data = np.sign(seismic_data) * np.log1p(np.abs(seismic_data))
    seismic_data = np.clip((seismic_data + SEISMIC_C_SUM) / SEISMIC_C_DIV * 255, 0, 255).astype(np.uint8)
    return seismic_data


def prediction_slice(model, seismic_slice, overlap, off_set, model_size_input):
    """
    Function to predict on a one slice of the seismic data. It uses a window sliding approach to predict on the whole
    slice.
    :param model: model to use for prediction
    :type model: tf.keras.Model
    :param seismic_slice: slice of the seismic data to predict on
    :type seismic_slice: np.ndarray
    :param overlap: overlap to use for the window sliding approach
    :type overlap: int
    :param off_set: offset to use for the window sliding approach to avoid the border effect
    :type off_set: int
    :param model_size_input: size of the model input
    :type model_size_input: int
    :return: prediction
    :rtype: np.ndarray
    """
    # Initialize prediction
    prediction = np.zeros((seismic_slice.shape[0], seismic_slice.shape[1], 1), dtype=np.float32)

    # calculate the size of the non-overlapping center and other params for the window sliding approach
    center_size = model_size_input - overlap * 2
    tiles = []
    rows = (seismic_slice.shape[0] - 2 * overlap) // center_size
    columns = (seismic_slice.shape[1] - 2 * overlap) // center_size

    # obtain the tiles for feeding the model
    for i in range(rows):
        for j in range(columns):
            tile = seismic_slice[i * center_size:i * center_size + model_size_input,
                   j * center_size:j * center_size + model_size_input]
            tiles.append(tile)
    tiles = np.stack(tiles, axis=0)

    # predict on the tiles
    tiles_pred = model.predict_on_batch(tiles)[0]

    # reconstruct the prediction from the predicted tiles
    for i in range(rows):
        for j in range(columns):
            tile_pred = tiles_pred[i * columns + j]
            if overlap > 0:
                tile_pred = tile_pred[overlap:-overlap, overlap:-overlap]

            prediction[overlap + i * center_size:overlap + (i + 1) * center_size,
            overlap + j * center_size:overlap + (j + 1) * center_size] = tile_pred

    prediction = prediction[overlap:-overlap - off_set, overlap:-overlap - off_set]
    prediction = np.where(prediction > 0.5, 1, 0)
    return prediction


def get_prediction(model, seismic_data, model_size_input, axis_order, overlap=None, smooth_zero=True):
    """
    Function to predict on the entire seismic data using a sliding window approach.
    :param model: Model to use for prediction
    :type model: tf.keras.Model
    :param seismic_data: Seismic data to predict on 4D array (n_files, height, width, channels)
    :type seismic_data: np.ndarray
    :param model_size_input: Size of the model input
    :type model_size_input: int
    :param axis_order: Order of the axes
    :type axis_order: list
    :param overlap: Overlap to use for the window sliding approach
    :type overlap: int
    :param smooth_zero: Whether to smooth zero values
    :type smooth_zero: bool
    :return: Prediction
    :rtype: np.ndarray
    """
    height, width = seismic_data.shape[1], seismic_data.shape[2]

    offset = height % (model_size_input - overlap * 2)
    if offset != 0:
        offset = (model_size_input - overlap * 2) - offset

    # Add xyz info
    xyz_info = np.zeros((height, width, 3), dtype='uint8')
    xyz_info[..., 0] = np.reshape(range(height), (height, 1)) / XYZ_DIV[axis_order[0]]
    xyz_info[..., 1] = np.reshape(range(width), (1, width)) / XYZ_DIV[axis_order[1]]

    final_prediction = []
    for i in range(seismic_data.shape[0]):
        # Add copy of first and last slice for the 3d module inside the model
        seismic_i = seismic_data[i]
        for _ in range(EXPAND_TILE * GAP_DIF):
            seismic_i = np.concatenate([seismic_i[..., 0][..., np.newaxis],
                                        seismic_i, seismic_i[..., -1][..., np.newaxis]], axis=-1)
        prediction_slices = []
        for j in range(EXPAND_TILE * GAP_DIF, seismic_i.shape[-1] - EXPAND_TILE * GAP_DIF):
            seismic_i_slice_j = seismic_i[...,
                                j - EXPAND_TILE * GAP_DIF:j + 1 + EXPAND_TILE * GAP_DIF:GAP_DIF]

            # Add xyz info
            xyz_info_temp = np.copy(xyz_info)
            xyz_info_temp[..., 2] = j / XYZ_DIV[axis_order[2]]
            seismic_i_slice_j = np.concatenate([seismic_i_slice_j, xyz_info_temp], axis=-1)

            pad_info = ((overlap, overlap + offset), (overlap, overlap + offset), (0, 0))
            seismic_i_slice_j_pad = np.pad(seismic_i_slice_j, pad_info, mode='constant')

            # make the prediction
            prediction_slice_j = prediction_slice(model, seismic_i_slice_j_pad, overlap, offset, model_size_input)
            prediction_slices.append(prediction_slice_j)

        prediction_slices = np.stack(prediction_slices, axis=-1)[..., 0, :]

        if np.sum(prediction_slices) < THRESHOLD_3D and smooth_zero:
            prediction_slices = prediction_slices*0

        prediction_slices = prediction_slices.astype('uint8')
        final_prediction.append(prediction_slices)
    return np.stack(final_prediction, axis=0)


def create_submission(sample_id: str, prediction: np.ndarray, submission_path: str, append: bool = True):
    """Function to create submission file out of one test prediction at time
    Parameters:
        sample_id: id of survey used for prediction.
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