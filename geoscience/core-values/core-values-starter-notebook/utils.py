import numpy as np
import random
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw
from anytree import Node, RenderTree
from glob import glob
import os
import re
import copy
import shutil
from tqdm import tqdm
from collections import Counter
import json


def find_unique_rgb(directory_path):
    """
    Find unique RGB values in label images within a specified directory.

    This function scans a specified directory for files with names ending in '_lab.png',
    extracts the unique RGB values from these images, and returns a list of these unique values.

    Parameters:
    directory_path (str): Path to the directory containing the label images.

    Returns:
    list: A list of unique RGB values found in the label images.
    """
    unique_rgb_values = set()

    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        # Filter for files ending with _lab.png
        lab_files = [f for f in files if f.endswith("_lab.png")]

        # Initialize the progress bar
        for file in tqdm(lab_files, desc="Processing files"):
            file_path = os.path.join(root, file)
            image = Image.open(file_path)
            label = np.asarray(image)

            # Reshape the array to a 2D array where each row is an RGB value
            reshaped_label = label.reshape(-1, label.shape[2])

            # Find unique rows (unique RGB values) and add to the set
            unique_rgb_values.update(map(tuple, np.unique(reshaped_label, axis=0)))

    # Convert the set to a list
    return list(unique_rgb_values)


def plot_class_RGBcolors(class_dict):
    """
    Plot RGB colors for each class in a dictionary.

    This function takes a dictionary where the keys are class numbers and the values are RGB color values.
    It plots each color as a rectangle with the corresponding RGB values and class numbers displayed.

    Parameters:
    class_dict (dict): A dictionary where keys are class numbers (int) and values are RGB color values (list of three ints).

    Returns:
    None
    """
    # Number of colors
    num_colors = len(class_dict)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(num_colors * 2, 2))

    # Plot each color as a rectangle
    for i, (class_num, rgb) in enumerate(class_dict.items()):
        color = [c / 255 for c in rgb]  # Normalize RGB values to [0, 1] range
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=color)
        ax.add_patch(rect)
        ax.text(
            i + 0.5, 0.5, str(rgb), ha="center", va="center", fontsize=12, color="black"
        )
        ax.text(
            i + 0.5,
            1.1,
            f"Class {class_num}",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
        )

    # Set limits and remove axes
    ax.set_xlim(0, num_colors)
    ax.set_ylim(0, 1.2)
    ax.axis("off")

    # Show the plot
    plt.show()
    return


def create_label_mask(image_path, class_dict):
    """
    Create a label mask for an image based on a class dictionary.

    This function loads an image, converts it to a numpy array, and creates a label mask where each pixel
    is assigned a class label based on its RGB value. The class labels are determined using a provided
    class dictionary that maps RGB values to class labels.

    Parameters:
    image_path (str): Path to the image file.
    class_dict (dict): Dictionary where keys are class labels (int) and values are RGB values (list of three ints).

    Returns:
    numpy.ndarray: A 2D numpy array where each element is a class label corresponding to the pixel in the image.
    """
    # Load the image and convert it to a numpy array
    image = Image.open(image_path)
    image_array = np.asarray(image)

    # Initialize the mask with zeros
    mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)

    # Create a reverse dictionary for quick lookup
    reverse_class_dict = {tuple(v): k for k, v in class_dict.items()}

    # Create a 2D array of tuples representing the RGB values
    rgb_tuples = image_array.reshape(-1, 3)

    # Create a 1D array to store the class labels
    labels = np.zeros(rgb_tuples.shape[0], dtype=np.uint8)

    # Assign class labels based on RGB values
    for rgb, class_label in reverse_class_dict.items():
        mask_indices = np.all(rgb_tuples == rgb, axis=1)
        labels[mask_indices] = class_label

    # Reshape the labels array back to the original image shape
    mask = labels.reshape(image_array.shape[0], image_array.shape[1])

    return mask


def create_sample_submission(directory_path, sample_submission_path):
    """
    Creates and saves a dummy sample submission file in .npz format.

    This function reads .png files from the specified directory, generates dummy masks for each image
    using the dummy_mask_pred function, and saves the results in a .npz file at the specified path.

    Parameters:
    directory_path (str): The path to the directory containing .png files.
    sample_submission_path (str): The path (including filename) where the .npz file will be saved.

    Returns:
    dict: A dictionary where keys are image IDs with '_lab' suffix and values are the predicted masks.
    """

    # Initialize the dictionary to store the predicted masks
    pred_masks_dict = {}

    # Get the list of .png files in the directory
    png_files = [f for f in os.listdir(directory_path) if f.endswith(".png")]

    # Process each .png file
    for png_file in tqdm(png_files, desc="Processing images"):
        image_path = os.path.join(directory_path, png_file)
        image_id = png_file.split("_")[0]

        # Generate a random class list
        m = np.random.randint(1, 6)
        class_list = np.random.choice(
            [1, 3, 4, 5, 6, 7, 8, 9], m, replace=False
        ).tolist()

        # Generate the dummy mask prediction
        pred_mask = dummy_mask_pred(image_path, class_list)

        # Store the predicted mask in the dictionary
        pred_masks_dict[f"{image_id}_lab"] = pred_mask

    # Save the dictionary as a .npz file
    np.savez_compressed(sample_submission_path, **pred_masks_dict)

    return pred_masks_dict


def plot_image_with_mask(image_path, mask):
    """
    Plots an image with an overlaid mask, displaying mask values at certain pixels.

    Parameters:
    image_path (str): The file path to the image to be loaded and displayed.
    mask (numpy.ndarray): A 2D array of the same dimensions as the image, containing values to be displayed on the image.

    Returns:
    None
    """
    # Load the image
    image = Image.open(image_path)
    image_array = np.asarray(image)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the image
    ax.imshow(image_array)

    # Overlay the mask numbers at some of the pixels with a big font
    step = max(
        1, min(mask.shape[0], mask.shape[1]) // 10
    )  # Adjust step size based on image size
    for i in range(0, mask.shape[0], step):
        for j in range(0, mask.shape[1], step):
            if (i // step + j // step) % 2 == 0:  # Skip some places to avoid cramping
                ax.text(
                    j,
                    i,
                    str(mask[i, j]),
                    color="black",
                    fontsize=12,
                    ha="center",
                    va="center",
                )

    # Remove axes
    ax.axis("off")

    # Show the plot
    plt.show()
    return


def mask_to_image(mask, class_dict=None):
    """
    Converts a mask to an RGB image and displays it with overlaid class numbers.

    Parameters:
    mask (numpy.ndarray): A 2D array where each value represents a class.
    class_dict (dict, optional): A dictionary mapping class labels to RGB color tuples.
                                 If None, a default color mapping is used.

    Returns:
    None
    """
    if class_dict is None:
        class_dict = {
            1: (0, 77, 38),
            2: (255, 255, 255),
            3: (203, 20, 20),
            4: (191, 191, 147),
            5: (133, 223, 246),
            6: (159, 157, 12),
            7: (230, 193, 156),
            8: (139, 87, 42),
            9: (200, 200, 200),
        }

    # Create an RGB image from the mask
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for cls, color in class_dict.items():
        rgb_image[mask == cls] = color

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the RGB image
    ax.imshow(rgb_image)

    # Overlay class numbers at some of the pixels with a big font
    step = max(
        1, min(mask.shape[0], mask.shape[1]) // 10
    )  # Adjust step size based on image size
    for i in range(0, mask.shape[0], step):
        for j in range(0, mask.shape[1], step):
            if (i // step + j // step) % 2 == 0:  # Skip some places to avoid cramping
                ax.text(
                    j,
                    i,
                    str(mask[i, j]),
                    color="black",
                    fontsize=12,
                    ha="center",
                    va="center",
                )

    # Remove axes
    ax.axis("off")

    # Show the plot
    plt.show()
    return


def dummy_mask_pred(image_path, class_list):
    """
    Generates a dummy mask prediction for an image with random polygons for each class.

    Parameters:
    image_path (str): The file path to the image to be loaded.
    class_list (list): A list of class labels to be included in the mask.

    Returns:
    numpy.ndarray: A 2D array representing the mask with the same dimensions as the input image.
    """
    # Load the image and get its dimensions
    image = Image.open(image_path)
    width, height = image.size

    # Initialize the mask with 2 (Class 2)
    mask_pred_dummy = np.full((height, width), 2, dtype=np.uint8)

    # Create a drawing context
    mask_image = Image.new("L", (width, height), 2)
    draw = ImageDraw.Draw(mask_image)

    for cls in class_list:
        # Set the number of polygons for each class to a random number between 1 to 4
        num_polygons = np.random.randint(1, 4)

        for _ in range(num_polygons):
            while True:
                # Randomly generate polygon vertices
                num_vertices = np.random.randint(4, 7)  # Polygons with 4 to 7 vertices
                vertices = [
                    (np.random.randint(0, width), np.random.randint(0, height))
                    for _ in range(num_vertices)
                ]

                # Create a temporary mask to check for overlap
                temp_mask = Image.new("L", (width, height), 0)
                temp_draw = ImageDraw.Draw(temp_mask)
                temp_draw.polygon(vertices, outline=cls, fill=cls)
                temp_mask_array = np.array(temp_mask)

                # Check for overlap
                if np.all((mask_pred_dummy == 2) | (temp_mask_array == 0)):
                    # No overlap, draw the polygon on the actual mask
                    draw.polygon(vertices, outline=cls, fill=cls)
                    mask_pred_dummy[temp_mask_array == cls] = cls
                    break

    # Convert the mask image back to a numpy array
    mask_pred_dummy = np.array(mask_image)

    return mask_pred_dummy


def sketch_directory_tree():
    # Create nodes for the directory structure
    root = Node("current_directory")
    starter_notebook = Node("starter_notebook.ipynb", parent=root)
    utils = Node("utils.py", parent=root)
    data = Node("data", parent=root)
    train = Node("train", parent=data)
    train_unlabeled = Node("train_unlabeled", parent=data)
    test = Node("test", parent=data)

    # Render the directory tree
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))
    return


def plot_raw_label(directory_path, sample_id):
    """
    Plots the raw core image and segmented image side by side.

    Parameters:
    directory_path (str): The path to the directory containing the images.
    sample_id (str): The ID of the sample to be plotted.

    Returns:
    None
    """
    # Construct file paths
    img_path = os.path.join(directory_path, f"{sample_id}_img.png")
    label_path = os.path.join(directory_path, f"{sample_id}_lab.png")

    # Load images
    img = mpimg.imread(img_path)
    label = mpimg.imread(label_path)

    # Plot images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img)
    axs[0].set_title("Raw Core Image")
    axs[0].axis("off")

    axs[1].imshow(label)
    axs[1].set_title("Segmented Image")
    axs[1].axis("off")

    plt.subplots_adjust(wspace=-0.6)  # Adjust the space between the plots
    plt.show()
    return


def create_sample_answerkey_pred(
    directory_path, n, class_dict, sample_gt_submission, sample_pred_submission
):
    """
    Create sample ground truth and prediction masks for a specified number of images.

    This function processes a specified number of '_lab.png' files in a directory, generates ground truth
    masks using a provided class dictionary, and creates dummy prediction masks. The ground truth and
    prediction masks are then saved as compressed .npz files at the specified file paths.

    Parameters:
    directory_path (str): Path to the directory containing the image files.
    n (int): Number of images to process.
    class_dict (dict): Dictionary where keys are class labels (int) and values are RGB values (list of three ints).
    sample_gt_submission (str): Path (including filename) where the ground truth .npz file will be saved.
    sample_pred_submission (str): Path (including filename) where the prediction .npz file will be saved.

    Returns:
    None
    """
    # Initialize an empty dictionary to store ground truth masks and prediction masks
    gt_masks_dict = {}
    pred_masks_dict = {}

    for i in tqdm(range(1, n + 1), desc="Processing images"):
        gt_filename = f"{i}_lab.png"
        gt_image_path = os.path.join(directory_path, gt_filename)

        if os.path.exists(gt_image_path):
            # Create ground truth label mask using create_label_mask function
            gt_mask = create_label_mask(gt_image_path, class_dict)
            gt_masks_dict[f"{i}_lab"] = gt_mask

            # Create dummy prediction mask using dummy_mask_pred function
            l1 = list(np.unique(gt_mask))
            full_set = set(range(1, 10))
            set2 = full_set - set(l1)
            l1.append(random.choice(list(set2)))
            l_final = list(l1)
            class_list = [int(x) for x in l_final if x != 2]

            pred_mask = dummy_mask_pred(gt_image_path, class_list)
            pred_masks_dict[f"{i}_lab"] = pred_mask

    # Save ground truth masks as an .npz file at sample_gt_submission path
    np.savez_compressed(sample_gt_submission, **gt_masks_dict)

    # Save prediction masks as an .npz file at sample_pred_submission path
    np.savez_compressed(sample_pred_submission, **pred_masks_dict)
    return


def calculate_dice(mask_gt, mask_pred, epsilon=1e-6, include_backgroundclass=True):
    """
    Calculates the mean Dice coefficient between two masks.
    Note that the default values of the parameters will be used for calculating your predictive leaderboard score.

    Parameters:
    mask_gt (numpy.ndarray): Ground truth mask, a 2D array where each value represents a class.
    mask_pred (numpy.ndarray): Predicted mask, a 2D array with the same dimensions as mask_gt.
    epsilon (float, optional): A small constant to avoid division by zero. Default is 1e-6.
    include_backgroundclass (bool, optional): If False, the background class (assumed to be class 2) is excluded from the calculation. Default is True.

    Returns:
    float: The mean Dice coefficient across all classes, or NaN if no classes are present.
    """
    dice = []

    # Find unique classes present in both masks
    unique_classes_gt = np.unique(mask_gt)
    unique_classes_pred = np.unique(mask_pred)
    union_classes = np.union1d(unique_classes_gt, unique_classes_pred)

    # Exclude background class if include_backgroundclass is False
    if not include_backgroundclass:
        union_classes = union_classes[union_classes != 2]

    for cls in union_classes:
        intersection = np.logical_and(mask_gt == cls, mask_pred == cls).sum()
        gt_sum = (mask_gt == cls).sum()
        pred_sum = (mask_pred == cls).sum()
        dice.append((2 * intersection + epsilon) / (gt_sum + pred_sum + epsilon))

    # Handle edge case where there are no classes to calculate Dice score for
    if len(dice) == 0:
        return np.nan

    return np.mean(dice)  # Mean Dice


def calculate_score(answerkey_file, submission_file):
    """
    Calculate the average Dice score for the given answer key and submission files.

    Parameters:
    answerkey_file (str): Path to the answer key .npz file.
    submission_file (str): Path to the submission .npz file.

    Returns:
    float: The average Dice score, excluding any nan values.
    """
    answerkey = dict(np.load(answerkey_file))
    submission = dict(np.load(submission_file))

    dice_samples = []

    for key in answerkey.keys():
        mask_gt = answerkey[key]
        mask_pred = submission[key]

        # Calculate the Dice score for the current key
        dice_score = calculate_dice(mask_gt, mask_pred)
        dice_samples.append(dice_score)

    # Calculate and return the average Dice score, excluding nan values
    return np.nanmean(dice_samples)
