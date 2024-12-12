import h5py
import numpy as np

# List of HDF5 files
h5_files = [
    'original_image-impeccable-train-data-part1.h5',
    'original_image-impeccable-train-data-part2.h5',
    'original_image-impeccable-train-data-part3.h5',
    'original_image-impeccable-train-data-part4.h5',
    'original_image-impeccable-train-data-part5.h5',
    'original_image-impeccable-train-data-part6.h5',
    'original_image-impeccable-train-data-part7.h5',
    'original_image-impeccable-train-data-part8.h5',
    'original_image-impeccable-train-data-part9.h5',
    'original_image-impeccable-train-data-part10.h5',
    'original_image-impeccable-train-data-part11.h5',
    'original_image-impeccable-train-data-part12.h5',
    'original_image-impeccable-train-data-part13.h5',
    'original_image-impeccable-train-data-part14.h5',
    'original_image-impeccable-train-data-part15.h5',
    'original_image-impeccable-train-data-part16.h5',
    'original_image-impeccable-train-data-part17.h5',
    'original_validation.h5',
]

# Initialize accumulators for statistics
all_data_max = []
all_data_min = []
all_data_sum = 0
all_data_sq_sum = 0
total_elements = 0

all_label_max = []
all_label_min = []
all_label_sum = 0
all_label_sq_sum = 0
total_label_elements = 0

# Process each HDF5 file
for file in h5_files:
    with h5py.File(file, 'r') as h5_file:
        # Load datasets
        data = h5_file['data'][:]
        label = h5_file['label'][:]
        
        # Max and min values
        all_data_max.append(np.max(data))
        all_data_min.append(np.min(data))
        all_label_max.append(np.max(label))
        all_label_min.append(np.min(label))
        
        # Sum and squared sum for mean and std calculation
        all_data_sum += np.sum(data)
        all_data_sq_sum += np.sum(data**2)
        total_elements += data.size
        
        all_label_sum += np.sum(label)
        all_label_sq_sum += np.sum(label**2)
        total_label_elements += label.size

# Compute overall statistics for data
max_value = np.max(all_data_max)
min_value = np.min(all_data_min)
mean_value = all_data_sum / total_elements
std_value = np.sqrt((all_data_sq_sum / total_elements) - mean_value**2)

# Compute overall statistics for label
max_value1 = np.max(all_label_max)
min_value1 = np.min(all_label_min)
mean_value1 = all_label_sum / total_label_elements
std_value1 = np.sqrt((all_label_sq_sum / total_label_elements) - mean_value1**2)

# Print the results
print(f"Data Statistics: Max={max_value}, Min={min_value}, Mean={mean_value}, Std={std_value}")
print(f"Label Statistics: Max={max_value1}, Min={min_value1}, Mean={mean_value1}, Std={std_value1}")
