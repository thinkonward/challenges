import numpy as np
import os
import pandas as pd
import h5py
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

def rescale_volume(seismic, low=0, high=100):
    """
    Rescaling 3D seismic volumes to 0-255 range, clipping values between low and high percentiles
    """
    minval = np.percentile(seismic, low)
    maxval = np.percentile(seismic, high)
    seismic = np.clip(seismic, minval, maxval)
    seismic = ((seismic - minval) / (maxval - minval)) * 255.0
    return seismic

class SliceServer:
    def __init__(
        self,
        root_dir,
        num_slices,
        cheesecake_factory_mode,
        limit,
        data_prefix,
        label_prefix,
        pretraining,
    ):
        self.root_dir = root_dir
        self.num_slices = num_slices
        self.cheesecake_factory_mode = cheesecake_factory_mode
        self.limit = limit
        self.data_prefix = data_prefix
        self.label_prefix = label_prefix
        self.pretraining = pretraining
        self.exclude_dirs = {".ipynb_checkpoints"}

        self.volume_ids = [
            d for path, dirs, files in os.walk(self.root_dir) for d in dirs if d not in self.exclude_dirs
        ]
        self.volume_filenames = [
            file for path, dirs, files in os.walk(self.root_dir) for file in files
        ]

        self.data_filenames = sorted(
            [file for file in self.volume_filenames if file.startswith(self.data_prefix)]
        )
        self.label_filenames = sorted(
            [file for file in self.volume_filenames if file.startswith(self.label_prefix)]
        )

        self.data_file_paths = []
        self.label_file_paths = []

        for volume_id in self.volume_ids:
            data_files = [data_file for data_file in self.data_filenames if volume_id in data_file]
            label_files = [label_file for label_file in self.label_filenames if volume_id in label_file]

            if data_files and label_files:
                data_file = data_files[0]
                label_file = label_files[0]
                self.data_file_paths.append(f"{self.root_dir}/{volume_id}/{data_file}")
                self.label_file_paths.append(f"{self.root_dir}/{volume_id}/{label_file}")

        self.dataframe = self.create_slice_menu()

    def create_slice_menu(self):
        if not self.pretraining:
            menu = pd.DataFrame(
                {"data": self.data_file_paths, "label": self.label_file_paths}
            )
        else:
            menu = pd.DataFrame({"data": self.data_file_paths})

        menu["axis"] = [["d"] for _ in range(len(self.data_file_paths))]
        menu["idx"] = [
            [str(s) for s in list(range(self.num_slices))] for _ in range(len(self.data_file_paths))
        ]

        menu = menu.explode("axis")
        menu = menu.explode("idx")
        menu = menu.reset_index(drop=True)

        if not self.cheesecake_factory_mode:
            menu = menu[: self.limit]

        return menu

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, column):
        return self.dataframe[column]

class CustomDataset(Dataset):
    def __init__(
        self,
        root_dir,
        num_slices=300,
        cheesecake_factory_mode=True,
        limit=None,
        data_prefix="",
        label_prefix="",
        pretraining=False,
    ):
        self.root_dir = root_dir
        self.num_slices = num_slices
        self.cheesecake_factory_mode = cheesecake_factory_mode
        self.limit = limit
        self.data_prefix = data_prefix
        self.label_prefix = label_prefix
        self.pretraining = pretraining
        self.slice_menu = SliceServer(
            self.root_dir,
            self.num_slices,
            self.cheesecake_factory_mode,
            self.limit,
            self.data_prefix,
            self.label_prefix,
            self.pretraining,
        )

    def __len__(self):
        return len(self.slice_menu)

    def __getitem__(self, idx):
        data = np.load(self.slice_menu["data"][idx], allow_pickle=True, mmap_mode="r+")
        # data = rescale_volume(data)

        if not self.pretraining:
            label = np.load(self.slice_menu["label"][idx], allow_pickle=True, mmap_mode="r+")
            # label = rescale_volume(label)
#            print(label.shape)
#            print(data.T.shape)
            slice_idx = int(self.slice_menu["idx"][idx])

            if data.shape == (1259, 300, 300) and label.shape == (1259, 300, 300):
                data_slice = data[slice_idx, :, :]
                label_slice = label[slice_idx, :, :]
                data_slice = data_slice[np.newaxis, :, :]
                label_slice = label_slice[np.newaxis, :, :]
                data_tensor = torch.from_numpy(data_slice).float()
                label_tensor = torch.from_numpy(label_slice).float()
            elif data.shape ==(1259, 300, 300) and label.T.shape == (1259, 300, 300):
                label = label.T
                data_slice = data[slice_idx, :, :]
                label_slice = label[slice_idx, :, :]
                data_slice = data_slice[np.newaxis, :, :]
                label_slice = label_slice[np.newaxis, :, :]
                data_tensor = torch.from_numpy(data_slice).float()
                label_tensor = torch.from_numpy(label_slice).float()
           
            else:
                print("Shape mismatch!")
                print(self.slice_menu["label"][idx])
                pass

           
            return data_tensor, label_tensor
      

def save_dataloader_to_h5(dataloader, filepath):
    with h5py.File(filepath, 'w') as h5file:
        for index, (data, label) in enumerate(tqdm(dataloader)):
            if label is not None:
                data = data.detach().cpu().numpy()
                label = label.detach().cpu().numpy()

                if index == 0:
                    data_shape = (len(dataloader),) + data.shape[1:]
                    label_shape = (len(dataloader),) + label.shape[1:]

                    h5file.create_dataset("data", shape=data_shape, dtype=data.dtype, compression=None)
                    h5file.create_dataset("label", shape=label_shape, dtype=label.dtype, compression=None)

                h5file["data"][index] = data
                h5file["label"][index] = label
            else:
                print("not match")
                pass
# Define the base directory for the training dataset
training_dataset = "training_data"

# Define the list of directories (relative to training_dataset)
root_dir = [
    "image-impeccable-train-data-part1", "image-impeccable-train-data-part2",
    "image-impeccable-train-data-part3", "image-impeccable-train-data-part4",
    "image-impeccable-train-data-part5", "image-impeccable-train-data-part6",
    "image-impeccable-train-data-part7", "image-impeccable-train-data-part8",
    "image-impeccable-train-data-part9", "image-impeccable-train-data-part10",
    "image-impeccable-train-data-part11", "image-impeccable-train-data-part12",
    "image-impeccable-train-data-part13", "image-impeccable-train-data-part14",
    "image-impeccable-train-data-part15", "image-impeccable-train-data-part16",
    "image-impeccable-train-data-part17"
]

# Ensure the data_pre directory exists
os.makedirs("h5py", exist_ok=True)

# Iterate over each directory
for i in root_dir:
    # Combine training_dataset path with the directory name
    full_dir = os.path.join(training_dataset, i)

    # Create dataset and dataloader (assuming CustomDataset is defined elsewhere)
    pretrain_dataset = CustomDataset(
        root_dir=full_dir,  # Use the full path here
        num_slices=1259,
        cheesecake_factory_mode=True,
        limit=5,
        data_prefix="seismic_w_noise",
        label_prefix="seismicCubes",
        pretraining=False,
    )

    print(f"Loaded dataset from {full_dir}, length: ", len(pretrain_dataset))

    dataloader = DataLoader(pretrain_dataset, batch_size=1, shuffle=False)

    # Extract base directory name for HDF5 file
    base_dir_name = os.path.basename(full_dir)
    h5_file_name = f"h5py/original_{base_dir_name}.h5"

    # Save dataloader to HDF5 (assuming save_dataloader_to_h5 is defined elsewhere)
    save_dataloader_to_h5(dataloader, h5_file_name)

    print(f"Saved {h5_file_name}")
