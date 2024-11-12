import numpy as np
import glob

def merge_X_data_files(directory_path):
    """
    Merges all X_data files in a specified directory into one single file.

    Returns:
    - merged_X_data: numpy array, merged data from all X_data files.
    """
    file_paths = glob.glob(f"{directory_path}/X_data_*.npy")
    print(file_paths)

    # Initialize an empty list to store the data arrays
    data_list = []

    # Load each file and append it to the list
    for file_path in file_paths:
        data = np.load(file_path)
        data_list.append(data)

    # Concatenate all arrays along the first dimension and save
    merged_X_data = np.concatenate(data_list, axis=0)
    np.save(f"{directory_path}/X_data_merged.npy", merged_X_data)
    return merged_X_data

# Example usage:
merged_data = merge_X_data_files('./data_collection/data')


def merge_y_data_files(directory_path):
    """
    Merges all X_data files in a specified directory into one single file.

    Returns:
    - merged_X_data: numpy array, merged data from all X_data files.
    """
    file_paths = glob.glob(f"{directory_path}/y_data_*.npy")

    # Initialize an empty list to store the data arrays
    data_list = []

    # Load each file and append it to the list
    for file_path in file_paths:
        data = np.load(file_path)
        data_list.append(data)

    # Concatenate all arrays along the first dimension and save
    merged_X_data = np.concatenate(data_list, axis=0)
    np.save(f"{directory_path}/y_data_merged.npy", merged_X_data)
    return merged_X_data

# Example usage:
# merged_data = merge_y_data_files('./data_collection/data')


def merge_img_data_files(directory_path):
    file_paths = glob.glob(f"{directory_path}/img_data_*.npy")
    data_list = []
    for file_path in file_paths:
        data = np.load(file_path)
        data_list.append(data)

    merged_img_data = np.concatenate(data_list, axis=0)
    np.save(f"{directory_path}/img_data_merged.npy", merged_img_data)
    return merged_img_data

# Example usage:
# merged_data = merge_img_data_files('./data_collection/data')
