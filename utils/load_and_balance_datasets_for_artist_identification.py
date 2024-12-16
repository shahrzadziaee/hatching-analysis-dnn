import numpy as np
import h5py
import argparse

def load_balance_datasets(dataset_paths, class_labels=None, output_file=None, seed=42):
    """
    Load multiple HDF5 datasets, assign class labels (explicit or automatic), balance the classes, and optionally save the combined dataset.
    
    Args:
        dataset_paths (list of str): List of paths to the datasets.
        class_labels (list of int or None): List of class labels. If None, labels are assigned automatically (default: None).
        output_dir (str or None): Directory to save the balanced HDF5 dataset. If None, the dataset is not saved (default: None).
        seed (int): Random seed for reproducibility (default: 42).
        
    Returns:
        Tuple (balanced_images, balanced_labels): Balanced datasets with image patches and their corresponding labels.
    """
    # Automatically assign class labels if not provided
    if class_labels is None:
        class_labels = list(range(len(dataset_paths)))
    
    if len(dataset_paths) != len(class_labels):
        raise ValueError("The number of datasets and class labels must match.")
    
    # Step 1: Determine the minimum size of the datasets
    min_size = float('inf')
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, 'r') as f:
            min_size = min(min_size, len(f['images']))
    
    print(f"Balancing all datasets to size: {min_size} per class")
    
    # Step 2: Load, sample, and balance the datasets
    np.random.seed(seed)
    balanced_images = []
    balanced_labels = []
    
    for dataset_path, class_label in zip(dataset_paths, class_labels):
        with h5py.File(dataset_path, 'r') as f:
            images = f['images'][:]
            labels = np.full(len(images), class_label)
            
            # Shuffle and sample to balance the dataset
            indices = np.random.choice(len(images), min_size, replace=False)
            balanced_images.append(images[indices])
            balanced_labels.append(labels[indices])
    
    # Combine all balanced datasets
    balanced_images = np.vstack(balanced_images)
    balanced_labels = np.concatenate(balanced_labels)
    
    # Shuffle the combined dataset
    combined_indices = np.random.permutation(len(balanced_images))
    balanced_images = balanced_images[combined_indices]
    balanced_labels = balanced_labels[combined_indices]
    
    # Optionally save the combined dataset to an HDF5 file
    if output_file:
        with h5py.File(output_file, 'w') as hdf5_file:
            hdf5_file.create_dataset('images', data=balanced_images)
            hdf5_file.create_dataset('labels', data=balanced_labels)
        print(f"Balanced dataset with {len(class_labels)} classes saved to {output_file}")
    
    # Return the balanced dataset
    return balanced_images, balanced_labels

def main():
    parser = argparse.ArgumentParser(description="load class datasets and return a class-balanced dataset.")
    parser.add_argument('--dataset_paths', type=str, nargs='+', required=True, help="List of paths to the HDF5 datasets for classes")
    parser.add_argument('--class_labels', type=int, nargs='+', help="List of class labels corresponding to the datasets")
    parser.add_argument("--output_dir", type = str, default=None, help="Directory for saving the balanced dataset.")
    args = parser.parse_args()

    load_balance_datasets(args.dataset_paths, args.class_labels, args.output_dir+'merged_balanced_datasets.h5')
if __name__=='__main__':
    main()

