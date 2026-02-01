import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset


def argument_check(args):
    """
    Validate command-line arguments for consistency.

    Args:
        args: Parsed command-line arguments

    Raises:
        ValueError: If arguments are inconsistent or invalid
    """
    if args.method == "elora":
        if args.lora_dict_directory is None:
            raise ValueError(
                "For EigenLoRA, --lora_dict_directory must be specified "
                "to provide the LoRA checkpoints for eigenvector computation."
            )
        if args.sampled_subsets_path is None:
            raise ValueError(
                "For EigenLoRA and VeRA, --sampled_subsets_path must be specified "
                "to ensure consistent class subsets across methods."
            )

    if args.method == "vera":
        if args.sampled_subsets_path is None:
            raise ValueError(
                "For VeRA, --sampled_subsets_path must be specified "
                "to ensure consistent class subsets with LoRA experiments."
            )

    if args.subset_size <= 0:
        raise ValueError("--subset_size must be a positive integer.")

    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer.")

    if args.lr <= 0:
        raise ValueError("--lr (learning rate) must be positive.")


def dataloader_from_subset(
    train_set,
    test_set,
    class_indices,
    experiment_folder,
    subset_index,
    batch_size=128,
    num_workers=4,
):
    """
    Create train and test DataLoaders for a subset of classes.

    This function filters the dataset to only include samples from the specified
    classes and creates a label mapping from original class indices to 0-indexed
    subset labels.

    Args:
        train_set: Full training dataset
        test_set: Full test dataset
        class_indices: List of class indices to include in this subset
        experiment_folder: Path to save label mappings
        subset_index: Index of this subset (1-indexed)
        batch_size: Batch size for DataLoaders (default: 128)
        num_workers: Number of worker processes for data loading (default: 4)

    Returns:
        tuple: (train_loader, test_loader) DataLoader objects for the subset
    """
    # Create label mapping: original_class_idx -> subset_class_idx (0-indexed)
    label_mapping = {
        orig_idx: new_idx for new_idx, orig_idx in enumerate(class_indices)
    }

    # Save label mapping for later use (e.g., inference)
    mapping_path = os.path.join(
        experiment_folder, "label_mappings", f"subset_{subset_index}_mapping.json"
    )
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=2)

    # Filter train set
    train_indices = []
    for idx in range(len(train_set)):
        try:
            _, label = train_set[idx]
        except Exception:
            # Handle datasets that return more than 2 values
            sample = train_set[idx]
            label = sample[1] if isinstance(sample, (tuple, list)) else sample["label"]

        if label in class_indices:
            train_indices.append(idx)

    # Filter test set
    test_indices = []
    for idx in range(len(test_set)):
        try:
            _, label = test_set[idx]
        except Exception:
            sample = test_set[idx]
            label = sample[1] if isinstance(sample, (tuple, list)) else sample["label"]

        if label in class_indices:
            test_indices.append(idx)

    # Create subset datasets with remapped labels
    class RemappedSubset(Subset):
        def __init__(self, dataset, indices, label_map):
            super().__init__(dataset, indices)
            self.label_map = label_map

        def __getitem__(self, idx):
            sample = super().__getitem__(idx)
            if isinstance(sample, (tuple, list)):
                img, label = sample[0], sample[1]
                return img, self.label_map[label]
            return sample

    train_subset = RemappedSubset(train_set, train_indices, label_mapping)
    test_subset = RemappedSubset(test_set, test_indices, label_mapping)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"Subset {subset_index}: {len(train_subset)} train, {len(test_subset)} test samples"
    )
    print(f"Classes: {class_indices} -> mapped to [0, {len(class_indices)-1}]")

    return train_loader, test_loader


def consolidate_loras(lora_directory, holdout_lora=None):
    """
    Load and consolidate LoRA weight matrices from multiple checkpoints.

    This function loads LoRA-trained model checkpoints and extracts the
    LoRA A and B matrices, stacking them for PCA/eigenvector computation.

    Args:
        lora_directory: Path to directory containing LoRA checkpoint files
        holdout_lora: Index of LoRA to exclude (for leave-one-out evaluation)

    Returns:
        dict: Dictionary mapping layer names to consolidated weight tensors
              Each tensor has shape (num_loras, original_shape...)
    """
    checkpoint_files = sorted(
        [
            f
            for f in os.listdir(lora_directory)
            if f.endswith(".pth")
            and (f.startswith("model_") or f.startswith("subset_"))
        ]
    )

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {lora_directory}")

    consolidated = {}

    for idx, ckpt_file in enumerate(checkpoint_files):
        if holdout_lora is not None and idx == holdout_lora:
            print(f"Holding out checkpoint: {ckpt_file}")
            continue

        ckpt_path = os.path.join(lora_directory, ckpt_file)
        state_dict = torch.load(ckpt_path, map_location="cpu")

        for key, value in state_dict.items():
            # Filter for LoRA-specific weights
            if "lora_A" in key or "lora_B" in key:
                if key not in consolidated:
                    consolidated[key] = []
                consolidated[key].append(value)

    # Stack tensors along new dimension
    for key in consolidated:
        consolidated[key] = torch.stack(consolidated[key], dim=0)

    print(
        f"Consolidated {len(checkpoint_files) - (1 if holdout_lora is not None else 0)} LoRA checkpoints"
    )
    print(f"Extracted {len(consolidated)} LoRA weight tensors")

    return consolidated


def get_eigenvectors(consolidated_lora_dict, num_components=None):
    """
    Compute eigenvectors (principal components) from consolidated LoRA weights.

    Performs PCA on the flattened LoRA weight matrices to find the principal
    directions of variation across different fine-tuned models.

    Args:
        consolidated_lora_dict: Dictionary from consolidate_loras()
        num_components: Number of principal components to keep (default: all)

    Returns:
        dict: Dictionary mapping layer names to eigenvector tensors
              Each tensor has shape (num_components, flattened_weight_size)
    """
    eigenvectors = {}

    for key, stacked_weights in consolidated_lora_dict.items():
        # Flatten each LoRA's weights: (num_loras, ...) -> (num_loras, -1)
        num_loras = stacked_weights.shape[0]
        flattened = stacked_weights.reshape(num_loras, -1).float()

        # Center the data
        mean = flattened.mean(dim=0, keepdim=True)
        centered = flattened - mean

        # Compute covariance and eigenvectors
        # Using SVD for numerical stability
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        # Vh contains the right singular vectors (principal components)
        n_components = (
            num_components if num_components else min(num_loras, flattened.shape[1])
        )
        eigenvectors[key] = {
            "components": Vh[:n_components],  # (n_components, flattened_size)
            "mean": mean.squeeze(0),
            "singular_values": S[:n_components],
            "original_shape": stacked_weights.shape[1:],
        }

    print(f"Computed eigenvectors for {len(eigenvectors)} layers")

    return eigenvectors


def calculate_initial_parameters(
    eigenvectors, lora_state_dict, num_components=8, rank=1
):
    """
    Calculate initial EigenLoRA parameters from a LoRA checkpoint.

    Projects the LoRA weights onto the eigenvector basis to get the initial
    loadings (coefficients) for EigenLoRA training.

    Args:
        eigenvectors: Dictionary from get_eigenvectors()
        lora_state_dict: State dict of a LoRA-trained model
        num_components: Number of eigenvector components to use
        rank: Rank parameter for EigenLoRA (affects loading shape)

    Returns:
        dict: State dict with initialized EigenLoRA parameters
    """
    init_params = {}

    for key, value in lora_state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            if key in eigenvectors:
                eig_info = eigenvectors[key]
                components = eig_info["components"][:num_components]
                mean = eig_info["mean"]
                original_shape = eig_info["original_shape"]

                # Flatten the LoRA weight
                flattened = value.reshape(-1).float()

                # Project onto eigenvector basis: loadings = (weight - mean) @ components.T
                centered = flattened - mean
                loadings = centered @ components.T  # (num_components,)

                # Store as EigenLoRA loading parameters
                # Reshape loadings based on rank if needed
                loading_key = key.replace("lora_A", "eigenlora_loading_A")
                loading_key = loading_key.replace("lora_B", "eigenlora_loading_B")
                init_params[loading_key] = loadings.reshape(num_components, rank)

                # Store components (these are typically frozen)
                component_key = key.replace("lora_A", "eigenlora_component_A")
                component_key = component_key.replace("lora_B", "eigenlora_component_B")
                init_params[component_key] = components.reshape(
                    num_components, *original_shape
                )
        else:
            # Copy non-LoRA parameters as-is
            init_params[key] = value

    print(
        f"Initialized {len([k for k in init_params if 'eigenlora_loading' in k])} EigenLoRA loading parameters"
    )

    return init_params
