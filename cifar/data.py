# Import third-party libraries
import numpy as np
from torchvision import datasets, transforms
from torchvision import transforms


# CIFAR-10 channel statistics (for standardization)
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD  = [0.2023, 0.1994, 0.2010]


# Function to load dataset for model 7
def _load_dataset_m7(config):
    return _load_dataset_m6(config)

# Function to load dataset for model 6
def _load_dataset_m6(config):
    """
    Loads the CIFAR-10 dataset using per-channel standardization.

    This loader is specialized for model 6 and uses zero-mean, unit-variance
    normalization (instead of naive /255 scaling) to improve convergence.
    Supports light mode for debugging and optional data augmentation.

    Args:
        config (Config): Configuration object with path and toggle settings.

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """

    # Print header for function execution
    print("\n🎯  _load_dataset_m6")

    # Download CIFAR-10 dataset
    train_set = datasets.CIFAR10(root=config.DATA_PATH, train=True, download=True)
    test_set = datasets.CIFAR10(root=config.DATA_PATH, train=False, download=True)

    # Normalize using CIFAR-10 per-channel standardization
    train_data = _standardize(train_set.data)
    test_data = _standardize(test_set.data)

    # Convert labels to numpy arrays
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    # Subsample if LIGHT_MODE is enabled
    if config.LIGHT_MODE:
        train_data = train_data[:1000]
        train_labels = train_labels[:1000]
        test_data = test_data[:200]
        test_labels = test_labels[:200]
    else:
        train_data = train_data[:-5000]
        train_labels = train_labels[:-5000]

    # Apply augmentation if enabled
    if config.AUGMENT_MODE:
        train_data = _augment_dataset(train_data)

    return train_data, train_labels, test_data, test_labels


# Function to standardize CIFAR-10 input images
def _standardize(images):
    """
    Standardizes CIFAR-10 images by scaling to [0, 1], then normalizing each
    channel to zero mean and unit variance using precomputed dataset statistics.

    Args:
        images (np.ndarray): Input image array of shape (N, 32, 32, 3) in uint8 format.

    Returns:
        np.ndarray: Standardized float32 image array with shape (N, 32, 32, 3).
    """

    # Print header for function execution
    print("\n🎯  _standardize")

    images = images.astype(np.float32) / 255.0
    for i in range(3):  # Normalize each RGB channel
        images[..., i] = (images[..., i] - _CIFAR10_MEAN[i]) / _CIFAR10_STD[i]
    return images


# Function to load dataset for model 5
def _load_dataset_m5(config):
    return _load_dataset_m0(config)


# Function to load dataset for model 4
def _load_dataset_m4(config):
    return _load_dataset_m0(config)


# Function to load dataset for model 3
def _load_dataset_m3(config):
    return _load_dataset_m0(config)


# Function to load dataset for model 2
def _load_dataset_m2(config):
    return _load_dataset_m0(config)


# Function to load dataset for model 1
def _load_dataset_m1(config):
    return _load_dataset_m0(config)


# Function to load dataset for model 0
def _load_dataset_m0(config):
    """
    Function to load and preprocess the CIFAR-10 dataset.

    Loads CIFAR-10 from the specified data path, normalizes pixel values,
    and trims the dataset based on LIGHT_MODE flag. Optionally applies
    data augmentation to training samples if augment=True.

    Args:
        config (Config): Configuration object containing DATA_PATH and LIGHT_MODE
        augment (bool): Whether to apply data augmentation on training set

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """

    # Print header for function execution
    print(f"\n🎯  _load_dataset_m0")

    # Download CIFAR-10 dataset
    train_set = datasets.CIFAR10(root=config.DATA_PATH, train=True, download=True)
    test_set = datasets.CIFAR10(root=config.DATA_PATH, train=False, download=True)

    # Normalize pixel values to [0, 1]
    train_data = train_set.data.astype(np.float32) / 255.0
    test_data = test_set.data.astype(np.float32) / 255.0

    # Convert labels to numpy arrays
    train_labels = np.array(train_set.targets)
    test_labels = np.array(test_set.targets)

    # Subsample if LIGHT_MODE is enabled
    if config.LIGHT_MODE:
        train_data = train_data[:1000]
        train_labels = train_labels[:1000]
        test_data = test_data[:200]
        test_labels = test_labels[:200]
    else:
        # Exclude last 5000 training samples
        train_data = train_data[:-5000]
        train_labels = train_labels[:-5000]

    # Apply augmentation if enabled
    if config.AUGMENT_MODE:
        train_data = _augment_dataset(train_data)

    # Return processed dataset
    return train_data, train_labels, test_data, test_labels


# Function to dispatch dataset loader by model number
def dispatch_load_dataset(model_number, config):
    """
    Function to route dataset loading based on the model variant number.

    Dynamically constructs the appropriate function name (e.g., load_dataset_m2)
    and invokes it using the global namespace. Raises ValueError if the function
    is not defined.

    Args:
        model_number (int): Model identifier (e.g., 0 to 5)
        config (Config): Configuration object to pass to the loader

    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """

    # Print header for function execution
    print("\n🎯  dispatch_load_dataset")

    try:
        # Dynamically resolve the dataset loader function by model number
        loader_fn = globals()[f"_load_dataset_m{model_number}"]
        return loader_fn(config)  # Return loaded dataset from resolved function

    except KeyError:
        # Raise clear error if the loader function is not defined
        raise ValueError(f"❌ ValueError from data.py at dispatch_load_dataset():\nmodel_number={model_number}\n")


# Function to apply torchvision-style augmentations to a NumPy image batch
def _augment_dataset(images):
    """
    Applies CIFAR-10 style augmentation using torchvision.transforms.

    Augmentations:
    - Random crop with padding
    - Random horizontal flip
    - Color jitter (brightness and contrast)

    Args:
        images (np.ndarray): Array of images (float32 in [0, 1], shape: [N, 32, 32, 3])

    Returns:
        np.ndarray: Augmented images (same shape)
    """

    # Print header for function execution
    print("\n🎯  _augment_dataset")

    # Define CIFAR-10 augmentation pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),                              # Convert to PIL for torchvision compatibility
        transforms.RandomCrop(32, padding=4),                 # Random crop with padding
        transforms.RandomHorizontalFlip(),                    # Random horizontal flip
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Slight brightness and contrast jitter
        transforms.ToTensor()                                 # Convert back to tensor [C, H, W]
    ])

    # Apply augmentation to each image and restore [H, W, C] format
    augmented = [transform(img).permute(1, 2, 0).numpy() for img in images]

    # Return stacked NumPy array with same shape as input
    return np.stack(augmented, axis=0)


# Print module successfully executed
print("\n✅  data.py successfully executed")
