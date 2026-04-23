# Data Pipeline Notes

This document explains the purpose of the data-loading steps implemented in `src/pneumonia_classifier/data.py`.

## Goal

The data pipeline converts the local Kaggle folder structure into PyTorch-ready batches for training, validation, and test evaluation.

Expected local dataset root:

```text
data/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

Raw image files are kept locally and are not committed to Git.

## Configuration Source

The pipeline reads its settings from `configs/default.yaml`.

Important fields:

- `data.root_dir`: dataset root path.
- `data.image_size`: target image size, currently `224`.
- `data.class_names`: expected class folders, currently `NORMAL` and `PNEUMONIA`.
- `preprocessing.normalize_mean`: ImageNet mean used for normalization.
- `preprocessing.normalize_std`: ImageNet standard deviation used for normalization.
- `preprocessing.augmentation`: training-only augmentation settings.
- `training.batch_size`: batch size used by the dataloaders.

## Step 1: Validate Dataset Structure

Before creating datasets, the pipeline checks that the expected root, split folders, and class folders exist.

This prevents silent mistakes such as:

- extracting the Kaggle download into an extra wrapper folder;
- missing the validation or test split;
- misspelling `NORMAL` or `PNEUMONIA`;
- pointing the config to the wrong dataset location.

If a required folder is missing, the code raises a clear `FileNotFoundError`.

## Step 2: Build Image Transforms

Images are transformed before they reach the model.

All splits use:

- resize to `224 x 224`;
- conversion to 3-channel grayscale/RGB-compatible tensors;
- tensor conversion;
- ImageNet normalization.

The training split additionally uses light augmentation when enabled:

- random horizontal flip;
- small random rotation.

Validation and test transforms are deterministic so evaluation remains reproducible.

## Step 3: Create ImageFolder Datasets

The pipeline uses `torchvision.datasets.ImageFolder`.

Each split becomes one dataset:

- `train`;
- `val`;
- `test`.

Labels are inferred from class folder names. The expected class folders are defined in the config.

## Step 4: Create Dataloaders

`get_dataloaders(config)` wraps the datasets in PyTorch `DataLoader` objects.

Behavior:

- training loader uses `shuffle=True`;
- validation and test loaders use `shuffle=False`;
- all loaders use the configured batch size.

The training loop will consume these loaders as batches of:

```text
images: [batch_size, 3, 224, 224]
labels: [batch_size]
```

## Step 5: Count Classes

`get_class_counts(dataset_root)` counts image files per class and per split.

This is useful for:

- confirming that the dataset was extracted correctly;
- documenting class imbalance;
- filling dataset checks in notebooks and reports.

Expected counts for the original Kaggle split:

| Split | NORMAL | PNEUMONIA | Total |
| --- | ---: | ---: | ---: |
| `train` | 1,341 | 3,875 | 5,216 |
| `val` | 8 | 8 | 16 |
| `test` | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

## Notes for Later Modules

- Models should expect logits-compatible targets for binary classification.
- Training code should convert labels to float tensors before using `BCEWithLogitsLoss`.
- Evaluation code should apply sigmoid to logits before thresholding probabilities.
- Notebook code should call these helpers instead of duplicating dataset logic.
