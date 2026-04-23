# Data Directory

This directory documents where the dataset should be placed locally. The raw image files are intentionally not committed to this repository.

## Dataset

- Kaggle dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?select=chest_xray)
- Source folder to download from Kaggle: `chest_xray`

## Expected Folder Structure

After downloading and extracting the Kaggle dataset, place the files under `data/chest_xray/` using the existing fixed split:

```text
data/
  README.md
  chest_xray/
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

## Expected Image Counts

The original Kaggle split is expected to contain 5,856 JPEG images:

| Split | NORMAL | PNEUMONIA | Total |
| --- | ---: | ---: | ---: |
| `train/` | 1,341 | 3,875 | 5,216 |
| `val/` | 8 | 8 | 16 |
| `test/` | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

These counts should be checked after downloading the dataset, because Kaggle downloads may include wrapper folders or metadata files that should not be treated as images.

## Example Images

Raw images are not stored in Git. After the dataset is downloaded locally, two representative examples can be inspected from:

- Normal example: `data/chest_xray/train/NORMAL/<example>.jpeg`
- Pneumonia example: `data/chest_xray/train/PNEUMONIA/<example>.jpeg`

If example images are needed for a report or presentation, export small copies separately and document their source, but do not commit them to this repository.

## Version Control Note

Do not commit raw dataset files, extracted images, or generated data artifacts to Git. Only this README and other lightweight documentation/configuration files should be tracked.

## Split Note

The project uses the fixed train/validation/test split provided by the Kaggle dataset. Do not reshuffle the raw data unless an experiment explicitly documents a different split.
