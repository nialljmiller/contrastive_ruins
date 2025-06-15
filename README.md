# contrastive_ruins

This project now uses **PyTorch** for training a Siamese network on DEM patches.

## Setup
Install Python packages using:

```bash
pip install -r requirements.txt
```

## Running
The main pipeline can be executed with:

```bash
python main.py --mode full --data_path <DATA_DIR> --output_dir <OUTPUT_DIR>
```

Use `--mode train` to only train the model or `--mode detect` to run detection with existing weights.

