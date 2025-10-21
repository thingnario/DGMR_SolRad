---
license: mit
tags:
- solar-radiation
- deep-learning
- nowcasting
- dgmr
---

# DGMR Solar Radiation Nowcasting ☀️

A deep learning model for solar radiation nowcasting using modified [Deep Generative Model of Rainfall (DGMR)](https://www.nature.com/articles/s41586-021-03854-z) architecture with Solar radiation Output (DGMR-SO). The model predicts clearsky index and converts it to solar radiation for up to 36 time steps ahead.

![Solar Prediction Example](docs/srad_example.gif)

## Overview

This repository implements two model variants for solar radiation forecasting:
- **DGMR_SO**: Full Deep Generative Models with one generator and two discriminators during the training stage
- **Generator_only**: Only one generator during the training stage

The model uses multiple input sources:
- **Himawari satellite data**: Clearsky index calculated from Himawari satellite data
- **WRF Prediction**: Clearsky index from WRF's solar irradiation prediction
- **Topography**: Static topographical features
- **Time features**: Temporal sin/cos encoding for day and hour

## Installation

1. Clone the repository & install Git LFS:
```bash
git lfs install
git clone <repository-url>
cd DGMR_SolRad
git lfs pull
git lfs ls-files # confirm whether models weights & sample data are downloaded
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.x
- PyTorch 2.4.0
- NumPy 1.26.4
- einops 0.8.0

## Usage

### Basic Inference

Run solar radiation prediction using the pre-trained models:

```bash
python inference.py --model-type DGMR_SO --basetime 202504131100
```

### Command Line Arguments

- `--model-type`: Choose between `DGMR_SO` or `Generator_only` (default: `DGMR_SO`)
- `--basetime`: Timestamp for input data in format YYYYMMDDHHMM (default: `202504131100`)

### Example

```bash
# Using DGMR_SO model
python inference.py --model-type DGMR_SO --basetime 202504131100

# Using Generator-only model
python inference.py --model-type Generator_only --basetime 202507151200
```

## Sample Data

The repository includes sample data files:
- `sample_202504131100.npz`
- `sample_202504161200.npz`
- `sample_202507151200.npz`

## Model Weights

Pre-trained weights are available for both models:
- `model_weights/DGMR_SO/ft36/weights.ckpt`
- `model_weights/Generator_only/ft36/weights.ckpt`

## License

This project is released under the MIT License.
