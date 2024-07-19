# Autoregressive Video Generation

## Overview
This project implements an autoregressive video generation model based on the paper [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/pdf/2406.11838).

## Progress

- [X] Implement trainable dummy version of Gneration Model (No performance yet!)
- [ ] Implement Image Gneration Model
- [ ] Implement Video Gneration Model


## Setup
1. Clone the repository.
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

### Training
To train the model, run:

```bash
bash entry.sh --task train --model_config small
```

### Evaluation
To evaluate the model, run:

```bash
bash entry.sh --task evaluate --model_config small
```

## Project Structure

```shell
autoregressive_image_generation/
├── ckpts/ # Folder for storing checkpoints
├── data/
│ └── prepare_data.py # Scripts for data preprocessing
├── logs/ # Folder for training loss plots
├── model_configs/
│ └── small.json # Config for dummy version
├── models/
│ ├── autoregressive.py # Autoregressive model definition
│ └── diffusion.py # Denoising diffusion network definition
├── utils/
│ ├── args.py # Script for parsing args
│ ├── metrics.py # Dummy evaluation metrics
│ ├── plot.py # Script for plotting loss
│ └── sampling.py # Sampling methods for inference
├── entry.sh # Entry for all scripts with args processsing
├── train.py # Main training script
├── sample.py # Script for generating samples
├── evaluate.py # Script for evaluating the model
├── requirements.txt # Required libraries
└── README.md # Project documentation
```