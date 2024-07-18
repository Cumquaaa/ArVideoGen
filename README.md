# Autoregressive Video Generation

## Overview
This project implements an autoregressive video generation model based on the paper [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/pdf/2406.11838).

## Progress

- [ ] Implement dummy version of Gneration Model
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
python train.py
```

### Sampling
To generate samples from the trained model, run:

```bash
python sample.py
```


### Evaluation
To evaluate the model, run:

```bash
python evaluate.py (Not Implemented Yet)
```

## Project Structure

```shell
autoregressive_image_generation/
├── data/
│ └── prepare_data.py # Scripts for data preprocessing
├── models/
│ ├── init.py # Initialize the models module
│ ├── autoregressive.py # Autoregressive model definition
│ └── diffusion.py # Denoising diffusion network definition
├── utils/
│ ├── init.py # Initialize the utils module
│ ├── losses.py # Implementation of dummy loss function
│ ├── sampling.py # Sampling methods for inference
│ └── metrics.py # Dummy evaluation metrics
├── train.py # Main training script
├── sample.py # Script for generating samples
├── evaluate.py # Script for evaluating the model
├── requirements.txt # Required libraries
└── README.md # Project documentation
```