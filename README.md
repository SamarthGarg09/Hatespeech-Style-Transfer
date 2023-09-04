# Hatespeech Style Transfer

This repository is dedicated to the task of style transfer, specifically focusing on transferring the style of hate speech text.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Files and Directories](#files-and-directories)
- [Contributing](#contributing)
- [License](#license)

## Overview
The main goal of this project is to understand and implement style transfer techniques for hate speech. By doing so, we aim to provide tools and methodologies that can be used to combat online hate speech and its negative impacts.

## Installation
To get started with this project, clone the repository and install the necessary dependencies.
```bash
git clone https://github.com/SamarthGarg09/Hatespeech-Style-Transfer.git
cd Hatespeech-Style-Transfer
pip install -r requirements.txt
```
## Usage
To train the model, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```
For evaluation, run:
```bash
python evaluate.py
```
## Files and Directories
- `main.py`: Main entry point of the application.
- `train.py`: Contains the training loop and related functionalities.
- `evaluator.py`: Used for evaluating the trained models.
- `data.py`: Data loading and preprocessing utilities.
- `utils.py`: General utility functions.
- `preprocess.py`: Preprocessing scripts for the dataset.
- `_transformers`: Directory containing transformer-based model components.
  - `attention.py`: Attention mechanisms for the transformer.
  - `decoder.py`: Decoder part of the transformer.
  - `encoder.py`: Encoder part of the transformer.
  - `utils.py`: Utility functions for the transformer.
- `config.yml`: Configuration file for the project.

## Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is open-source and available under the MIT License.