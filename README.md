# UK Flowers

[![check.yml](https://github.com/psaw/uk-flowers/actions/workflows/check.yml/badge.svg)](https://github.com/psaw/uk-flowers/actions/workflows/check.yml)
[![publish.yml](https://github.com/psaw/uk-flowers/actions/workflows/publish.yml/badge.svg)](https://github.com/psaw/uk-flowers/actions/workflows/publish.yml)
[![Documentation](https://img.shields.io/badge/documentation-available-brightgreen.svg)](https://psaw.github.io/uk-flowers/)
[![License](https://img.shields.io/github/license/psaw/uk-flowers)](https://github.com/psaw/uk-flowers/blob/main/LICENSE.txt)
[![Release](https://img.shields.io/github/v/release/psaw/uk-flowers)](https://github.com/psaw/uk-flowers/releases)

UK flowers classification with neural networks.

This project implements a deep learning solution for classifying UK flower species. It includes a complete pipeline for data loading, preprocessing, model training, and inference, utilizing modern MLOps tools and best practices.

## Introduction

This project is inspired by [Kaggle Competition](https://www.kaggle.com/competitions/oxford-102-flower-pytorch/overview).
The goal of this project is to implement a deep learning solution for classifying UK flower species based on the original "[102 category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)" (by Visual Geometry Group - University of Oxford).
The solution is to fine-tune ResNet to accomodate 102 classes.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and [DVC](https://dvc.org/) for data version control.

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) installed
- [DVC](https://dvc.org/) installed (usually installed via project dependencies, but good to have)
- AWS credentials configured (if accessing private S3 buckets, though this project uses a public read bucket)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/psaw/uk-flowers.git
    cd uk-flowers
    ```

2. Install dependencies and set up the environment using `uv`:

    ```bash
    uv sync
    ```

3. Install pre-commit hooks to ensure code quality:

    ```bash
    uv run pre-commit install
    ```

4. (Optional) Verify installation by running checks:

    ```bash
    uv run pre-commit run -a
    ```

## Usage

The project provides a CLI tool `uk-flowers` to manage training and inference.

### Data Management

Data is managed using DVC and stored in an S3 bucket. The training pipeline automatically attempts to pull data using `dvc pull` if it's missing locally.

To manually pull data:
```bash
uv run dvc pull
```

### Training

Make sure to start MLfLow server first

```bash
uv run just mlflow
```

To train the model, use the `train` command. This will:

1.  Download the dataset (if not present).
2.  Preprocess the data.
3.  Train a ResNet18 (configurable) model using PyTorch Lightning.
4.  Log metrics and artifacts to MLFlow.

```bash
uv run uk-flowers train
```

You can override configuration parameters using Hydra syntax. For example, to change the number of epochs or batch size:

```bash
uv run uk-flowers train "train.epochs=5" "data.batch_size=32"
```

### Inference

To run inference on an image or a directory of images, use the `infer` command. You must specify the path to the image(s) via the `inference.image_path` override.

```bash
uv run uk-flowers infer "inference.image_path=path/to/image.jpg"
```

Or for a directory:

```bash
uv run uk-flowers infer "inference.image_path=path/to/images_dir/"
```

By default, it uses the best checkpoint found in `outputs/`. You can specify a specific checkpoint:

```bash
uv run uk-flowers infer "inference.image_path=image.jpg" "inference.checkpoint_path=path/to/model.ckpt"
```

Results will be saved to `inference_results.json` by default.

### MLFlow

MLFlow is used for experiment tracking. To view the logs, ensure the MLFlow server is running (default: `http://127.0.0.1:8080`).

You can start a local MLFlow server using:
```bash
uv run just mlflow
```
(or `uv run mlflow server --host 127.0.0.1 --port 8080`)

## Project Structure

-   `src/uk_flowers`: Source code for the package.
    -   `data/`: Data loading and processing (DataModule, Dataset).
    -   `model/`: Model definition (LightningModule).
    -   `utils/`: Utility functions.
    -   `scripts.py`: CLI entry points.
-   `confs/`: Hydra configuration files.
-   `notebooks/`: Jupyter notebooks for exploration.
-   `tests/`: Unit tests.
-   `pyproject.toml`: Project configuration and dependencies.
-   `dvc.yaml` / `.dvc/`: DVC configuration.

## Technologies

-   **Language**: Python
-   **Dependency Management**: uv
-   **DL Framework**: PyTorch, PyTorch Lightning
-   **Configuration**: Hydra
-   **Data Versioning**: DVC (S3 backend)
-   **Experiment Tracking**: MLFlow
-   **Code Quality**: pre-commit, ruff, mypy
-   **CLI**: Fire, Just
