# T-GCN-PyTorch

[![GitHub stars](https://img.shields.io/github/stars/martinwhl/T-GCN-PyTorch?label=stars&maxAge=2592000)](https://gitHub.com/martinwhl/T-GCN-PyTorch/stargazers/) [![issues](https://img.shields.io/github/issues/martinwhl/T-GCN-PyTorch)](https://github.com/martinwhl/T-GCN-PyTorch/issues) [![License](https://img.shields.io/github/license/martinwhl/T-GCN-PyTorch)](./LICENSE) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/martinwhl/T-GCN-PyTorch/graphs/commit-activity) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![Codefactor](https://www.codefactor.io/repository/github/martinwhl/T-GCN-PyTorch/badge)

This is a PyTorch implementation of T-GCN in the following paper: [T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction](https://arxiv.org/abs/1811.05320).

A stable version of this repository can be found at [the official repository](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch).

Note that [the original implementation](https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-TensorFlow) is in TensorFlow, which performs a tiny bit better than this implementation for now.

## Requirements

* numpy
* matplotlib
* pandas
* torch
* lightning>=2.0
* torchmetrics>=0.11

⚠️ The repository is currently based on Lightning 2.0. To use PyTorch Lightning v1.x, please switch to the `pl_v1` branch.

## Model Training

* CLI

    ```bash
    # GCN
    python main.py fit --trainer.max_epochs 3000 --trainer.accelerator cuda --trainer.devices 1 --data.dataset_name losloop --data.batch_size 64 --data.seq_len 12 --data.pre_len 3 --model.model.class_path models.GCN --model.learning_rate 0.001 --model.weight_decay 0 --model.loss mse --model.model.init_args.hidden_dim 100
    # GRU
    python main.py fit --trainer.max_epochs 3000 --trainer.accelerator cuda --trainer.devices 1 --data.dataset_name losloop --data.batch_size 64 --data.seq_len 12 --data.pre_len 3 --model.model.class_path models.GRU --model.learning_rate 0.001 --model.weight_decay 1.5e-3 --model.loss mse --model.model.init_args.hidden_dim 100
    # T-GCN
    python main.py fit --trainer.max_epochs 1500 --trainer.accelerator cuda --trainer.devices 1 --data.dataset_name losloop --data.batch_size 32 --data.seq_len 12 --data.pre_len 3 --model.model.class_path models.TGCN --model.learning_rate 0.001 --model.weight_decay 0 --model.loss mse_with_regularizer --model.model.init_args.hidden_dim 64
    ```

* YAML config file

    ```bash
    # GCN
    python main.py fit --config configs/gcn.yaml
    # GRU
    python main.py fit --config configs/gru.yaml
    # T-GCN
    python main.py fit --config configs/tgcn.yaml
    ```

Please refer to `python main.py fit -h` for more CLI arguments.

Run `tensorboard --logdir ./lightning_logs` to monitor the training progress and view the prediction results.
