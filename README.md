# AdaProb

Machine unlearning for deep neural networks with support for knowledge distillation, selective forgetting, and membership inference evaluation.

## Repository Structure

```
.
├── main.py                  # Training entry point
├── main_merged.py           # Alternative training script with interclass confusion support
├── models.py                # Model architectures (AllCNN, ResNet, etc.)
├── datasets.py              # Dataset loaders (binary classification)
├── datasets_multiclass.py   # Dataset loaders (multiclass classification)
├── loops.py                 # Training/validation loops
├── layers.py                # Custom layers
├── logger.py                # Training logger
├── lacuna.py                # Lacuna dataset classes
├── image.py                 # Image visualization utilities
├── create_lacuna.py         # Script to generate Lacuna datasets
├── convert_parameters.py    # Parameter conversion utilities
├── refine_prediction.py     # Prediction refinement utilities
├── notebooks/               # Experiment notebooks
│   ├── small_scale_unlearning*.ipynb
│   ├── large_scale_unlearning*.ipynb
│   ├── large_scale_ictest*.ipynb
│   ├── BadT_experiments.ipynb
│   ├── MIA_experiments.ipynb
│   └── graph.ipynb
└── LICENSE
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --dataset <dataset> --model <model> [options]
```

See `python main.py --help` for all available options.

## License

Academic Software License -- non-commercial use only. See [LICENSE](LICENSE) for details.
