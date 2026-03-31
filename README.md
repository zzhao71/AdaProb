# AdaProb: Efficient Machine Unlearning via Adaptive Probability

Machine unlearning for deep neural networks with support for selective forgetting and membership inference evaluation.

## Repository Structure

```
.
├── main.py
├── main_merged.py
├── models.py
├── datasets.py
├── datasets_multiclass.py
├── loops.py
├── layers.py
├── logger.py
├── lacuna.py
├── image.py
├── create_lacuna.py
├── convert_parameters.py
├── refine_prediction.py
├── notebooks/
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

## Acknowledgements

This project uses code from [SCRUB](https://github.com/meghdadk/SCRUB).

## License

Academic Software License -- non-commercial use only. See [LICENSE](LICENSE) for details.
