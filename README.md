 # start of file
# Model Evaluation Framework

This repository contains a framework for evaluating language models on various tasks, including MMLU (Massive Multitask Language Understanding).

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## MMLU Evaluation

The MMLU benchmark covers 57 subjects across STEM, humanities, social sciences, and more, testing models on their knowledge and reasoning abilities.

### Downloading the MMLU Dataset

The dataset will be downloaded automatically when you run the MMLU task for the first time. 
Alternatively, you can manually download it using:

```bash
python scripts/download_mmlu.py --data_dir data
```

### Running Evaluations

To evaluate a model on MMLU:

```bash
python -m val.val --task mmlu --provider openrouter --n 2
```

## Available Tasks

- `add`: Simple addition task for testing
- `mmlu`: Massive Multitask Language Understanding benchmark
- `task`: Generic task template

## Customization

You can create custom evaluation tasks by adding new task modules in the `val/task/` directory.

## License

MIT
