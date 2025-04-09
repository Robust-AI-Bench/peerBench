 # start of file
# MMLU Task Evaluation

This repository contains code for evaluating language models on the Massive Multitask Language Understanding (MMLU) benchmark.

## Overview

MMLU covers 57 subjects across STEM, humanities, social sciences, and more, testing models on their knowledge and reasoning abilities.

## Structure

- `mmlu/task.py`: Main task definition for MMLU evaluation
- `mmlu/categories.py`: Contains subject categorization information
- `mmlu/evaluate.py`: Script for evaluating models using OpenAI API
- `mmlu/evaluate_flan.py`: Script for evaluating FLAN-T5 models

## Usage

To run an evaluation:

```bash
python -m mmlu.evaluate --ntrain 5 --data_dir data --save_dir results --engine davinci
```

For FLAN-T5 models:

```bash
python -m mmlu.evaluate_flan --ntrain 5 --data_dir data --save_dir results --model google/flan-t5-small
```

## Data Format

The MMLU data should be organized in the following structure:
```
data/
  dev/
    abstract_algebra_dev.csv
    ...
  test/
    abstract_algebra_test.csv
    ...
```

Each CSV file contains questions with multiple-choice options and the correct answer.
