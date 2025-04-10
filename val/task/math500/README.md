
# MATH-500 Multiple Choice Task

This task converts math problems from the HuggingFace MATH-500 dataset into multiple choice questions. The implementation randomly selects incorrect options from other problems in the dataset to create challenging multiple choice questions.

## Features

- Converts open-ended math problems into multiple choice format
- Configurable number of choices per question
- Randomized selection of distractor options from other problems
- Simple scoring mechanism for model evaluation

## Usage

```python
from task import Task

# Initialize the task with default parameters
task = Task()

# Get a random multiple choice problem
problem = task.sample()

# To evaluate a model
result = task.forward(model)
```

## Configuration

You can configure the task by passing parameters:

```python
# Create a task with 5 choices per question
task = Task(params={'num_choices': 5})
```

## Dataset

This task uses the [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) dataset, which contains challenging math problems across various topics.
