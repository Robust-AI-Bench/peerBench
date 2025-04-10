
# Val - Decentralized Model Evaluation Framework

Val is a powerful, flexible framework for evaluating and benchmarking language models across various tasks. It provides a simple, unified interface to test models from different providers and track their performance.

## Features

- **Multi-provider support**: Test models from OpenRouter and other providers
- **Customizable tasks**: Built-in support for MMLU and other benchmarks, with an extensible task system
- **Parallel evaluation**: Efficiently evaluate multiple models in parallel
- **Result storage**: Automatically store and retrieve evaluation results
- **Cryptographic verification**: Sign and verify evaluation results with ECDSA keys

## Installation

```bash
# Install from PyPI
pip install val

# Or install from source
git clone https://github.com/val-ai/val.git
cd val
pip install -e .
```

## Quick Start

```python
from val import Val

# Run a quick evaluation on MMLU
results = Val.run(
    task='mmlu',
    n_samples=10,
    models=[
        'meta-llama/llama-4-maverick',
        'anthropic/claude-3.7-sonnet',
        'qwen/qwen-2.5-7b-instruct'
    ]
)

print(results)
```

## Command Line Interface

Val includes a convenient CLI for running evaluations:

```bash
# Evaluate models on MMLU
python -m val.val --task mmlu --provider openrouter --n_models 3 --n_samples 10

# View available tasks
python -m val.val tasks
```

## Core Concepts

### Tasks

Tasks define the evaluation criteria. Each task provides:
- Sample generation
- Scoring logic
- Result aggregation

Built-in tasks include:
- `mmlu`: Massive Multitask Language Understanding benchmark
- `math500`: Mathematical reasoning evaluation

### Models

Val supports various model providers, with OpenRouter as the default. You can specify:
- Which models to evaluate
- How many samples to test
- Batch size for parallel evaluation

### Results

Evaluation results include:
- Score: Performance metric (task-specific)
- Time delta: Execution time
- Sample information: Test case details
- Cryptographic verification: Signed results for verification

## Advanced Usage

### Custom Tasks

Create your own evaluation tasks by extending the `Task` class:

```python
from val.task.task import Task

class MyCustomTask(Task):
    description = "Tests model ability to solve custom problems"
    
    def sample(self, idx=None, sample=None):
        # Generate or return sample
        pass
        
    def score(self, data):
        # Score the model's response
        pass
```

### Background Evaluation

Run evaluations in the background:

```python
val = Val(
    task='mmlu',
    n_models=3,
    background=True,
    tempo=3600  # Run every hour
)
```

## Development

- **Code style**: Follow PEP 8 guidelines
- **Testing**: Run tests with pytest
- **Contributing**: See CONTRIBUTING.md for guidelines

## License

MIT License - See LICENSE file for details

## Community

- GitHub: [https://github.com/val-ai/val](https://github.com/val-ai/val)
- Discord: [Join our community](https://discord.gg/val-ai-941362322000203776)
