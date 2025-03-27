# deval: Decentralized Evaluation Framework

deval is a robust, scalable system designed to evaluate AI model performance across distributed networks. By leveraging cryptographic verification and standardized benchmarking, deval enables transparent, secure, and reproducible model evaluation.

## Features

- Cryptographically signed evaluation results
- Pluggable task modules for diverse evaluation scenarios
- Distributed evaluation across multiple models
- Persistent storage of evaluation results
- Configurable evaluation parameters
- Support for various model providers

## Installation

```bash
git clone https://github.com/yourusername/deval.git
cd deval
pip install -e .
```

## Quick Start

```python
from deval import deval

# Initialize the evaluator
evaluator = deval(
    task='add',              # The evaluation task
    provider='model.openrouter',  # The model provider
    batch_size=64,           # Parallel evaluation batch size
    n=64,                    # Number of models to evaluate
    timeout=4                # Timeout per evaluation
)

# Run an evaluation epoch
results = evaluator.epoch()

# Display results
print(results)
```

## Components

### Authentication System (JWT)

The JWT authentication system provides cryptographic verification for evaluation results:

```python
from deval.auth import JWT

# Create a JWT instance
jwt = JWT()

# Generate a token
token = jwt.get_token(data={'task': 'add', 'score': 0.95})

# Verify a token
decoded = jwt.verify_token(token)
```

### Storage System

The storage system provides persistent storage of evaluation results:

```python
from deval.storage import Storage

# Create a storage instance
storage = Storage('~/.deval/results')

# Store data
storage.put('model1/result1.json', {'score': 0.95})

# Retrieve data
data = storage.get('model1/result1.json')
```

### Task Modules

Task modules define specific evaluation scenarios:

```python
# Example task module
class MyTask:
    features = ['params', 'result', 'target', 'score', 'model']
    show_features = ['params', 'result', 'target', 'score', 'model']
    sort_by = ['score']
    sort_by_asc = [False]
    
    def forward(self, model):
        # Task implementation
        params = {'message': 'test input'}
        result = model(**params)
        target = 'expected output'
        
        data = {
            'params': params,
            'result': result,
            'target': target,
        }
        data['score'] = self.score(data)
        return data

    def score(self, data):
        # Scoring implementation
        return 1.0 if data['result'] == data['target'] else 0.0
```

### Model Providers

Model providers serve as interfaces to various AI model APIs:

```python
# Using the OpenRouter provider
from deval.model.openrouter import OpenRouter

provider = OpenRouter(api_key='your-api-key')
models = provider.models()
response = provider.forward(
    message="Hello, how are you?",
    model="anthropic/claude-3.7-sonnet"
)
```

## Running Evaluations

### Single Epoch

```python
# Run a single evaluation epoch
results = evaluator.epoch()
```

### Background Evaluation

```python
# Initialize with background=True to run continuous evaluations
evaluator = deval(
    task='add',
    provider='model.openrouter',
    background=True,
    tempo=3600  # Run every hour
)
```

### Customizing Tasks

```python
# Set a different task
evaluator.set_task('custom_task')

# Run with the new task
results = evaluator.epoch()
```

## API Reference

For detailed API documentation, please refer to the [API Reference](docs/api_reference.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for providing valuable tools and libraries%