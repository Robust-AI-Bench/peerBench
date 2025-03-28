# Deval - Decentralized Evaluation Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deval is a powerful, flexible framework for evaluating and benchmarking language models in a decentralized manner. It provides tools for creating standardized tasks, running evaluations across multiple models, and securely storing and verifying results.

## 🚀 Features

- **Decentralized Evaluation**: Run evaluations across multiple providers and models
- **Secure Authentication**: Cryptographic signing of evaluation results with multiple key types (ECDSA, SR25519, ED25519)
- **Flexible Task System**: Create custom evaluation tasks with standardized interfaces
- **Model Provider Abstraction**: Support for multiple model providers through a unified interface
- **Persistent Storage**: Save and retrieve evaluation results
- **Parallel Execution**: Efficiently evaluate multiple models in parallel

## 📋 Installation

```bash
pip install deval
```

## 🔧 Quick Start

```python
import deval as d

# Initialize deval with default settings
deval = d.deval(
    task='add',               # Task to evaluate
    provider='providers.openrouter',  # Model provider
    batch_size=16,            # Number of parallel evaluations
    n=10                      # Number of models to evaluate
)

# Run an evaluation epoch
results = deval.epoch()

# View the results
print(results)
```

## 🔐 Authentication

Deval uses cryptographic signatures to ensure the integrity and authenticity of evaluation results:

```python
# Create a new key
key = d.get_key('my_key', crypto_type='ecdsa')

# Sign some data
signature = key.sign("Hello, world!")

# Verify the signature
is_valid = d.verify(data="Hello, world!", signature=signature, address=key.key_address)
```

The framework supports multiple cryptographic schemes:
- ECDSA (Ethereum-compatible)
- SR25519 (Substrate/Polkadot)
- ED25519 (widely used in cryptography)

## 🧪 Creating Custom Tasks

Tasks define how models are evaluated. Create a custom task by defining a class with a `forward` method:

```python
class MyTask:
    features = ['params', 'result', 'target', 'score', 'model', 'provider', 'token']
    show_features = ['params', 'result', 'target', 'score', 'model', 'duration']
    sort_by = ['score', 'duration']
    sort_by_asc = [False, True]
    
    def forward(self, model):
        # Your evaluation logic here
        params = {'message': 'What is 2+2?'}
        result = model(**params)
        target = '4'
        
        data = {
            'params': params,
            'result': result,
            'target': target,
        }
        data['score'] = self.score(data)
        return data
    
    def score(self, data):
        return int(data['target'] in data['result'])
```

## 🔌 Model Providers

Deval abstracts away the differences between model providers. Currently supported:

- OpenRouter
- LiteLLM (supporting multiple backend providers)

Adding a new provider is as simple as implementing the provider interface:

```python
class MyProvider:
    def __init__(self, api_key=None, **kwargs):
        # Initialize your provider
        pass
        
    def forward(self, model, message, **kwargs):
        # Implement model inference
        pass
        
    def models(self):
        # Return available models
        return ['model1', 'model2']
```

## 📊 Storing and Analyzing Results

Results are automatically stored and can be retrieved for analysis:

```python
# Get all results for the current task
all_results = deval.results()

# Display as a pandas DataFrame
print(all_results)
```

## 🔄 Continuous Evaluation

Run evaluations in the background:

```python
deval = d.deval(
    task='add',
    background=True,  # Run evaluations in the background
    tempo=3600        # Run every hour
)
```

## 🛠️ Command Line Interface

Deval includes a command-line interface for running evaluations:

```bash
python -m deval epoch --task=add --n=10
```

## 🧩 Architecture

Deval consists of several key components:

1. **Core Engine** (`deval.py`): Manages the evaluation process
2. **Authentication** (`auth.py`): Handles cryptographic signing and verification
3. **Key Management** (`key.py`): Manages cryptographic keys
4. **Storage** (`storage.py`): Persists evaluation results
5. **Tasks** (`task/`): Defines evaluation tasks
6. **Model Providers** (`model/`): Abstracts model APIs

## 📚 API Reference

### `deval`

The main class for running evaluations.

```python
d.deval(
    task='add',               # Task to evaluate
    provider='providers.openrouter',  # Model provider
    batch_size=64,            # Number of parallel evaluations
    key=None,                 # Key for signing results
    tempo=3000,               # Time between epochs in seconds
    n=64,                     # Number of models to evaluate
    background=False,         # Run in background
    verbose=False             # Print verbose output
)
```

### `Key`

Manages cryptographic keys for signing and verification.

```python
key = d.get_key('my_key', crypto_type='ecdsa')
key.sign(data)
key.verify(data, signature, address)
```

### `Auth`

Handles JWT token generation and verification.

```python
auth = d.module('auth')()
token = auth.get_token(data)
verified_data = auth.verify_token(token)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.%  