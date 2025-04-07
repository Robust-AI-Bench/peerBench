# val: Decentralized Evaluation Framework

val is a powerful framework for evaluating and benchmarking AI models across different providers in a decentralized manner. It provides a simple, flexible interface for testing model performance on various tasks with cryptographic verification.

## Quick Start

```python
import val as v

# Initialize the evaluator
val = v.Val(
    task='add',          # Task to evaluate (e.g., 'add', 'divide')
    provider='openrouter', # Model provider
    n=4,                 # Number of models to test
    samples_per_epoch=2  # Samples per evaluation epoch
)

# Run an evaluation epoch
results = val.epoch()
print(results)
```

## Installation

### Using pip

```bash
pip install val
```

### Using Docker

We provide a Docker environment for easy setup and isolation:

```bash
# Clone the repository
git clone https://github.com/val-ai/val.git
cd val

# Build the Docker image
make build

# Start the container
make start

# Enter the container
make enter

# Run tests
make test
```

## Core Components

### Tasks

Tasks define what you want to evaluate. val comes with several built-in tasks:

```python
# List available tasks
tasks = val.tasks()
print(tasks)  # ['add', 'divide', ...]

# Set a specific task
val.set_task('add')
```

### Providers

Providers connect to different AI model APIs:

```python
# Set a provider
val.set_provider('openrouter')

# List available models from the provider
models = val.models()
print(models)
```

### Authentication

Secure your evaluations with cryptographic authentication:

```python
# Generate a new key
key = v.get_key('my_key', crypto_type='ecdsa')

# Create an authentication token
auth = v.module('auth')()
token = auth.get_token({'data': 'test'}, key=key)

# Verify a token
verified_data = auth.verify_token(token)
```

## Advanced Usage

### Custom Tasks

Create custom evaluation tasks by extending the base Task class:

```python
# Define a custom task in task/custom/task.py
class CustomTask:
    features = ['params', 'result', 'target', 'score', 'model', 'provider', 'token']
    sort_by = ['score']
    sort_by_asc = [False]
    description = 'My custom evaluation task'
    
    def sample(self, idx=None, sample=None):
        # Generate or return a sample
        return {'message': {'prompt': 'Your test prompt'}}
    
    def forward(self, model, sample=None, idx=None):
        # Run the model on the sample
        sample = self.sample(idx=idx, sample=sample)
        result = model(**sample)
        return self.score({'sample': sample, 'result': result})
    
    def score(self, data):
        # Score the model's response
        data['score'] = 1.0  # Your scoring logic here
        return data
```

### Background Evaluation

Run evaluations in the background:

```python
evaluator = v.Val(
    task='add',
    background=True,  # Run in background
    tempo=60          # Run every 60 seconds
)
```

### Aggregating Results

View and analyze evaluation results:

```python
# Get aggregated results
print(val.results())
```

## Command Line Interface

val includes a CLI for common operations:

```bash
# Run an evaluation epoch
d epoch --task=add --n=4

# List available tasks
d tasks

# Test components
d test
```

## Docker Environment

The included Docker environment provides a complete setup for val:

```dockerfile
# FROM ubuntu:22.04 base with Python, Docker, and other dependencies
# See Dockerfile for details

# Build the image
docker build -t val .

# Run the container
docker run -d \
  --name val \
  --network=host \
  --restart unless-stopped \
  --privileged --shm-size 4g \
  -v $(pwd):/app \
  -v /var/run/docker.sock:/var/run/docker.sock \
  val