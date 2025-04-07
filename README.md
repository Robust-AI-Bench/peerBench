import deval as d

# Initialize the evaluator
val = d.Deval(
    task='add',          # Task to evaluate (e.g., 'add', 'divide')
    provider='openrouter', # Model provider
    n=4,                 # Number of models to test
    samples_per_epoch=2  # Samples per evaluation epoch
)

# Run an evaluation epoch
results = val.epoch()
print(results)
```

## Core Components

### Tasks

Tasks define what you want to evaluate. Deval comes with several built-in tasks:

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
models = evaluator.models()
print(models)
```

### Authentication

Secure your evaluations with cryptographic authentication:

```python
# Generate a new key
key = Deval().get_key('my_key', crypto_type='ecdsa')

# Create an authentication token
auth = Deval().module('auth')()
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
evaluator = Deval(
    task='add',
    background=True,  # Run in background
    tempo=60          # Run every 60 seconds
)
```

### Aggregating Results

View and analyze evaluation results:

```python
# Get aggregated results
print( val.results())
```

## Command Line Interface

Deval includes a CLI for common operations:

```bash
# Run an evaluation epoch
d epoch --task=add --n=4

# List available tasks
d tasks

# Test components
d test
```

## Security

Deval includes cryptographic functions for securing evaluations:

- ECDSA, ED25519, and SR25519 key types
- JWT-like token authentication
- Secure storage of keys and results

## License

MIT%  