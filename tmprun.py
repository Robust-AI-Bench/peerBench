import val as v

# Initialize val with default settings
val = v.val(
    task='add',               # Task to evaluate
    provider='providers.openrouter',  # Model provider
    batch_size=16,            # Number of parallel evaluations
    n=10                      # Number of models to evaluate
)

# Run an evaluation epoch
results = val.epoch()


all_results = val.results()




 


# View the results
print(results)