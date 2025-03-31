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


all_results = deval.results()




 


# View the results
print(results)