import val as v
models = [ 'meta-llama/llama-4-maverick']

validator = v.val(models=models, task='math500', n_samples=1, timeout=12)


for i in range(2): 
    print(validator.epoch())