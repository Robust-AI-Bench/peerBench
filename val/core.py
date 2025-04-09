import os
import pandas as pd
import time
import os
from .utils import *
from typing import *
import inspect
import tqdm
import hashlib
from functools import partial
import sys
import random
from concurrent.futures import ThreadPoolExecutor


print = log
class val:

    epoch_time = 0 # the time of the last epoch defaults to 0 utc timestamp
    def __init__(self,
                    task : str= 'mmlu', # score function
                    n_models  : int = 2, # the number of models to test
                    n_samples = 2, # the number of n_samples per epoch
                    provider = 'openrouter',
                    batch_size : int = 64, # the batch size of the most parallel tasks
                    key : str = None, # the key for the model
                    tempo : int = 3000, # the time between epochs
                    crypto_type='ecdsa',
                    store = 'store',
                    models = None, # the models to test
                    max_sample_age : int = 3600, # the maximum age of the n_samples
                    timeout : int = 4, # timeout per evaluation of the model
                    update : bool =True, # update during the first epoch
                    background : bool = False, # This is the key that we need to change to false
                    verbose: bool = True , # print verbose output
                    path : str= None, # the store path for the model eval, if not null then the model eval is stored in this directory
                 **kwargs):

        self.key = self.get_key(key or 'val', crypto_type=crypto_type)
        self.set_task(task)
        self.set_models(models, provider=provider, n_models=n_models)
        self.batch_size = min(batch_size, self.n_models)
        self.timeout = timeout
        self.n_samples = n_samples
        self.verbose = verbose
        self.tempo = tempo
        if background:
            thread(self.background) if background else ''

    def eval(self, 
                model:dict = None, 
                sample:Optional[dict]=None, 
                idx:Optional[str]=None,
                catch_error:bool = False,
                task:Optional[str]=None,
                 **kwargs):
        if task != None:
            self.set_task(task)
        if catch_error:
            try:
                return self.eval(model=model, sample=sample, idx=idx, catch_error=False, **kwargs)
            except Exception as e:
                print(f'Error({e})')
                return {'success': False, 'msg': str(e)}
        start_time = time.time() # the timestamp
        # resolve the model
        if model is None:
            model = self.models[0]
        sample = self.task.sample(sample=sample, idx=idx)
        # run the task over the model function
        model_fn = lambda **kwargs : self.provider.forward(model=model, sample=sample,  **kwargs)
        data = self.task.forward(model_fn)
        # requires the data to be a dictionary
        assert 'score' in data, f'Task({self.task}) did not return a score'
        # add the model and sample to the data
        data['model'] = model
        data['provider'] = self.provider.info['name']
        data['sample'] = sample
        data['task_cid'] = self.task.info['cid']
        data['sample_cid'] = self.hash(sample)
        data['validator'] = self.key.address
        data['time_start'] = start_time
        data['time_end'] = time.time()
        data['time_delta'] = data['time_end'] - data['time_start']
        # generate token over sorted keys (sorted to avoid non-collisions due to key order)
        data = {k: data[k] for k in sorted(data.keys())}
        # jwt token that involves the hash of the data
        data['token'] = self.key.get_token(self.hash(data), key=self.key)
        assert self.key.verify_token(data['token']), 'Failed to verify token'
        sample_path = f'{data["sample_cid"]}/sample.json'
        if not self.store.exists(sample_path):
            self.store.put(sample_path, sample)
        self.store.put(f"{data['sample_cid']}/{data['model']}.json", data)
        return data


    def sample_paths(self):
        """
        Get the sample paths
        """
        paths = self.store.paths()
        paths = [p for p in paths if '/sample.json' in p]
        return paths

    def providers(self):
        """
        Get the providers
        """
        return [p.split('model.')[-1] for p in modules(search='model') if 'model.' in p]

    def set_models(self, models=None, provider='openrouter', provider_prefix = 'model.', n_models=6):
        
        self.provider = self.module('model.'+provider)()
        info = {
            'name': provider,
            'cid': self.hash(provider),
            'time': time.time(),
            'description': self.provider.description if hasattr(self.provider, 'description') else '',
        }
        self.provider.info = info
        self.models = models or self.provider.models() # get the models from the provider
        shuffle(self.models)
        self.models = self.models[:n_models]
        self.n_models = len(self.models)
        assert len(self.models) > 0, f'No models found for provider {provider}'
        assert len(self.models) > 0, f'No models found for provider {provider}'
        assert hasattr(self.provider, 'forward'), f'Provider {self.provider} does not have a forward function'
        return {'success': True, 'msg': 'Provider set', 'provider': provider}


    def set_task(self, task: str, tasks_path='~/.val/tasks', store='store'):
        if task == None and hasattr(self, 'task'):
            return self.task.info
        if task == None and hasattr(self, 'task'):
            return self.task
        if isinstance(task, str):
            task_name = task
            task = self.module('task.'+task)()
            print(f'Loading task {task_name}', task)
        else:
            task_name = task.__class__.__name__.lower()
        assert hasattr(task, 'forward'), f'Task {task} does not have a forward function'
        assert callable(task.forward), f'No task function in task {task}'
        tasks_path= tasks_path + f'/{task_name}'
        self.store = self.module(store)(tasks_path) # prefix the store with the tasks path
        task_cid = self.hash(inspect.getsource(task.__class__))
        task_path = f'{task_name}/task.json'
        """
        Get the task info
        """
        # if the task is not found, create it
        if self.store.exists(task_path):
            info = self.store.get(task_path)
        else:
            info = {
                'name': task_name,
                'cid': task_cid,
                'time': time.time(),
                'description': task.description if hasattr(task, 'description') else '',
            }
            self.store.put(task_path, info)
        self.task = task
        self.task.info = info
        print(f'Task(name={self.task.info["name"]}, cid={self.task.info["cid"]})')
        return info

    def wait_for_epoch(self):
        while True:
            seconds_until_epoch = int(self.epoch_time + self.tempo - time.time())
            if seconds_until_epoch > 0:
                print(f'Waiting for epoch {seconds_until_epoch} seconds')
                time.sleep(seconds_until_epoch)
            else:
                break
        return {'success': True, 'msg': 'Epoch has arrived'}

    def background(self, step_time=2):
        while True:
            # wait until the next epoch
            self.wait_for_epoch()
            try:
                print(self.epoch())
            except Exception as e:
                print('XXXXXXXXXX EPOCH ERROR ----> XXXXXXXXXX {e}')
        raise Exception('Background process has stopped')

    def aggregate(self, results, **kwargs):

        """
        DEFAULT AGGREGATE FUNCTION
        This function aggregates the results of the task into a dataframe
        and returns the top n_models results
        """
        results =  df(self.store.items())

        results = results.sort_values(by=self.task.sort_by, ascending=self.task.sort_by_asc )
        # aggregate by model
        results = results.groupby('model').agg(lambda x: x.tolist()).reset_index()
        results =  results[['model', 'score']]
        
        results = results.sort_values(by='score', ascending=False)
        results['n_samples'] = results['score'].apply(lambda x: len(x))
        results['score'] = results['score'].apply(lambda x: sum(x)/len(x))

        results = results.sort_values(by='score', ascending=False)
        results = results.reset_index(drop=True)
        results['rank'] = results.index + 1
        return results

    # TODO: UPLOAD THE AGGREGATE FUNCTION TO SERVER
    def results(self, task = None, **kwargs):
        if task != None:
            self.set_task(task)
        aggfn = self.task.aggregate if hasattr(self.task, 'aggregate') else self.aggregate
        data = self.store.items()
        data = [d for d in data if 'score' in d]

        if len(data) == 0:
            return df([])
        else:
            data = df(data)
            data = aggfn(data, **kwargs)
        return data

    def _rm_all_store(self):
        return self.store._rm_all()

    def tasks(self):
        return [t.split('task.')[-1] for t in  modules(search='task')]

    def file2text(self, path) -> dict:
        assert os.path.isdir(path), f'File {path} is not a directory'
        files = os.listdir(path)
        file2text = {}
        for file in files:
            if file in ['__init__.py', '__pycache__', '.DS_Store', '.git']:
                continue
            file2text[file] =  self.get_text(os.path.join(path, file))
        return file2text

    def task2hash(self, task = 'mmlu'):
        """
        Get the task hash
        """
        task = self.module('task.'+task)()
        dirpath = os.path.dirname(inspect.getsourcefile(task.__class__))
        return self.hash(self.file2text(dirpath))

    def sample(self, idx:int=None):
        """
        Get the sample from the task
        """
        return self.task.sample(idx=idx)

    def epoch(self, task:Optional[str]=None, models:Optional[List[str]]=None, n_samples=None, **kwargs):
        buffer = f'\n{"-"*42}\n'

        n_samples = n_samples or self.n_samples
        if task != None:
            self.set_task(task)
        if models != None:
            self.set_models(models)
        threadpool = ThreadPoolExecutor(max_workers=128)
        n = len(self.models)
        batched_models = [self.models[i:i+self.batch_size] for i in range(0, n, self.batch_size)]
        num_batches = len(batched_models)
        results = []
        epoch_info = {
            'epoch_time': time.time(),
            'n_samples': self.n_samples,
            'batch_size': self.batch_size,
            'num_batches': num_batches,
            'task': self.task.info,
            'models': self.models,
        }
        print(buffer, f'Epoch({self.epoch_time})', buffer, epoch_info, buffer)
        for sample_idx in range(self.n_samples):
            sample = self.sample()
            sample_cid = self.hash(sample) # hash the sample
            print(buffer, f'Sample({sample_cid[:8]} idx={sample_idx}/{self.n_samples})', buffer, sample, buffer)
            for batch_idx, model_batch in enumerate(batched_models):
                future2model = {}
                sample_cid = self.hash(sample) # hash the sample
                batch_idx = batch_idx + 1
                print(buffer, f'Batch({batch_idx}/{num_batches})', buffer)
                for model in model_batch:
                    print(f'⚡️Eval(model={model})')
                    future = threadpool.submit(self.eval, model=model, sample=sample)
                    future2model[future] = model
                print(buffer, f'Results', buffer)
                try:
                    for f in as_completed(future2model, timeout=self.timeout):
                        model = future2model[f]
                        try:
                            r = f.result()
                            if isinstance(r, dict) and 'score' in r:
                                # Add emoji to the result
                                results.append(r)
                                print(f"✅Result(score={r['score']} model={r['model']})")
                            else:
                                # Handle the case where the result is not a dictionary
                                if self.verbose:
                                    print(f'❌EvalError(model={model} result={r})❌')
                        except Exception as e:
                            if self.verbose:
                                print(f'❌BatchError({e})❌')
                except TimeoutError as e:
                    print(f'Timeout error: {e}')

        self.epoch_time = time.time()
        results = self.process_results(results)
        return results






    def process_results(self,results, features = ['model', 'score', 'time_delta']):
        """
        Process the results of the epoch
        """
        if len(results) == 0:
            return {'success': False, 'msg': 'No results to vote on'}
        
        results = df(results)
        results = results.sort_values(by=self.task.sort_by, ascending=False)
        # aggregate by model
        results = results.groupby('model').agg(lambda x: x.tolist()).reset_index()
        results =  results[features]
        results = results.sort_values(by='score', ascending=False)
        results['n_samples'] = results['score'].apply(lambda x: len(x))
        results['score'] = results['score'].apply(lambda x: sum(x)/len(x))
        results['time_delta'] = results['time_delta'].apply(lambda x: sum(x)/len(x))
        results['sample_cids'] = results['sample_cid'].apply(lambda x: list(set(x)))
        # count the number o n_samples per model
        results = results.sort_values(by=['score', 'time_delta'], ascending=[False, True])
        results = results.reset_index(drop=True)
        print(f'Epoch complete! Processed {len(results)} results successfully')
        return results[features].reset_index(drop=True)
    
    @classmethod
    def module(cls, module_name):
        return module(module_name)

    def utils(self):
        from functools import partial
        filename = __file__.split('/')[-1]
        lines =  get_text(__file__.replace(filename, 'utils.py')).split('\n')
        fns = [l.split('def ')[1].split('(')[0] for l in lines if l.startswith('def ')]
        fns = list(set(fns))
        return fns

    def util(self, util_name):
        return self.module(f'val.utils.{util_name}')

    def get_key(self, key='fam', crypto_type='ecdsa'):
        return self.module('key')().get_key(key, crypto_type=crypto_type)

    def add_key(self, key='fam', crypto_type='ecdsa'):
        return self.get_key().add_key(key, crypto_type=crypto_type)

    def keys(self, crypto_type='ecdsa'):
        return self.get_key().keys(crypto_type=crypto_type)

    def sign(self, data, **kwargs):
        return self.key.sign(data, **kwargs)
    
    def verify(self, data, signature, address, **kwargs):
        return self.key.verify(data, signature, address, **kwargs)

    def cid(self, data):
        """
        Get the cid of the data
        """
        return self.hash(data)


    def task2cid(self, max_age=None, update=False, verbose=False):
        tasks = self.tasks()
        task_cids = {}
        for task in tasks:
            task_cid = self.task_cid(task, max_age=max_age, update=update, verbose=verbose)
            task_cids[task] = task_cid
        return task_cids

    def cli(self, default_fn = 'forward') -> None:
        """
        Run the command line interface
        """
        t0 = time.time()
        argv = sys.argv[1:]
        fn = argv.pop(0)
        if '/' in fn:
            module_path = '/'.join(fn.split('/')[:-1]).replace('/', '.')
            module_obj = module(module_path)()
            if fn.endswith('/'):
                fn = default_fn
            fn = fn.split('/')[-1]

        else:
            module_obj = self
        fn_obj = getattr(module_obj, fn)
        args = []
        kwargs = {}
        parsing_kwargs = False
        for arg in argv:
            if '=' in arg:
                parsing_kwargs = True
                key, value = arg.split('=')
                kwargs[key] = str2python(value)
            else:
                assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                args.append(str2python(arg))
        module_name = module_obj.__class__.__name__.lower()
        if len(args)> 0:
            kwargs_from_args = {k: v for k, v in zip(inspect.getfullargspec(fn_obj).args[1:], args)}
            params  = {**kwargs, **kwargs_from_args}
        else:
            params = kwargs
        # remove the self and kwargs from the params
        print(f'Running(fn={module_name}/{fn} params={params})')
        output = fn_obj(**params) if callable(fn_obj) else fn_obj
        speed = time.time() - t0
        print(output)

    @classmethod
    def init(cls, globals_dict=None, **kwargs):
        if globals_dict != None:        
            #add the functions and classes of the module to the global namespace
            globals_dict = globals_dict or {}
            for k,v in val.__dict__.items():
                globals_dict[k] = v     
            for f in dir(val):
                def wrapper_fn(f, *args, **kwargs):
                    fn = getattr(val(), f)
                    return fn(*args, **kwargs)
                globals_dict[f] = partial(wrapper_fn, f)

        for util in cls().utils():
            def wrapper_fn(util, *args, **kwargs):
                import importlib
                fn = obj(f'val.utils.{util}')
                return fn(*args, **kwargs)
            setattr(val, util, partial(wrapper_fn, util))

    def hash(self, data='hey', mode  = 'sha256', **kwargs):
        """
        Hash the data
        """
        return get_hash(data, mode=mode, **kwargs)



    @classmethod
    def run(cls, *args, **kwargs):
        return cls(*args, **kwargs).epoch()


def main():
    return val().cli()


