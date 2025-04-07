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

print = log
class val:


    def __init__(self,
                    search : Optional[str] =  None, # (OPTIONAL) the search string for the network 
                    batch_size : int = 64, # the batch size of the most parallel tasks
                    task : str= 'task', # score function
                    key : str = None, # the key for the model
                    tempo : int = 3000, # the time between epochs
                    provider = 'openrouter',
                    crypto_type='ecdsa',
                    auth = 'auth',
                    storage = 'storage',
                    samples_per_epoch = 4, # the number of samples per epoch
                    models = None, # the models to test
                    n : int = 8, # the number of models to test
                    max_sample_age : int = 3600, # the maximum age of the samples
                    timeout : int = 4, # timeout per evaluation of the model
                    update : bool =True, # update during the first epoch
                    background : bool = False, # This is the key that we need to change to false
                    verbose: bool = False, # print verbose output
                    path : str= None, # the storage path for the model eval, if not null then the model eval is stored in this directory
                 **kwargs):  
        self.epoch_time = 0
        self.timeout = timeout
        self.batch_size = min(batch_size, n)
        self.samples_per_epoch = samples_per_epoch
        self.verbose = verbose
        self.key = self.get_key(key, crypto_type=crypto_type)
        self.tempo = tempo
        self.auth = self.module(auth)()
        self.set_provider(provider)
        self.set_task(task)
        shuffle(self.models)
        self.models = self.models[:n]
        if background:
            thread(self.background) if background else ''


    def eval(self,  model:dict = None, sample=None, idx=None, **kwargs):
        start_time = time.time() # the timestamp
        # resolve the model
        if model is None:
            model = self.models[0]
        sample = self.task.sample(sample=sample, idx=idx)
        # run the task over the model function
        model_fn = lambda **kwargs : self.provider.forward(model=model, sample=sample,  **kwargs)
        data = self.task.forward(model_fn)
        # add the model and sample to the data
        data['model'] = model
        data['sample'] = sample
        data['task_cid'] = self.task.cid
        data['sample_cid'] = self.hash(sample)
        data['score'] = data['score'] if 'score' in data else 0
        data['provider'] = self.provider_name
        data['validator'] = self.key.address
        data['time_start'] = start_time
        data['time_end'] = time.time()

        # generate token over sorted keys (sorted to avoid non-collisions due to key order)
        data = {k: data[k] for k in sorted(data.keys())}

        # verify the token
        data['token'] = self.auth.get_token(self.hash(data), key=self.key)

        # verify the token
        assert self.auth.verify_token(data['token']), 'Failed to verify token'
        self.storage.put(f"{data['model']}/{data['sample_cid']}.json", data)
        return data

    def set_provider(self, provider):
        self.provider = self.module(provider)()
        provider_prefix = 'providers.'
        if provider.startswith(provider_prefix):
            provider_name = provider[len(provider_prefix):]
        else:
            provider_name = provider
        self.provider_name = provider_name
        self.models = self.provider.models()
        return {'success': True, 'msg': 'Provider set', 'provider': provider}

    def set_task(self, task: str, task_results_path='~/.val/results', storage='val.storage'):

        self.task = self.module('task.'+task)()
        self.task.name = task.lower()
        assert callable(self.task.forward), f'No task function in task {task}'
        self.storage = self.module(storage)(f'{task_results_path}/{self.task.name}')
        self.task.cid = self.hash(inspect.getsource(self.task.__class__))
        return {'success': True, 'msg': 'Task set', 'task': task, 'cid': self.task.cid }

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
        and returns the top n results
        """
        results =  df(self.storage.items())

        results = results.sort_values(by=self.task.sort_by, ascending=self.task.sort_by_asc )
        # aggregate by model
        results = results.groupby('model').agg(lambda x: x.tolist()).reset_index()
        results =  results[['model', 'score']]
        
        results = results.sort_values(by='score', ascending=False)
        results['n'] = results['score'].apply(lambda x: len(x))
        results['score'] = results['score'].apply(lambda x: sum(x)/len(x))

        results = results.sort_values(by='score', ascending=False)
        results = results.reset_index(drop=True)
        results['rank'] = results.index + 1
        return results

    # TODO: UPLOAD THE AGGREGATE FUNCTION TO SERVER
    def results(self, data=None, **kwargs):
        aggfn = self.task.aggregate if hasattr(self.task, 'aggregate') else self.aggregate
        data = data or self.storage.items()
        if len(items) == 0:
            return df([])
        return df(data)

    def _rm_all_storage(self):
        return self.storage._rm_all()

    def tasks(self):
        return [t.split('task.')[-1] for t in  modules(search='task')]

    def sample(self, idx:int=None):
        """
        Get the sample from the task
        """
        return self.task.sample(idx=idx)
    def epoch(self, task=None,  **kwargs):
        if task != None:
            self.set_task(task)
        from concurrent.futures import ThreadPoolExecutor
        threadpool = ThreadPoolExecutor(max_workers=128)
        n = len(self.models)
        batched_models = [self.models[i:i+self.batch_size] for i in range(0, n, self.batch_size)]
        num_batches = len(batched_models)
        results = []
        results_count = 0

        for sample_idx in range(self.samples_per_epoch):
            sample = self.sample()
            for batch_idx, model_batch in enumerate(batched_models):
                futures = []
                sample_cid = self.hash(sample)
                # random color for the sample
                sample_color = random_color()
                # sending the sample to the server
                print(f'Batch({batch_idx}/{num_batches}, batch_size={self.batch_size})', color=sample_color)
                print(f'Sample({sample_idx}/{self.samples_per_epoch} cid={sample_cid})', color=sample_color)
                abrev_sample_cid = sample_cid[:8] + '..'
                for model in model_batch:
                    print(f'Sample({abrev_sample_cid}) --> Model({model}))', color=sample_color)
                    future = threadpool.submit(self.eval, model=model, sample=sample)
                    futures.append(future)
                try:
                    for f in as_completed(futures):
                        try:
                            r = f.result()
                            if isinstance(r, dict) and 'score' in r:
                                results.append(r)
                                print( f"Result(model={r['model']} score={r['score']} sample={abrev_sample_cid})", color=sample_color)
                        except Exception as e:
                            if self.verbose:
                                print(f'BatchError({e})')
                except TimeoutError as e:
                    print(f'TimeoutError({e})')

        self.epoch_time = time.time()
        if len(results) == 0:
            return {'success': False, 'msg': 'No results to vote on'}
        results =  df(results)
        results = results.sort_values(by=self.task.sort_by, ascending=False)
        return results[['model', 'provider', 'score', 'sample_cid']].reset_index(drop=True)

    def module(self, module_name):
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
        return self.module('val.key')().get_key(key, crypto_type=crypto_type)

    def add_key(self, key='fam', crypto_type='ecdsa'):
        return self.get_key().add_key(key, crypto_type=crypto_type)

    def keys(self, crypto_type='ecdsa'):
        return self.get_key().keys(crypto_type=crypto_type)

    def sign(self, data, **kwargs):
        return self.key.sign(data, **kwargs)
    
    def verify(self, data, signature, address, **kwargs):
        return self.key.verify(data, signature, address, **kwargs)

    @classmethod
    def add_globals(cls, globals_input:dict = None):
        """
        add the functions and classes of the module to the global namespace
        """
        globals_input = globals_input or {}
        for k,v in val.__dict__.items():
            globals_input[k] = v     
        for f in dir(val):
            def wrapper_fn(f, *args, **kwargs):
                fn = getattr(val(), f)
                return fn(*args, **kwargs)
            globals_input[f] = partial(wrapper_fn, f)


    def cid(self, data):
        """
        Get the cid of the data
        """
        if isinstance(data, str):
            return self.hash(data)
        elif isinstance(data, dict):
            return self.hash(json.dumps(data))
        elif isinstance(data, list):
            return self.hash(json.dumps(data))
        else:
            raise Exception('Data type not supported')

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
        duration = time.time() - t0
        print(output)

    def test(self, modules = ['key', 'auth']):
        """
        Test the val module
        """
        for m in modules:
            print(f'Testing {m}')
            obj = self.module(m)()
            obj.test()
        return {'success': True, 'msg': 'All tests passed'}


    @classmethod
    def init(cls, globals_dict=None, **kwargs):
        if globals_dict != None:
            cls.add_globals(globals_dict)
        
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

def main():
    return val().cli()


