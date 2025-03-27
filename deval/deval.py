
import os
import pandas as pd
import time
import os
from .utils import *
from typing import *
import inspect
import tqdm
from functools import partial
import sys

class deval:

    def __init__(self,
                    search : Optional[str] =  None, # (OPTIONAL) the search string for the network 
                    batch_size : int = 64, # the batch size of the most parallel tasks
                    task : Union['callable', int]= 'add', # score function
                    key : str = None, # the key for the model
                    tempo : int = 3000, # the time between epochs
                    provider = 'model.openrouter',
                    crypto_type='ecdsa',
                    auth = 'auth',
                    storage = 'storage',
                    models = None, # the models to test
                    n : int = 64, # the number of models to test
                    max_sample_age : int = 3600, # the maximum age of the samples
                    timeout : int = 4, # timeout per evaluation of the model
                    update : bool =True, # update during the first epoch
                    background : bool = False, # This is the key that we need to change to false
                    verbose: bool = False, # print verbose output
                    path : str= None, # the storage path for the model eval, if not null then the model eval is stored in this directory
                 **kwargs):  
        self.epochs = 0 # the number of epochs
        self.epoch_time = 0
        self.timeout = timeout
        self.batch_size = batch_size
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

    def set_provider(self, provider):
        self.provider = self.module(provider)()
        provider_prefix = 'deval.model.'
        if provider.startswith(provider_prefix):
            provider_name = provider[len(provider_prefix):]
        else:
            provider_name = provider
        self.provider_name = provider_name
        self.models = self.provider.models()
        return {'success': True, 'msg': 'Provider set', 'provider': provider}

    def set_task(self, task: str, task_results_path='~/.deval/results', storage='deval.storage'):
        self.task = self.module('task.'+task)()
        self.task.task_name = task.lower()
        assert callable(self.task.forward), f'No task function in task {task}'
        self.task.task_id = sha256(inspect.getsource(self.task.forward))
        self.storage = self.module(storage)(f'{task_results_path}/{self.task.task_name}')
        return {'success': True, 'msg': 'Task set', 'task': task, 'task_id': self.task.task_id, }

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


    def score_model(self,  model:dict, task=None, **kwargs):
        t0 = time.time() # the timestamp
        print(f'Scoring({model}, task={self.task.task_name})')
        model_fn = lambda **kwargs : self.provider.forward(model=model,  **kwargs)
        data = self.task.forward(model_fn)
        extra_data = {
            'model': model,
            'time': t0,
            'duration': time.time() - t0,
            'vali': self.key.key_address,
            'task_id': self.task.task_id,
            'provider': self.provider_name
        }
        data.update(extra_data)
        data['token'] = self.auth.get_token(data)
        assert self.auth.verify_token(data['token']), 'Failed to verify token'
        self.storage.put(f"{data['model']}/{data['time']}.json", data)
        return data

    def results(self, **kwargs):
        results =  df(self.storage.items())[self.task.show_features]
        results = results.sort_values(by=self.task.sort_by, ascending=self.task.sort_by_asc )
        return results

    def _rm_all(self):
        return self.storage._rm_all()

    def tasks(self):
        return [t.split('task.')[-1] for t in  modules(search='task')]

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
        for i, model_batch in enumerate(batched_models):
            print(f'Batch {i+1}/{num_batches} ({len(model_batch)})')
            futures = []
            for m in model_batch:
                future = threadpool.submit(self.score_model, m)
                futures.append(future)
                
            try:
                for f in tqdm.tqdm(as_completed(futures), total=len(futures), desc='Scoring models'):
                    try:
                        r = f.result()
                        if isinstance(r, dict) and 'score' in r:
                            results.append(r)
                            print({'score': r['score'], 'model': r['model']})
                        else:
                            print('Invalid result', r) if self.verbose else ''
                    except Exception as e:
                        print('Error', e)
            except TimeoutError as e:
                print('Timeout Error', e)

        self.epochs += 1
        self.epoch_time = time.time()
        if len(results) == 0:
            return {'success': False, 'msg': 'No results to vote on'}
        results =  df(results)[self.task.show_features]
        results = results.sort_values(by=self.task.sort_by, ascending=False)
        return results

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
        return self.module(f'deval.utils.{util_name}')

    def get_key(self, key='fam', crypto_type='ecdsa'):
        return self.module('deval.key')().get_key(key, crypto_type=crypto_type)

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
        for k,v in deval.__dict__.items():
            globals_input[k] = v     
        for f in dir(deval):
            def wrapper_fn(f, *args, **kwargs):
                fn = getattr(deval(), f)
                return fn(*args, **kwargs)
            globals_input[f] = partial(wrapper_fn, f)

    def forward(self, model='microsoft/wizardlm-2-7b'):
        return self.score_model(model)

    @classmethod
    def init(cls, **kwargs):
        for util in cls().utils():
            def wrapper_fn(util, *args, **kwargs):
                import importlib
                fn = obj(f'deval.utils.{util}')
                return fn(*args, **kwargs)
            setattr(deval, util, partial(wrapper_fn, util))

    def cli(self) -> None:
        """
        Run the command line interface
        """
        t0 = time.time()
        argv = sys.argv[1:]
        fn = argv.pop(0)
        if '/' in fn:
            module_obj = module('/'.join(fn.split('/')[:-1]).replace('/', '.'))()
            if fn.endswith('/'):
                fn = 'forward'
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
        print(f'Running(fn={module}/{fn} args={args} kwargs={kwargs})')
    
        output = fn_obj(*args, **kwargs) if callable(fn_obj) else fn_obj
        duration = time.time() - t0
        print(output)


    def test(self, modules = ['key', 'auth']):
        """
        Test the deval module
        """
        for m in modules:
            print(f'Testing {m}')
            obj = self.module(m)()
            obj.test()
        return {'success': True, 'msg': 'All tests passed'}

def main():
    return deval().cli()
deval.init()
