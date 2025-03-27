
import os
import pandas as pd
import time
import os
from .utils import *
from typing import *
import inspect
import commune as c
import tqdm

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
                    verbose: bool = True, # print verbose output
                    path : str= None, # the storage path for the model eval, if not null then the model eval is stored in this directory
                    storage_path = '~/.deval',
                 **kwargs):  
        self.storage_path = abspath(storage_path)
        self.epochs = 0 # the number of epochs
        self.epoch_time = 0
        self.timeout = timeout
        self.batch_size = batch_size
        self.verbose = verbose
        self.key = c.get_key(key, crypto_type=crypto_type)
        self.tempo = tempo
        self.set_provider(provider)
        self.auth = self.module(auth)()
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
        self.task_name = task.lower()
        assert callable(self.task.forward), f'No task function in task {task}'
        self.task_id = sha256(inspect.getsource(self.task.forward))
        self.storage = self.module(storage)(f'{task_results_path}/{self.task_name}')
        print(f'Task set to {task}')
        return {'success': True, 'msg': 'Task set', 'task': task, 'task_id': self.task_id, }

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

    def score_model(self,  model:dict, **kwargs):
        t0 = time.time() # the timestamp
        model_fn = lambda **kwargs : self.provider.forward(model=model,  **kwargs)
        data = self.task.forward(model_fn)
        extra_data = {
            'model': model,
            'time': t0,
            'duration': time.time() - t0,
            'vali': self.key.key_address,
            'task_id': self.task_id,
            'provider': self.provider_name
        }
        data.update(extra_data)
        assert self.auth.verify_token(data['token']), 'Failed to verify token'
        self.storage.put(f"{data['model']}/{data['time']}.json", data)
        return data

    def results(self, **kwargs):
        df =  self.df(self.storage.items())[self.task.show_features]
        df = df.sort_values(by=self.task.sort_by, ascending=self.task.sort_by_asc )
        return df

    def df(self, x):
        import pandas as pd
        return pd.DataFrame(x)

    def _rm_all(self):
        return self.storage._rm_all()

    def epoch(self, task=None,  **kwargs):
        if task != None:
            self.set_task(task)
        n = len(self.models)
        batched_models = [self.models[i:i+self.batch_size] for i in range(0, n, self.batch_size)]
        num_batches = len(batched_models)
        results = []
        results_count = 0
        for i, model_batch in enumerate(batched_models):
            futures = [c.submit(self.score_model, [m], timeout=self.timeout) for m in model_batch]
            try:
                for f in c.as_completed(futures, timeout=self.timeout):
                    r = f.result()
                    if isinstance(r, dict) and 'score' in r:
                        results.append(r)
                        print({'idx': len(results) + 1, 'model': r['model'], 'score': r['score']})
            except TimeoutError as e:
                print('Timeout Error', e)

        self.epochs += 1
        self.epoch_time = time.time()
        if len(results) == 0:
            return {'success': False, 'msg': 'No results to vote on'}
        df =  self.df(results)[self.task.show_features]
        df = df.sort_values(by=self.task.sort_by, ascending=False)
        return df

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
