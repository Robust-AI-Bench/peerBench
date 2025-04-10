import random
from typing import Optional
import json
import os
import inspect

class Task:
    features = ['params', 'result', 'target', 'score', 'model', 'provider', 'token']
    sort_by = ['score']
    sort_by_asc = [False]
    description = 'tests a model to add two numberts'
    output_bounds = ['<OUTPUT_JSON>', '</OUTPUT_JSON>']
    temperature = 0
    max_tokens = 10000

    def sample(self , idx:int = None, sample=None) -> dict:
        """
        generate the sample
        """
        # if params is not None, use it
        if sample is not None:
            return sample
        # generate two random numbers from the id seed
        if idx != None:
            random.seed(idx)
        idx = idx or random.randint(1, 1000)
        random.seed(idx)
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        # return a sample with the two numbers
        return {'message': {
                    'a': a,
                    'b': b,
                    'goal': 'return a json object with the sum  ',
                    'output_format': f'strictly as {self.output_bounds[0]}json(y:int){self.output_bounds[1]}'
                },  
                'temperature': self.temperature, 
                'max_tokens': self.max_tokens
                }

    def forward(self, model: callable, sample:Optional[dict]=None, idx=None) -> dict:
        """
        run the model on the sample
        Args:
            model: the model to run
            sample: the sample to run on
            idx: the index of the sample
        Returns:
            dict: the result of the model
        """ 
        
        sample = self.sample(idx=idx, sample=sample)

        result = model(**sample)

        data = {
            'sample': sample,
            'result': result,
        }

        # step 3 : score the data
        return self.score(data)
    
    def score(self, data:dict) -> float:
        sample_data = data['sample']['message']
        target = str(sample_data['a']+ sample_data['b'])
        data['score'] =  float(str(target) in  data['result'])
        return data
 
    def verify_sample(self, data):
        assert all([f in data for f in self.features]), f'Missing features {self.features}'
