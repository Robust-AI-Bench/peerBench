import random
import json
class AddTask:
    features = ['params', 'result', 'target', 'score', 'model', 'provider', 'token']
    show_features = ['params', 'result', 'target', 'score', 'model', 'duration']
    sort_by = ['score', 'duration']
    sort_by_asc = [False, True]
    description = 'tests a model to add two numberts'
    output_bounds = ['<OUTPUT_JSON>', '</OUTPUT_JSON>']
    temperature = 0
    max_tokens = 10000
    
    def forward(self, model):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        x = 'add two numbers {} and {}'.format(a, b)
        message =  {'input': str({'a': a, 'b': b}), 
                 'goals': [
                    f'return a json object with the sum of {a} and {b}',
                 ],
                 'output_format': f' strictly as {self.output_bounds[0]}json(y:int){self.output_bounds[1]}'
                 }
        params = {'message': message,  'temperature': self.temperature, 'max_tokens': self.max_tokens}
             
        target = a + b
        y =  model(**params)
        data = {
            'params': params,
            'result': y,
            'target': target,
        }
        data['score'] = self.score(data)
        return data

    def score(self, data):
        return int(str(data['target']) in  data['result'])

    
    def verify_sample(self, data):
        assert all([f in data for f in self.features]), f'Missing features {self.features}'