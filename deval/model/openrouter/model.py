from typing import Generator
import requests
import json
import openai
import deval as d
import random
import os
import time

class OpenRouter:
    storage_path = os.path.expanduser('~/.deval/model/openrouter') # path to store models (relative to storage_path) 
    api_key_path = f'{storage_path}/api.json' # path to store api keys (relative to storage_path)
    def __init__(
        self,
        api_key = None,
        base_url: str = 'https://openrouter.ai/api/v1',
        timeout: float = None,
        prompt:str=None,
        max_retries: int = 10,
        **kwargs
    ):
        """
        Initialize the OpenAI with the specified model, API key, timeout, and max retries.

        Args:
            model (OPENAI_MODES): The OpenAI model to use.
            api_key (API_KEY): The API key for authentication.
            base_url (str, optional): can be used for openrouter api calls
            timeout (float, optional): The timeout value for the client. Defaults to None.
            max_retries (int, optional): The maximum number of retries for the client. Defaults to None.
        """
        self.base_url = base_url
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=api_key or self.get_key(),
            timeout=timeout,
            max_retries=max_retries,
        )
        self.prompt = prompt

    def forward(
        self,
        message: str,
        *extra_text , 
        history = None,
        prompt: str =  None,
        system_prompt: str = None,
        stream: bool = False,
        model:str = 'anthropic/claude-3.7-sonnet',
        max_tokens: int = 10000000,
        temperature: float = 0,
        **kwargs
    ) -> str :
        """
        Generates a response using the OpenAI language model.

        Args:
            message (str): The message to send to the language model.
            history (ChatHistory): The conversation history.
            stream (bool): Whether to stream the response or not.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The sampling temperature to use.

        Returns:
        Generator[str] | str: A generator for streaming responses or the full streamed response.
        """
        prompt = prompt or system_prompt
        message = str(message)
        if len(extra_text) > 0:
            message = message + ' '.join(extra_text)
        history = history or []
        prompt = prompt or self.prompt
        message = message + prompt if prompt else message
        model = self.resolve_model(model)
        model_info = self.get_model_info(model)
        num_tokens = len(message)
        max_tokens = min(max_tokens, model_info['context_length'] - num_tokens)
        messages = history.copy()
        messages.append({"role": "user", "content": message})
        result = self.client.chat.completions.create(model=model, 
                                                    messages=messages, 
                                                    stream= bool(stream),
                                                    max_tokens = max_tokens, 
                                                    temperature= temperature  )
        if stream:
            def stream_generator( result):
                for token in result:
                    yield token.choices[0].delta.content
            return stream_generator(result)
        else:
            return result.choices[0].message.content
        
    generate = forward

    def resolve_model(self, model=None):
        models =  self.models()
        model = str(model)
        if str(model) not in models:
            if ',' in model:
                models = [m for m in models if any([s in m for s in model.split(',')])]
            else:
                models = [m for m in models if str(model) in m]
            print(f"Model {model} not found. Using {models} instead.")
            assert len(models) > 0
            model = models[0]

        return model

    def get_key(self):
        """
        get the api keys
        """
        keys = self.get(self.api_key_path, [])
        if len(keys) > 0:
            return random.choice(keys)
        else:
            return 'password'


    def get(self, path, default=None,  update=False):
        """
        Get the json file from the path
        """
        if update :
            return default
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, str):
                    data = json.loads(data)
                return data
        except Exception as e:
            return default

        
    def put(self, path, data):
        """
        Put the json file to the path
        """
        dirpqth = os.path.dirname(path)
        if not os.path.exists(dirpqth):
            os.makedirs(dirpqth)
        with open(path, 'w') as f:
            json.dump(data, f)
        return {'status': 'success', 'path': path}

    def keys(self):
        """
        Get the list of API keys
        """
        try:
            return self.get(self.api_key_path, [])
        except Exception as e:
            return []

    def add_key(self, key):
        keys = self.keys()
        keys.append(key)
        keys = list(set(keys))
        self.put(self.api_key_path, keys)
        return keys

    def resolve_path(self, path):
        return 

    def model2info(self, search: str = None, update=False):
        path =  f'{self.storage_path}/models.json'
        models = self.get(path, default={}, update=update)
        if len(models) == 0:
            response = requests.get(self.base_url + '/models')
            models = json.loads(response.text)['data']
            self.put(path, models)
        models = self.filter_models(models, search=search)
        return {m['id']:m for m in models}
    
    def models(self, search: str = None, update=False):
        return list(self.model2info(search=search,  update=update).keys())

    def get_model_info(self, model):
        model = self.resolve_model(model)
        model2info = self.model2info()
        return model2info[model]
    
    @classmethod
    def filter_models(cls, models, search:str = None):
        if search == None:
            return models
        if isinstance(models[0], str):
            models = [{'id': m} for m in models]
        if ',' in search:
            search = [s.strip() for s in search.split(',')]
        else:
            search = [search]
        models = [m for m in models if any([s in m['id'] for s in search])]
        return [m for m in models]
    
    def test(self):
        params = dict(
        message = 'Hello, how are you?',
        stream = False
        )
        result  =  self.forward(**params)
        assert isinstance(result, str)
        print('Test passed')
        params = dict(
        message = 'Hello, how are you?',
        stream = True
        )
        stream_result = self.forward(**params)
        print(next(stream_result))
        return {'status': 'success', 'params_stream': params, 'params': params, 'result': result, 'stream_result': stream_result}