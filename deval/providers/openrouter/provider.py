import os
import json
import random
import requests
import openai
from dotenv import load_dotenv

class OpenRouter:
    def __init__(
        self,
        api_key: str = 'OPENROUTER_API_KEY',
        base_url: str = 'https://openrouter.ai/api/v1',
        timeout: float = None,
        max_retries: int = 10,
        storage_path = '~/.deval/model/openrouter',
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
            storage_path (str, optional): The path to store the models. Defaults to '~/.deval/model/openrouter'.
        """
        # Load environment variables from .env file in the current working directory

    
        self.storage_path = os.path.abspath(os.path.expanduser(storage_path)) # path to store models (relative to storage_path) 
        self.api_key_path = f'{self.storage_path}/api.json' # path to store api keys (relative to storage_path)
        self.base_url = base_url
        self.api_key= self.get_api_key(api_key)
        # Use API key from parameters, or from environment variable, or from stored keys

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key, 
            timeout=timeout,
            max_retries=max_retries,
        )

    def forward(
        self,
        message: str,
        *extra_text , 
        history: list = None,
        stream: bool = False,
        model:str = 'anthropic/claude-3.7-sonnet',
        max_tokens: int = 10000000,
        temperature: float = 0,
        verbose: bool = False,

        **kwargs
    ) -> str :
        """
        Generates a response using the OpenAI language providers.

        Args:
            message (str): The message to send to the language providers.
            history (ChatHistory): The conversation history.
            stream (bool): Whether to stream the response or not.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The sampling temperature to use.

        Returns:
        Generator[str] | str: A generator for streaming responses or the full streamed response.
        """
        message = str(message)
        if len(extra_text) > 0:
            message = message + ' '.join(extra_text)
        history = history or []
        model = self.get_model(model)
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
                    token = token.choices[0].delta.content
                    if verbose:
                        print(token, end='', flush=True)
                    yield token
            return stream_generator(result)
        else:
            return result.choices[0].message.content
        
    def get_model(self, model=None):
        models =  self.models()
        model = str(model)
        if str(model) not in models:
            if ',' in model:
                models = [m for m in models if any([s in m for s in providers.split(',')])]
            else:
                models = [m for m in models if str(model) in m]
            print(f"Model {model} not found. Using {models} instead.")
            assert len(models) > 0
            model = models[0]

        return model

    def get_json(self, path, default=None , update=False):
        if not os.path.exists(path) and not update:
            return default
        else:
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, str):
                    data = json.loads(data)
            return data

    def put_json(self, path, data):
        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        with open(path, 'w') as f:
            json.dump(data, f)
        return {'status': 'success', 'path': path}

    def get_api_key(self, api_key='OPENROUTER_API_KEY', save_key_if_not_found=True):
        """
        get the api keys
        """
        keys = self.get_json(self.api_key_path, [])
        load_dotenv()
        if isinstance(api_key, str):
            env_dict = os.environ
            if api_key in env_dict:
                env_var_found = True
                # how to change the color of the text in the terminal
                api_key = env_dict[api_key]
            if env_var_found:
                if save_key_if_not_found:
                    keys.append(api_key)
                    keys = list(set(keys))
                    self.put_json(self.api_key_path, keys)
        assert len(keys) > 0, f'No API key found. Please set the {api_key} environment variable or add a key to {self.api_key_path}'
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
        return self.get_json(self.api_key_path, [])

    def add_key(self, key):
        keys = self.keys()
        keys.append(key)
        keys = list(set(keys))
        self.put_json(self.api_key_path, keys)
        return keys

    def resolve_path(self, path):
        return 

    def model2info(self, search: str = None, update=False):
        path =  f'{self.storage_path}/models.json'
        models = self.get_json(path, default={}, update=update)
        if len(models) == 0:
            response = requests.get_json(self.base_url + '/models')
            models = json.loads(response.text)['data']
            self.put_json(path, models)
        models = self.filter_models(models, search=search)
        return {m['id']:m for m in models}
    
    def models(self, search: str = None, update=False):
        return list(self.model2info(search=search,  update=update).keys())

    def get_model_info(self, model):
        model = self.get_model(model)
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
    
    def test(self, **kwargs):
        self = self.__class__(**kwargs)
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