import os
from litellm import completion, acompletion
import asyncio

class LiteLLM:
    def __init__(self, provider_keys: dict = {}):
        """
        Initialize LiteLLMClient with provider API keys.

        Args:
            provider_keys (dict): API keys for providers, e.g., {'openai': 'key', 'anthropic': 'key'}
        """
        for provider, key in provider_keys.items():
            os.environ[f"{provider.upper()}_API_KEY"] = key

    def forward(self, model: str, messages: list, stream: bool = False):
        """
        Generate completion synchronously.

        Args:
            model (str): Provider and model name (e.g., "openai/gpt-4o").
            messages (list): List of message dicts.
            stream (bool): Stream response or not.

        Returns:
            Completion response.
        """
        response = completion(model=model, messages=messages, stream=stream)
        if stream:
            return (part.choices[0].delta.content or "" for part in response)
        return response.choices[0].message.content

    async def async_forward(self, model: str, messages: list):
        """
        Generate completion asynchronously.

        Args:
            model (str): Provider and model name.
            messages (list): List of message dicts.

        Returns:
            Completion response.
        """
        response = await acompletion(model=model, messages=messages)
        return response.choices[0].message.content

    def set_callbacks(self, callbacks: list):
        """
        Set logging and observability callbacks.

        Args:
            callbacks (list): List of callbacks (e.g., ['lunary', 'mlflow']).
        """
        import litellm
        litellm.success_callback = callbacks
        

    def test(self):
        provider_keys = {
            "openai": "your-openai-key",
            "anthropic": "your-anthropic-key"
        }

        client = LiteLLMClient(provider_keys)

        messages = [{"content": "Hello, how are you?", "role": "user"}]

        # synchronous
        response = client.forward("openai/gpt-4o", messages)
        print(response)

        # asynchronous
        async def main():
            response = await client.async_forward("anthropic/claude-3-sonnet-20240229", messages)
            print(response)

        asyncio.run(main())
