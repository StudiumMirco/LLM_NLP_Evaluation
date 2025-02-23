from abc import ABC
import torch
import os
import utils


# Based on promptbench

class LMMBaseModel(ABC):
    """
    Abstract base class for language model interfaces.

    This class provides a common interface for various language models and includes methods for prediction.

    Parameters:
    -----------
    model_name : str
        The name of the language model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    device : str
        The device to use for inference (default is 'auto').

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text.
    __call__(input_text, **kwargs)
        Shortcut for the predict method.
    """

    def __init__(self, model_name, max_new_tokens, temperature, device='auto'):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

    def predict(self, input_text, **kwargs):
        # Determine the device to use: if 'auto', choose 'cuda' if available, otherwise 'cpu'
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device

        # Tokenize the input text and move it to the selected device.
        # Note: self.tokenizer and self.model should be defined in subclasses.
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)

        # Generate tokens using the model's generate method.
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            **kwargs
        )

        # Decode the generated tokens to obtain the output text.
        out = self.tokenizer.decode(outputs[0])
        return out

    def __call__(self, input_text, **kwargs):
        # Allow the instance to be called directly to generate predictions.
        return self.predict(input_text, **kwargs)


class LLamaModel(LMMBaseModel):
    """
    Language model class for interfacing with Meta's LLama models.

    Inherits from LMMBaseModel and sets up a model interface for LLama models.

    Parameters:
    -----------
    model_name : str
        The name of the LLama model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    """

    def __init__(self, model_name, max_new_tokens, temperature=0):
        super(LLamaModel, self).__init__(model_name, max_new_tokens, temperature)

    def predict(self, input_text, **kwargs):

        from huggingface_hub import InferenceClient
        # Create an inference client using the API key from environment variables.
        client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

        # Format the input text into a list of message dictionaries.
        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]

        # Use parameters from kwargs if provided, otherwise use the default object attributes.
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens
        stream = kwargs['stream'] if 'stream' in kwargs else False

        # Build the request payload for the LLama model.
        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "stream": stream,
        }

        # Generate a response using the LLama model via the inference client.
        response = client.chat.completions.create(**request_payload)

        # If streaming is enabled, accumulate all chunks; otherwise, return the complete response.
        if stream:
            result = ""
            for chunk in response:
                result += chunk.choices[0].delta.content
        else:
            result = response.choices[0].message.content

        return result

    def predict_multi_turn(self, messages, **kwargs):
        # Wrapper for handling multi-turn conversations; delegates to the predict method.
        return self.predict(messages, **kwargs)


class OpenAIModel(LMMBaseModel):
    """
    Language model class for interfacing with OpenAI's GPT models.

    Inherits from LMMBaseModel and sets up a model interface for OpenAI GPT models.

    Parameters:
    -----------
    model_name : str
        The name of the OpenAI model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation (default is 0).
    system_prompt : str
        The system prompt to be used (default is None).

    Methods:
    --------
    predict(input_text, **kwargs)
        Generates a prediction based on the input text using the OpenAI model.
    """

    def __init__(self, model_name, max_new_tokens, temperature, system_prompt):
        super(OpenAIModel, self).__init__(model_name, max_new_tokens, temperature)
        self.system_prompt = system_prompt

    def predict(self, input_text, **kwargs):
        from openai import OpenAI

        # Create an OpenAI client using the API key from environment variables.
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_2"))

        # Prepare the system message if applicable.
        if self.system_prompt is None and self.model_name != "o1-preview":
            system_messages = {'role': "system", 'content': "You are a helpful assistant."}
        elif self.model_name != "o1-preview":
            system_messages = {'role': "system", 'content': self.system_prompt}

        # Format the input text into a list of message dictionaries.
        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]

        # For models other than "o1-preview", insert the system message at the beginning.
        if self.model_name != "o1-preview":
            messages.insert(0, system_messages)

        # Use extra parameters from kwargs if provided; defaults are taken from the object attributes.
        n = kwargs['n'] if 'n' in kwargs else 1  # Number of completions to generate.
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens
        response_format = kwargs.get('response_format', None)

        # Build the request payload for the OpenAI model.
        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_new_tokens,
            "n": n
        }
        # Specify JSON response format if requested.
        if response_format == "json":
            request_payload["response_format"] = {"type": "json_object"}

        # For GPT models (excluding "o1-preview"), make the standard API call.
        if self.model_name != "o1-preview":
            response = client.chat.completions.create(**request_payload)
        else:
            # For the "o1-preview" model, temperature and max tokens cannot be changed and JSON format is not supported.
            request_payload = {
                "model": self.model_name,
                "messages": messages,
                "n": n
            }
            response = client.chat.completions.create(**request_payload)

        # Return multiple completions if requested, otherwise return the single result.
        if n > 1:
            result = [choice.message.content for choice in response.choices]
        else:
            result = response.choices[0].message.content

        return result

    def predict_multi_turn(self, messages, **kwargs):
        # Wrapper for multi-turn conversations; delegates to the predict method.
        return self.predict(messages, **kwargs)


class GeminiModel(LMMBaseModel):
    """
    Language model class for interfacing with Google's Gemini models.

    Inherits from LMMBaseModel and sets up a model interface for Gemini models.

    Parameters:
    -----------
    model_name : str
        The name of the Gemini model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    """

    def __init__(self, model_name, max_new_tokens, temperature=0):
        super(GeminiModel, self).__init__(model_name, max_new_tokens, temperature)

    def predict(self, input_text, **kwargs):
        import google.generativeai as genai

        # Configure the Google Generative AI client with the API key from environment variables.
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

        # Set up the generation configuration using kwargs or default values.
        generation_config = {
            "temperature": kwargs.get('temperature', self.temperature),
            # TODO: Clarify the parameters 'top_p' and 'top_k'
            "top_p": kwargs.get('top_p', 1),
            "top_k": kwargs.get('top_k', 1),
            "max_output_tokens": kwargs.get('max_new_tokens', self.max_new_tokens)
        }

        # Define safety settings to mitigate harmful outputs.
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

        # Initialize the Gemini generative model with the given configuration and safety settings.
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # Generate content using the Gemini model and extract the text output.
        response = model.generate_content(input_text).text

        return response

    def predict_multi_turn(self, messages, **kwargs):
        # Parse messages using the helper function from utils to convert "content" to "parts".
        messages = utils.parse_messages_for_gemini(messages)
        return self.predict(messages, **kwargs)


class AnthropicModel(LMMBaseModel):
    """
    Language model class for interfacing with Anthropic's models.

    Inherits from LMMBaseModel and sets up a model interface for Anthropic models.

    Parameters:
    -----------
    model_name : str
        The name of the Anthropic model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float, optional
        The temperature for text generation (default is 0).
    """

    def __init__(self, model_name, max_new_tokens, temperature=0):
        super(AnthropicModel, self).__init__(model_name, max_new_tokens, temperature)

    def predict(self, input_text, **kwargs):
        import anthropic

        # Create an Anthropic client using the API key from environment variables.
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        # Format the input text into a list of message dictionaries.
        if isinstance(input_text, list):
            messages = input_text
        elif isinstance(input_text, dict):
            messages = [input_text]
        else:
            messages = [{"role": "user", "content": input_text}]

        # Use extra parameters from kwargs if provided, otherwise use default values.
        temperature = kwargs['temperature'] if 'temperature' in kwargs else self.temperature
        max_new_tokens = kwargs['max_new_tokens'] if 'max_new_tokens' in kwargs else self.max_new_tokens

        # Build the request payload for the Anthropic model.
        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }

        # Generate a message response using the Anthropic client.
        message = client.messages.create(**request_payload)

        # Return only the text content of the first message in the response.
        return message.content[0].text

    def predict_multi_turn(self, messages, **kwargs):
        # Wrapper for multi-turn conversations; delegates to the predict method.
        return self.predict(messages, **kwargs)
