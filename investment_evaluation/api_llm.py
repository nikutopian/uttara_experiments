import os
import ollama
from enum import Enum
from typing import Any

from groq import Groq
from openai import OpenAI

class LLM:
    def __init__(self, model: Enum):
        self.model = model
        self.seed = 42
        self.temperature = 0.1

    def chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ):
        raise NotImplementedError("should be implemented in sub-class")

    def chat_stream(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ):
        raise NotImplementedError("should be implemented in sub-class")

class ApiLLM(LLM):
    def __init__(self, client: Any, model: Enum):
        super().__init__(model)
        self.client = client

    def chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ):
        response = self.client.chat.completions.create(
            model=self.model.value,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=self.seed,
            temperature=self.temperature,
            response_format=(
                {"type": "json_object"} if force_json_format else {"type": "text"}
            ),
        )
        return response

    def chat_stream(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ):
        response_stream = self.client.chat.completions.create(
            model=self.model.value,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=self.seed,
            temperature=self.temperature,
            stream=True,
            response_format=(
                {"type": "json_object"} if force_json_format else {"type": "text"}
            ),
        )
        for response in response_stream:
            yield response.choices[0].delta.content


class OpenAIModelType(Enum):
    GPT3_5_TURBO = "gpt-3.5-turbo"
    GPT4_TURBO = "gpt-4-0125-preview"
    GPT4_O = "gpt-4o"

class OpenAILLM(ApiLLM):
    def __init__(self, model: OpenAIModelType):
        super().__init__(
            client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), model=model
        )

class GroqModelType(Enum):
    MIXTRAL_8X7B = "mixtral-8x7b-32768"

class GroqLLM(ApiLLM):
    def __init__(self, model: GroqModelType) -> None:
        super().__init__(
            client=Groq(api_key=os.environ.get("GROQ_API_KEY")), model=model
        )

class DeepInfraModelType(Enum):
    MIXTRAL_8X7B_INSTRUCT = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    LLAMA_2_70B_CHAT = "meta-llama/Llama-2-70b-chat-hf"
    LLAMA_2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
    GEMMA_7B = "google/gemma-7b-it"

class DeepInfraLLM(ApiLLM):
    def __init__(self, model: DeepInfraModelType) -> None:
        super().__init__(
            client=OpenAI(
                api_key=os.environ.get("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai",
            ),
            model=model,
        )

class LocalLLM(LLM):
    def __init__(self, model: Enum):
        super().__init__(model)
        ollama.pull(self.model.value)

    def chat_internal(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
        stream: bool = False,
    ):
        assert user_prompt != "", "User prompt must be provided"

        response = ollama.chat(
            model=self.model.value,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format="json" if force_json_format else "",
            stream=stream,
            options={"temperature": self.temperature, "seed": self.seed},
        )
        
        return response
    
    def chat(
        self,
        system_prompt: str = None,
        user_prompt: str = None,
        force_json_format: bool = False,
    ):
        response = self.chat_internal(system_prompt=system_prompt, user_prompt=user_prompt, force_json_format=force_json_format)
        return response['message']['content']
        
    def chat_stream(
        self,
        system_prompt: str = None,
        user_prompt: str = None,
        force_json_format: bool = False,
    ):
        response = self.chat_internal(system_prompt=system_prompt, user_prompt=user_prompt, force_json_format=force_json_format, stream=True)
        for chunk in response:
                yield chunk['message']['content']
class OllamaModelType(Enum):
    GEMMA_2B = "gemma:2b"
    LLAMA2_7B = "llama2"
    MISTRAL_7B = "mistral"
    MIXTRAL_8X7B = "mixtral"
class OllamaModel(LocalLLM):
    def __init__(self, model: OllamaModelType):
        super().__init__(model=model)
