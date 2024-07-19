import os
from enum import Enum
from typing import Any, Tuple

import cohere
import google.generativeai as genai
import ollama
from anthropic import Anthropic
from groq import Groq
from openai import OpenAI
from model_types import (
    OpenAIModelType,
    ClaudeModelType,
    GroqModelType,
    DeepInfraModelType,
    OllamaModelType,
    CohereModelType,
)


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
        return response.choices[0].message.content

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
    GPT3_5_TURBO = "gpt-3.5-turbo-1106"
    GPT4_TURBO = "gpt-4-0125-preview"
    GPT4_O = "gpt-4o"


class OpenAILLM(ApiLLM):
    def __init__(self, model: OpenAIModelType):
        super().__init__(
            client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), model=model
        )


class GroqLLM(ApiLLM):
    def __init__(self, model: GroqModelType) -> None:
        super().__init__(
            client=Groq(api_key=os.environ.get("GROQ_API_KEY")), model=model
        )


class DeepInfraLLM(ApiLLM):
    def __init__(self, model: DeepInfraModelType) -> None:
        super().__init__(
            client=OpenAI(
                api_key=os.environ.get("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai",
            ),
            model=model,
        )


class ClaudeLLM(ApiLLM):
    def __init__(self, model: ClaudeModelType) -> None:
        super().__init__(
            client=Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            ),
            model=model,
        )

    def chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ):
        response = self.client.messages.create(
            model=self.model.value,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=4096,
        )
        return response.content[0].text

    def chat_stream(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ):
        response_stream = self.client.messages.create(
            model=self.model.value,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=4096,
            stream=True,
        )
        for response in response_stream:
            if response.type == "content_block_delta":
                yield response.delta.text


class CohereLLM(ApiLLM):
    def __init__(self, model: CohereModelType) -> None:
        super().__init__(
            client=cohere.Client(
                api_key=os.environ.get("COHERE_API_KEY"),
            ),
            model=model,
        )

    def chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ):
        response = self.client.chat(
            model=self.model.value,
            preamble_override=system_prompt,
            message=user_prompt,
        )
        return response.text

    def chat_stream(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ):
        response_stream = self.client.chat(
            model=self.model.value,
            preamble_override=system_prompt,
            message=user_prompt,
            stream=True,
        )
        for response in response_stream:
            if response.event_type == "text-generation":
                yield response.text


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
        assert user_prompt is not None, "User prompt must be provided"

        messages = [
            {"role": "user", "content": user_prompt},
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

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
        if stream:
            for chunk in response:
                yield chunk["message"]["content"]
        else:
            return response["message"]["content"]

    def chat(
        self,
        system_prompt: str = None,
        user_prompt: str = None,
        force_json_format: bool = False,
    ):
        return self.chat_internal(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            force_json_format=force_json_format,
            stream=False,
        )

    def chat_stream(
        self,
        system_prompt: str = None,
        user_prompt: str = None,
        force_json_format: bool = False,
    ):
        return self.chat_internal(
            system_prompt=system_prompt, user_prompt=user_prompt, stream=True
        )


class OllamaModel(LocalLLM):
    def __init__(self, model: OllamaModelType):
        super().__init__(model=model)


class GoogleModelType(Enum):
    GEMINI_PRO_1_5 = "gemini-1.5-pro-latest"
    GEMINI_FLASH_1_5 = "gemini-1.5-flash-latest"
    GEMINI_PRO_VISION = "gemini-pro-vision"


class GoogleLLM(LLM):
    def __init__(self, model: GoogleModelType) -> None:
        super().__init__(model)
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    def _chat_client(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ) -> Tuple[genai.GenerativeModel, str]:
        generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            response_mime_type=(
                "application/json" if force_json_format else "text/plain"
            ),
        )
        client = genai.GenerativeModel(
            model_name=self.model.value, generation_config=generation_config
        )
        message = f"System Prompt: {system_prompt}\n\nUser Prompt: {user_prompt}"
        return client, message

    def chat(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ) -> str:
        client, message = self._chat_client(
            system_prompt, user_prompt, force_json_format
        )
        response = client.generate_content(message)
        return response.text

    def chat_stream(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        force_json_format: bool = False,
    ):
        client, message = self._chat_client(
            system_prompt, user_prompt, force_json_format
        )
        response_stream = client.generate_content(message, stream=True)
        for chunk in response_stream:
            yield chunk.text
