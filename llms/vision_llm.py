import abc
from typing import List, Union, Dict, Any, Optional
from PIL import Image
import base64
import io
import anthropic
import openai
import os
import llms.model_types as model_types


class VisionLLM:
    def __init__(
        self,
        model_service_type: str,
        api_key: str = None,
        model_path: str = None,
    ):
        self.model_service_type = model_service_type
        self.api_key = api_key
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if self.model_service_type == model_types.ModelServiceType.OPENAI.value:
            return OpenAIVisionModel(self.api_key)
        elif self.model_service_type == model_types.ModelServiceType.CLAUDE.value:
            return ClaudeVisionModel(self.api_key)
        elif self.model_service_type == model_types.ModelServiceType.LOCAL.value:
            return LocalVisionModel(self.model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def chat(
        self,
        user_prompt: str,
        images: List[Image.Image],
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        for image in images:
            messages.append({"role": "user", "content": image})
        return self.model.generate_response(messages)


class BaseVisionModel(abc.ABC):
    @abc.abstractmethod
    def generate_response(
        self, messages: List[Dict[str, Union[str, Image.Image]]]
    ) -> str:
        pass

    def _encode_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")



class OpenAIVisionModel(BaseVisionModel):
    def __init__(self, api_key: str):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # TODO: Initialize OpenAI client here
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_response(
        self, messages: List[Dict[str, Union[str, Image.Image]]]
    ) -> str:
        formatted_messages = []
        user_content = []
        for message in messages:
            if message["role"] == "system":
                formatted_messages.append(
                    {"role": "system", "content": message["content"]}
                )
            elif message["role"] == "user":
                if isinstance(message["content"], str):
                    user_content.append({"type": "text", "text": message["content"]})
                elif isinstance(message["content"], Image.Image):
                    base64_image = self._encode_image(message["content"])
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        }
                    )
        formatted_messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=model_types.OpenAIModelType.GPT4_O.value,
            messages=formatted_messages,
            max_tokens=1024,
        )

        return response.choices[0].message.content


class ClaudeVisionModel(BaseVisionModel):
    def __init__(self, api_key: str):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Client(api_key=api_key)

    def _encode_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_response(
        self, messages: List[Dict[str, Union[str, Image.Image]]]
    ) -> str:
        system_prompt = ""
        formatted_messages = []
        user_content = []

        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] == "user":
                if isinstance(message["content"], str):
                    user_content.append({"type": "text", "text": message["content"]})
                elif isinstance(message["content"], Image.Image):
                    image_data = self._encode_image(message["content"])
                    user_content.extend(
                        [
                            {"type": "text", "text": "Image:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                        ]
                    )
        formatted_messages.append({"role": "user", "content": user_content})

        response = self.client.messages.create(
            model=model_types.ClaudeModelType.CLAUDE35_SONNET,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
        )
        return response.content[0].text


class LocalVisionModel(BaseVisionModel):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # TODO: Load local model here

    def generate_response(
        self, messages: List[Dict[str, Union[str, Image.Image]]]
    ) -> str:
        # TODO: Implement local model inference here
        # This is a placeholder implementation
        return "Local model response placeholder"
