from enum import Enum

class ModelServiceType(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL = "local"

class ClaudeModelType(Enum):
    CLAUDE3_OPUS = "claude-3-opus-20240229"
    CLAUDE3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE35_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE3_HAIKU = "claude-3-haiku-20240307"

class OpenAIModelType(Enum):
    GPT3_5_TURBO = "gpt-3.5-turbo-1106"
    GPT4_TURBO = "gpt-4-0125-preview"
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"

class DeepInfraModelType(Enum):
    MIXTRAL_8X7B_INSTRUCT = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    LLAMA_2_70B_CHAT = "meta-llama/Llama-2-70b-chat-hf"
    LLAMA_2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
    GEMMA_7B = "google/gemma-7b-it"

class GroqModelType(Enum):
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    LLAMA_3_70B = "llama3-70b-8192"

class CohereModelType(Enum):
    CommandR = "command-r"

class OllamaModelType(Enum):
    GEMMA_2B = "gemma:2b"
    LLAMA2_7B = "llama2"
    LLAMA3_7B = "llama3"
    MISTRAL_7B = "mistral"
    MIXTRAL_8X7B = "mixtral"

