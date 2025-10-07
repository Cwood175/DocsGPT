from application.llm.anthropic import AnthropicLLM
from application.llm.docsgpt_provider import DocsGPTAPILLM
from application.llm.google_ai import GoogleLLM
from application.llm.groq import GroqLLM
from application.llm.huggingface import HuggingFaceLLM
from application.llm.llama_cpp import LlamaCpp
from application.llm.novita import NovitaLLM
from application.llm.openai import OpenAILLM, AzureOpenAILLM
from application.llm.premai import PremAILLM
from application.llm.sagemaker import SagemakerAPILLM


LLM_CLASSES = {
    "openai": OpenAILLM,
    "azure_openai": AzureOpenAILLM,
    "sagemaker": SagemakerAPILLM,
    "huggingface": HuggingFaceLLM,
    "llama.cpp": LlamaCpp,
    "anthropic": AnthropicLLM,
    "docsgpt": DocsGPTAPILLM,
    "premai": PremAILLM,
    "groq": GroqLLM,
    "google": GoogleLLM,
    "novita": NovitaLLM,
}

class LLMCreator:
    @classmethod
    def create_llm(cls, type, api_key, user_api_key, decoded_token, *args, **kwargs):
        llm_class = LLM_CLASSES.get(type.lower())
        if not llm_class:
            raise ValueError(f"No LLM class found for type {type}")
        return llm_class(
            api_key, user_api_key, decoded_token=decoded_token, *args, **kwargs
        )
