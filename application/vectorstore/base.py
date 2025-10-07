import logging
import os
from abc import ABC, abstractmethod

from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

from application.core.settings import settings


class EmbeddingsWrapper:
    def __init__(self, model_name, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing EmbeddingsWrapper with model: {model_name}")
        try:
            kwargs.setdefault("trust_remote_code", True)
            self.model = SentenceTransformer(
                model_name,
                *args,
                config_kwargs={"allow_dangerous_deserialization": True},
                **kwargs,
            )
            if self.model is None or self.model._first_module() is None:
                raise ValueError(f"SentenceTransformer model failed to load properly for: {model_name}")
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Successfully loaded model with dimension: {self.dimension}")
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            logger.exception(f"Failed to initialize SentenceTransformer with model {model_name}")
            raise

    def embed_query(self, query: str):
        return self.model.encode(query).tolist()

    def embed_documents(self, documents: list):
        return self.model.encode(documents).tolist()

    def __call__(self, text):
        if isinstance(text, str):
            return self.embed_query(text)
        elif isinstance(text, list):
            return self.embed_documents(text)
        else:
            raise ValueError("Input must be a string or a list of strings")


class EmbeddingsSingleton:
    _instances: ClassVar[dict] = {}

    @staticmethod
    def get_instance(embeddings_name, *args, **kwargs):
        if embeddings_name not in EmbeddingsSingleton._instances:
            EmbeddingsSingleton._instances[embeddings_name] = EmbeddingsSingleton._create_instance(
                embeddings_name, *args, **kwargs
            )
        return EmbeddingsSingleton._instances[embeddings_name]

    @staticmethod
    def _create_instance(embeddings_name, *args, **kwargs):
        embeddings_factory = {
            "openai_text-embedding-ada-002": OpenAIEmbeddings,
            "huggingface_sentence-transformers/all-mpnet-base-v2": lambda: EmbeddingsWrapper(
                "sentence-transformers/all-mpnet-base-v2"
            ),
            "huggingface_sentence-transformers-all-mpnet-base-v2": lambda: EmbeddingsWrapper(
                "sentence-transformers/all-mpnet-base-v2"
            ),
            "huggingface_hkunlp/instructor-large": lambda: EmbeddingsWrapper("hkunlp/instructor-large"),
        }

        if embeddings_name in embeddings_factory:
            return embeddings_factory[embeddings_name](*args, **kwargs)
        else:
            return EmbeddingsWrapper(embeddings_name, *args, **kwargs)


class BaseVectorStore(ABC):
    def __init__(self):
        return None

    @abstractmethod
    def search(self, *args, **kwargs):
        """Search for similar documents/chunks in the vectorstore"""
        raise NotImplementedError

    @abstractmethod
    def add_texts(self, texts, metadatas=None, *args, **kwargs):
        """Add texts with their embeddings to the vectorstore"""
        raise NotImplementedError

    def delete_index(self, *args, **kwargs):
        """Delete the entire index/collection"""
        return

    def save_local(self, *args, **kwargs):
        """Save vectorstore to local storage"""
        return

    def get_chunks(self, *args, **kwargs):
        """Get all chunks from the vectorstore"""
        return

    def add_chunk(self, text, metadata=None, *args, **kwargs):
        """Add a single chunk to the vectorstore"""
        return

    def delete_chunk(self, chunk_id, *args, **kwargs):
        """Delete a specific chunk from the vectorstore"""
        return

    def is_azure_configured(self):
        return settings.OPENAI_API_BASE and settings.OPENAI_API_VERSION and settings.AZURE_DEPLOYMENT_NAME

    def _get_embeddings(self, embeddings_name, embeddings_key=None):
        logger = logging.getLogger(__name__)
        if embeddings_name == "openai_text-embedding-ada-002":
            if self.is_azure_configured():
                os.environ["OPENAI_API_TYPE"] = "azure"
                embedding_instance = EmbeddingsSingleton.get_instance(
                    embeddings_name, model=settings.AZURE_EMBEDDINGS_DEPLOYMENT_NAME
                )
            else:
                embedding_instance = EmbeddingsSingleton.get_instance(embeddings_name, openai_api_key=embeddings_key)
        elif embeddings_name == "huggingface_sentence-transformers/all-mpnet-base-v2":
            possible_paths = [
                "/app/models/all-mpnet-base-v2",  # Docker absolute path
                "./models/all-mpnet-base-v2",  # Relative path
            ]
            local_model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    local_model_path = path
                    logger.info(f"Found local model at path: {path}")
                    break
                else:
                    logger.info(f"Path does not exist: {path}")
            if local_model_path:
                embedding_instance = EmbeddingsSingleton.get_instance(
                    local_model_path,
                )
            else:
                logger.warning(
                    f"Local model not found in any of the paths: {possible_paths}. Falling back to HuggingFace download."
                )
                embedding_instance = EmbeddingsSingleton.get_instance(
                    embeddings_name,
                )
        else:
            embedding_instance = EmbeddingsSingleton.get_instance(embeddings_name)
        return embedding_instance
