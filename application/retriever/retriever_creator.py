from application.retriever.classic_rag import ClassicRAG


class RetrieverCreator:
    retrievers = {
        "classic": ClassicRAG,
        "default": ClassicRAG,
    }

    @classmethod
    def create_retriever(cls, retriever_type, *args, **kwargs):
        retriever_type = (retriever_type or "default").lower()
        retiever_class = cls.retrievers.get(retriever_type)
        if not retiever_class:
            raise ValueError(f"No retievers class found for type {retriever_type}")
        return retiever_class(*args, **kwargs)
