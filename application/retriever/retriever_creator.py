from application.retriever.classic_rag import ClassicRAG


class RetrieverCreator:
    @classmethod
    def create_retriever(cls, type, *args, **kwargs):
        retriever_type = (type or "default").lower()
        retriever_class = cls.retrievers.get(retriever_type)
        if not retriever_class:
            raise ValueError(f"No retrievers class found for type {type}")
        return retriever_class(*args, **kwargs)

RETRIEVERS = {
    "classic": ClassicRAG,
    "default": ClassicRAG,
}
