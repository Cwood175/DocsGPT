from application.core.settings import settings
from application.vectorstore.base import BaseVectorStore
from application.vectorstore.document_class import Document


class ElasticsearchStore(BaseVectorStore):
    def __init__(self, source_id: str, embeddings_key: str, index_name: str = settings.ELASTIC_INDEX) -> None:
        super().__init__()
        self.source_id = source_id.replace("application/indexes/", "").rstrip("/")
        self.embeddings_key = embeddings_key
        self.index_name = index_name
        self._es_connection = None
        self.docsearch = self._connect()

    def _connect(self):
        if self._es_connection is None:
            connection_params = {}
            if settings.ELASTIC_URL:
                connection_params["hosts"] = [settings.ELASTIC_URL]
                connection_params["http_auth"] = (settings.ELASTIC_USERNAME, settings.ELASTIC_PASSWORD)
            elif settings.ELASTIC_CLOUD_ID:
                connection_params["cloud_id"] = settings.ELASTIC_CLOUD_ID
                connection_params["basic_auth"] = (settings.ELASTIC_USERNAME, settings.ELASTIC_PASSWORD)
            else:
                raise ValueError("Please provide either elasticsearch_url or cloud_id.")

            import elasticsearch
            self._es_connection = elasticsearch.Elasticsearch(**connection_params)
        return self._es_connection

    def connect_to_elasticsearch(
        self,
        *,
        es_url: str | None = None,
        cloud_id: str | None = None,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> Any:
        try:
            import elasticsearch
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        if es_url and cloud_id:
            raise ValueError(
                "Both es_url and cloud_id are defined. Please provide only one."
            )

        connection_params = {}

        if es_url:
            connection_params["hosts"] = [es_url]
        elif cloud_id:
            connection_params["cloud_id"] = cloud_id
        else:
            raise ValueError("Please provide either elasticsearch_url or cloud_id.")

        if api_key:
            connection_params["api_key"] = api_key
        elif username and password:
            connection_params["basic_auth"] = (username, password)

        es_client = elasticsearch.Elasticsearch(
            **connection_params,
        )
        try:
            es_client.info()
        except Exception as e:
            raise e

        return es_client

    def search(self, question: str, k: int = 2, index_name: str = settings.ELASTIC_INDEX) -> list[Document]:
        embeddings = self._get_embeddings(settings.EMBEDDINGS_NAME, self.embeddings_key)
        vector = embeddings.embed_query(question)
        knn = {
            "filter": [{"match": {"metadata.source_id.keyword": self.source_id}}],
            "field": "vector",
            "k": k,
            "num_candidates": 100,
            "query_vector": vector,
        }
        full_query = {
            "knn": knn,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "text": {
                                    "query": question,
                                }
                            }
                        }
                    ],
                    "filter": [{"match": {"metadata.source_id.keyword": self.source_id}}],
                }
            },
            "rank": {"rrf": {}},
        }
        resp = self.docsearch.search(index=self.index_name, query=full_query['query'], size=k, knn=full_query['knn'])
        # create Documents objects from the results page_content ['_source']['text'], metadata ['_source']['metadata']
        doc_list = []
        for hit in resp['hits']['hits']:
            
            doc_list.append(Document(page_content = hit['_source']['text'], metadata = hit['_source']['metadata']))
        return doc_list

    def _create_index_if_not_exists(
            self, index_name: str, dims_length: int
        ) -> None:

        if self._es_connection.indices.exists(index=index_name):
            print(f"Index {index_name} already exists.")

        else:

            indexSettings = self.index(
                dims_length=dims_length,
            )
            self._es_connection.indices.create(index=index_name, **indexSettings)

    def index(
            self,
            dims_length: int,
        ) -> dict:
        return {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": dims_length,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        refresh_indices: bool = True,
        create_index_if_not_exists: bool = True,
        bulk_kwargs: dict | None = None,
    ) -> list[str]:
        
        bulk_kwargs = bulk_kwargs or {}
        import uuid
        embeddings = []
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        requests = []
        embeddings = self._get_embeddings(settings.EMBEDDINGS_NAME, self.embeddings_key)

        vectors = embeddings.embed_documents(list(texts))

        dims_length = len(vectors[0])

        if create_index_if_not_exists:
            self._create_index_if_not_exists(
                index_name=self.index_name, dims_length=dims_length
            )

        for i, (text, vector) in enumerate(zip(texts, vectors)):
            metadata = metadatas[i] if metadatas else {}

            requests.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "text": text,
                    "vector": vector,
                    "metadata": metadata,
                    "_id": ids[i],
                }
            )


        if len(requests) > 0:
            from elasticsearch.helpers import BulkIndexError, bulk
            try:
                success, failed = bulk(
                    self._es_connection,
                    requests,
                    stats_only=True,
                    refresh=refresh_indices,
                    **bulk_kwargs,
                )
                return ids
            except BulkIndexError as e:
                print(f"Error adding texts: {e}")
                firstError = e.errors[0].get("index", {}).get("error", {})
                print(f"First error reason: {firstError.get('reason')}")
                raise e

        else:
            return []

    def delete_index(self) -> None:
        self._es_connection.delete_by_query(index=self.index_name, query={"match": {
                                      "metadata.source_id.keyword": self.source_id}},)
