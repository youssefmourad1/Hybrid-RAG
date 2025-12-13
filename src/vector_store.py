import os
from typing import List, Dict, Any, Optional
from uuid import uuid4
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from llama_index.core.schema import TextNode
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from omegaconf import DictConfig

class HybridQdrantClient:
    def __init__(self, cfg: DictConfig):
        if cfg.qdrant.host == ":memory:":
             self.client = QdrantClient(location=":memory:")
        else:
             self.client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)

        self.collection_name = cfg.qdrant.collection_name
        self.dense_model_name = cfg.retrieval.dense_model
        # Initialize embedding models
        # Note: BGE-M3 is supported by SentenceTransformer
        print(f"Loading Dense Model: {self.dense_model_name}")
        self.dense_model = SentenceTransformer(self.dense_model_name)
        
        # init sparse model if hybrid
        self.use_sparse = cfg.retrieval.strategy == "hybrid"
        if self.use_sparse:
             # Using BGE-M3 for sparse if possible, or fallback to SPLADE which is reliable in FastEmbed
             self.sparse_model_name = "prithivida/Splade_PP_En_v1" # Standard good sparse
             print(f"Loading Sparse Model: {self.sparse_model_name}")
             self.sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)

        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection {self.collection_name}")
            vectors_config = {
                "dense": VectorParams(size=1024, distance=Distance.COSINE) # BGE-M3 is 1024
            }
            sparse_vectors_config = {}
            if self.use_sparse:
                sparse_vectors_config["sparse"] = SparseVectorParams()

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )

    def index(self, nodes: List[TextNode]):
        documents = [node.get_content() for node in nodes]
        metadatas = [node.metadata for node in nodes]
        ids = [node.node_id for node in nodes]

        # Generate Dense Embeddings
        print("Generating Dense Embeddings...")
        dense_embeddings = self.dense_model.encode(documents, convert_to_numpy=True)
        
        points = []
        
        sparse_embeddings = None
        if self.use_sparse:
            print("Generating Sparse Embeddings...")
            sparse_embeddings = list(self.sparse_model.embed(documents))

        for i in range(len(nodes)):
            vector = {"dense": dense_embeddings[i].tolist()}
            if self.use_sparse:
                # NOTE: check the slicing of sparse_embeddings[i] 
                vector["sparse"] = Models.SparseVector(
                    indices=sparse_embeddings[i].indices.tolist(), 
                    values=sparse_embeddings[i].values.tolist()
                )
            
            points.append(models.PointStruct(
                id=ids[i], # Qdrant prefers UUID or int
                vector=vector,
                payload={ "text": documents[i], **metadatas[i] }
            ))
            
        print(f"Upserting {len(points)} points to Qdrant...")
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        # NOTE FOR ABDELLAH: This will be used by retrieval.py
        # Dense Search
        dense_gen = self.dense_model.encode([query], convert_to_numpy=True)[0]
        dense_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedVector(name="dense", vector=dense_gen.tolist()),
            limit=top_k,
            with_payload=True
        )
        
        results = {"dense": dense_results, "sparse": []}

        if self.use_sparse:
            sparse_gen = list(self.sparse_model.embed([query]))[0]
            sparse_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=models.NamedSparseVector(
                    name="sparse", 
                    vector=models.SparseVector(
                        indices=sparse_gen.indices.tolist(),
                        values=sparse_gen.values.tolist()
                    )
                ),
                limit=top_k,
                with_payload=True
            )
            results["sparse"] = sparse_results
            
        return results
