import torch
from typing import List, Dict, Any
from collections import defaultdict
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self, vector_store, cfg):
        self.vector_store = vector_store
        self.cfg = cfg
        self.reranker_model_name = cfg.retrieval.reranker
        print(f"Loading Reranker: {self.reranker_model_name}")
        # NOTE: trust_remote_code=True might be needed for some BGE models, 
        self.reranker = CrossEncoder(
            self.reranker_model_name, 
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.top_k_retrieval = cfg.retrieval.top_k_retrieval
        self.top_k_final = cfg.retrieval.top_k_final

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        # 1. First stage retrieval (Dense + Sparse if hybrid)
        results = self.vector_store.search(query, top_k=self.top_k_retrieval)
        
        # 2. Merge (RRF or just dense)
        if self.cfg.retrieval.strategy == "hybrid":
            merged_results = self._rrf_merge(results["dense"], results["sparse"])
        else:
            # Just dense results, reformatted
            merged_results = self._normalize_results(results["dense"])

        # 3. Rerank
        reranked_results = self._rerank(query, merged_results)
        
        return reranked_results[:self.top_k_final]

    def _normalize_results(self, qdrant_points: List[Any]) -> List[Dict[str, Any]]:
        """Converts Qdrant points to a standard dict format."""
        normalized = []
        for point in qdrant_points:
            normalized.append({
                "id": point.id,
                "score": point.score,
                "text": point.payload.get("text", ""),
                "metadata": point.payload
            })
        return normalized

    def _rrf_merge(self, dense_results: List[Any], sparse_results: List[Any], k: int = 60) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion."""
        # Maps doc_id -> RRF score
        rrf_scores = defaultdict(float)
        
        # Helper to map id -> payload (to reconstruct doc later)
        doc_map = {}

        for rank, point in enumerate(dense_results):
            rrf_scores[point.id] += 1 / (k + rank + 1)
            doc_map[point.id] = point

        for rank, point in enumerate(sparse_results):
            rrf_scores[point.id] += 1 / (k + rank + 1)
            if point.id not in doc_map:
                doc_map[point.id] = point
        
        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        merged = []
        for doc_id in sorted_ids:
            point = doc_map[doc_id]
            merged.append({
                "id": doc_id,
                "score": rrf_scores[doc_id], # RRF score, not original similarity
                "text": point.payload.get("text", ""),
                "metadata": point.payload
            })
            
        return merged

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
            
        pairs = [[query, doc["text"]] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])
            
        # Sort by new rerank score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates
