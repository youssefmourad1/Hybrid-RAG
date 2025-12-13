import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Document
from omegaconf import DictConfig
import nest_asyncio

nest_asyncio.apply()

class PDFLoader:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY is required for LlamaParse")
        
        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            verbose=True
        )

    def load(self, file_path: str) -> List[Document]:
        """Loads a PDF and returns a list of Documents (usually 1 per PDF with markdown content)."""
        print(f"Parsing {file_path} with LlamaParse...")
        documents = self.parser.load_data(file_path)
        # Ensure metadata is preserved/added if needed
        for doc in documents:
            doc.metadata["source"] = file_path
        return documents

class Chunker(ABC):
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Any]:
        pass

class FixedChunker(Chunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def chunk(self, documents: List[Document]) -> List[Any]:
        nodes = self.splitter.get_nodes_from_documents(documents)
        return nodes

class SemanticChunker(Chunker):
    def __init__(self, breakpoint_percentile_threshold: int = 95, buffer_size: int = 1, embed_model_name: str = "BAAI/bge-m3"):
        storage_dir = os.environ.get("HF_HOME", "./models") # Ensure we don't redownload unnecessarily
        embed_model = HuggingFaceEmbedding(model_name=embed_model_name, cache_folder=storage_dir)
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size, 
            breakpoint_percentile_threshold=breakpoint_percentile_threshold, 
            embed_model=embed_model
        )

    def chunk(self, documents: List[Document]) -> List[Any]:
        nodes = self.splitter.get_nodes_from_documents(documents)
        return nodes

def get_chunker(cfg: DictConfig) -> Chunker:
    if cfg.chunking.strategy == "fixed":
        return FixedChunker(
            chunk_size=cfg.chunking.chunk_size,
            chunk_overlap=cfg.chunking.chunk_overlap
        )
    elif cfg.chunking.strategy == "semantic":
        return SemanticChunker(
            breakpoint_percentile_threshold=cfg.chunking.breakpoint_percentile_threshold,
            buffer_size=cfg.chunking.buffer_size
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {cfg.chunking.strategy}")
