import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
)

import hydra
import logging
from omegaconf import DictConfig
from src.ingestion import PDFLoader, get_chunker
from src.vector_store import HybridQdrantClient
from src.retrieval import HybridRetriever
from src.generation import InferenceEngine
from datasets import load_dataset
import pandas as pd
import os
import glob
from tqdm import tqdm

log = logging.getLogger(__name__)

def get_collection_name(cfg):
    # Dynamic collection name based on chunking strategy
    if cfg.chunking.strategy == "fixed":
        return f"finance_fixed_{cfg.chunking.chunk_size}_{cfg.chunking.chunk_overlap}"
    else:
        return f"finance_semantic_{cfg.chunking.breakpoint_percentile_threshold}"

def ensure_ingestion(cfg, vector_store):
    if vector_store.client.collection_exists(vector_store.collection_name):
        log.info(f"Collection {vector_store.collection_name} exists. Skipping ingestion.")
        return

    log.info(f"Ingesting PDFs for {vector_store.collection_name}...")
    pdf_loader = PDFLoader()
    chunker = get_chunker(cfg)
    
    # Process all PDFs in data/raw
    pdf_files = glob.glob(os.path.join(cfg.data.raw_dir, "*.pdf"))
    if not pdf_files:
        log.warning(f"No PDF files found in {cfg.data.raw_dir}!")
        return

    all_nodes = []
    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        docs = pdf_loader.load(pdf_path)
        nodes = chunker.chunk(docs)
        all_nodes.extend(nodes)
    
    vector_store.index(all_nodes)
    log.info("Ingestion complete.")

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Update collection name in config based on chunking
    cfg.qdrant.collection_name = get_collection_name(cfg)
    
    # 2. Ingest
    vector_store = HybridQdrantClient(cfg)
    ensure_ingestion(cfg, vector_store)
    
    # 3. Load Test Set (FinanceBench)
    # Assuming we use the HF sub-sample or full dataset
    # For now, let's load a sample or expect a local file
    try:
        ds = load_dataset("patronus-ai/financebench", split="train[:10]") # Sample for speed
    except Exception as e:
        log.warning(f"Could not load FinanceBench from HF: {e}. Checking local.")
        # Fallback to dummy or local
        ds = [] 

    retriever = HybridRetriever(vector_store, cfg)
    engine = InferenceEngine(cfg)

    # 4. Run Loop
    run_id = f"{cfg.model.name}_{cfg.chunking.strategy}_{cfg.retrieval.strategy}"
    trec_file = os.path.join("outputs", f"{run_id}.trec")
    results_file = os.path.join("outputs", f"{run_id}_results.csv")
    
    os.makedirs("outputs", exist_ok=True)
    
    trec_lines = []
    final_results = []
    
    for i, row in enumerate(tqdm(ds)):
        query = row['question']
        query_id = str(i) # Use index or row['id']
        
        # Retrieve
        candidates = retriever.retrieve(query)
        
        # Log TREC
        for rank, doc in enumerate(candidates):
            # query_id Q0 doc_id rank score run_id
            trec_lines.append(f"{query_id} Q0 {doc['id']} {rank+1} {doc['score']} {run_id}")

        # Generate (pass top 5 candidates)
        best_candidates = candidates[:5]
        answer = engine.generate(query, best_candidates)
        
        final_results.append({
            "question": query,
            "ground_truth": row.get('answer', ''),
            "generated_answer": answer,
            "context": [str(c['text']) for c in best_candidates],
            "config": run_id
        })

    # Save outputs
    with open(trec_file, "w") as f:
        f.write("\n".join(trec_lines))
    
    pd.DataFrame(final_results).to_csv(results_file, index=False)
    log.info(f"Finished run {run_id}. Saved to {results_file}")

if __name__ == "__main__":
    main()
