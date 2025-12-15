# Hybrid-RAG: Modular Financial Analysis System

A production-ready, modular RAG (Retrieval-Augmented Generation) system designed for high-precision financial benchmarking. This project features a hot-swappable architecture for models, chunking strategies, and retrieval methods, backed by **Hydra** for configuration, **Qdrant** for vector storage, and **LlamaParse** for SOTA PDF ingestion.

## 🚀 Key Features

*   **Modular Architecture**: Swap Retrieval (Dense/Hybrid), Chunking (Fixed/Semantic), and Models (Local/Cloud) via simple config changes.
*   **Hybrid Search**: Combines **Dense Vectors** (BGE-M3) with **Sparse Vectors** (SPLADE) using Reciprocal Rank Fusion (RRF).
*   **Production ETL**: Uses **LlamaParse** for layout-aware PDF extraction (tables, charts).
*   **Evaluations**: Built-in pipelines for **Ranx** (NDCG@10) and **Ragas** (Faithfulness/Relevancy).
*   **Local & Cloud Inference**: Unified interface for `llama.cpp` (local) and OpenAI (cloud).

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   Docker & Docker Compose
*   [Optional] `uv` or `poetry` (standard `pip` works too)
*   🔴 **REQUIRED**: See [SETUP.md](SETUP.md) for Llama.cpp installation and API keys.

### Setup

1.  **Clone and Install Dependencies**:
    ```bash
    git clone https://github.com/youssefmourad1/Hybrid-RAG.git
    cd Hybrid-RAG
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    Create a `.env` file in the root directory:
    ```ini
    OPENAI_API_KEY=sk-...
    LLAMA_CLOUD_API_KEY=llx-...
    HF_TOKEN=hf_...
    ```

3.  **Start Vector Database**:
    ```bash
    docker-compose up -d
    ```
    This spins up Qdrant on `localhost:6333`.

## 🏃 Running Experiments

The project uses **Hydra** for configuration management. You can override any parameter from the command line.

### Quick Start (Local Verification)
To verify your setup without downloading massive models:
```bash
python run_local.py
```

### Full Benchmark Execution
Run the full pipeline (Ingestion -> Retrieval -> Generation -> Logging):

**Default (Hybrid Search + BGE-M3 + Llama-3-8B):**
```bash
python scripts/benchmark.py
```

**Custom Configuration:**
```bash
# Example: Use Dense-only search, Semantic Chunking, and GPT-4o
python scripts/benchmark.py \
    retrieval=dense \
    chunking=semantic \
    model=gpt4o
```

**Running with Smaller Models (Faster/Laptop-friendly):**
```bash
python scripts/benchmark.py \
    retrieval.dense_model="BAAI/bge-small-en-v1.5" \
    retrieval.reranker="cross-encoder/ms-marco-MiniLM-L-6-v2"
```

## 📊 Configuration Guide

Configurations are located in `conf/`.

| Category | Config File | Options |
| :--- | :--- | :--- |
| **Retrieval** | `conf/retrieval/` | `hybrid` (Dense+Sparse+RRF), `dense` (Dense only) |
| **Chunking** | `conf/chunking/` | `fixed` (512/50), `semantic` (Embedding-based breakpoints) |
| **Model** | `conf/model/` | `llama3_8b` (Local), `gpt4o` (Cloud) |

### Adding a New Configuration
1.  Create a new yaml file in the respective folder (e.g., `conf/model/mistral.yaml`).
2.  Run with `model=mistral`.

## 📈 Evaluation

After generating results, run the grading script to calculate metrics (NDCG, Faithfulness, etc.):

```bash
python scripts/grade.py
```
This generates a `outputs/report.md` and a trade-off plot.

## 📁 Directory Structure

```
├── conf/               # Hydra configurations
├── data/               # Raw PDFs and processed data
├── src/
│   ├── ingestion.py    # LlamaParse & Chunking logic
│   ├── vector_store.py # Qdrant wrapper (Dense+Sparse)
│   ├── retrieval.py    # RRF & Reranking logic
│   └── generation.py   # LLM Client (Jinja2 prompts)
├── scripts/
│   ├── benchmark.py    # Main execution loop
│   └── grade.py        # Scoring script (Ragas/Ranx)
├── run_local.py        # Integration test (In-Memory)
└── docker-compose.yaml # Qdrant service
```

## 📚 References & Citations

If you use this codebase in your research or production system, please check the licenses of the underlying models and libraries.

### Models
*   **BGE-M3 (Dense & Sparse)**:
    > Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff. "BGE-M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)
*   **SPLADE (Sparse)**:
    > Thibault Formal, Benjamin Piwowarski, Stéphane Clinchant. "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking." [arXiv:2107.05720](https://arxiv.org/abs/2107.05720)
*   **Llama-3**:
    > Meta AI. "Llama 3 Model Card." [AI at Meta](https://ai.meta.com/blog/meta-llama-3/)

### Evaluation Frameworks
*   **Ragas (LLM-as-a-Judge)**:
    > Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
*   **Ranx (Information Retrieval Evaluation)**:
    > Elias Bassani. "ranx: A Blazing-Fast Python Library for Ranking Evaluation and Comparison." [ECIR 2022](https://github.com/AmenRa/ranx)

### Dataset
*   **FinanceBench**:
    > Pranabesh Das, et al. "FinanceBench: A New Benchmark for Financial Question Answering." [arXiv:2311.11944](https://arxiv.org/abs/2311.11944)

### Core Technologies
*   **Qdrant**: High-performance vector similarity search engine. [Website](https://qdrant.tech/)
*   **LlamaIndex**: Data framework for LLM applications. [Website](https://www.llamaindex.ai/)
*   **Hydra**: Configuration management by Meta Research. [Website](https://hydra.cc/)
