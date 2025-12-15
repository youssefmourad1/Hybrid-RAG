# Setup Guide: Llama.cpp & API Keys

This guide will help you set up the local inference engine and obtain necessary API keys for the Hybrid-RAG system.

## 1. Installing Llama.cpp (Local Inference)

### Option A: Via Homebrew (Easiest for Mac)
If you have Homebrew installed:

```bash
brew install llama.cpp
```
*Note: This might not always have the latest "server" binary alias. If `llama-server` is not found, try Option B.*

### Option B: Compiling from Source (Recommended for Performance)
This ensures you get the latest features and Metal (GPU) acceleration on Mac.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    ```

2.  **Build with Metal (Apple Silicon) support**:
    ```bash
    make
    ```

3.  **Locate the Server Binary**:
    The binary will be in the `llama.cpp` folder, named `llama-server` (or just `server` in older versions).
    You can run it directly: `./llama-server ...`

## 2. Downloading GGUF Models

You need a quantized model file (.gguf) to run locally. We recommend **Llama-3-8B-Instruct**.

1.  **Install HuggingFace CLI**:
    ```bash
    pip install huggingface_hub
    ```

2.  **Download the Model**:
    ```bash
    huggingface-cli download Meta-Llama/Meta-Llama-3-8B-Instruct-GGUF --include "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" --local-dir ./models
    ```
    *This saves the model to the `models/` folder in your project.*

## 3. Running the Local Server

Once installed and downloaded, start the server in a separate terminal window:

```bash
# Verify path to your llama-server binary
./path/to/llama.cpp/llama-server \
    -m ./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    -c 2048 \
    --port 8080
```
*   `-c 2048`: Context window size.
*   `--port 8080`: The port our project expects.

**Verification**:
Open your browser to `http://127.0.0.1:8080`. You should see the Llama.cpp web interface.

## 4. Getting API Keys (For Ingestion & Cloud Fallback)

### A. LlamaCloud API Key (Required for PDF Parsing)
1.  Go to [LlamaCloud Console](https://cloud.llamaindex.ai/).
2.  Sign up or Log in (GitHub/Google).
3.  Navigate to **API Keys**.
4.  Generate a new key (e.g., `llx-...`).
5.  Add it to your `.env` file:
    ```ini
    LLAMA_CLOUD_API_KEY=llx-your-key-here
    ```

### B. OpenAI API Key (Optional - for GPT-4o)
1.  Go to [OpenAI Platform](https://platform.openai.com/api-keys).
2.  Log in.
3.  Click **Create new secret key**.
4.  Copy the key (e.g., `sk-...`).
5.  Add it to your `.env` file:
    ```ini
    OPENAI_API_KEY=sk-your-key-here
    ```
