# Setup Guide: Llama.cpp & API Keys

This guide will help you set up the local inference engine and obtain necessary API keys for the Hybrid-RAG system.

## 1. Installing Llama.cpp (Local Inference)

### Option A: Via Homebrew (Mac Only)
If you have Homebrew installed:

```bash
brew install llama.cpp
```
*Note: This might not always have the latest "server" binary alias. If `llama-server` is not found, try Option B.*

### Option B: Pre-built Binaries (Easiest for Windows)
1. Visit the [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases)
2. Download the latest Windows build (e.g., `llama-*-bin-win-*.zip`)
3. Extract to a folder like `C:\llama.cpp`
4. The `llama-server.exe` will be in the `bin` folder

### Option C: Compiling from Source (Recommended for Performance)

#### For Mac:
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    ```

2.  **Build with Metal (Apple Silicon) support**:
    ```bash
    make
    ```

#### For Windows:
1.  **Prerequisites**: Install [CMake](https://cmake.org/download/) and [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)

2.  **Clone the repository**:
    ```cmd
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    ```

3.  **Build with CMake**:
    ```cmd
    cmake -B build
    cmake --build build --config Release
    ```

4.  **Locate the Server Binary**:
    The binary will be in `build\bin\Release\llama-server.exe`

## 2. Downloading GGUF Models

You need a quantized model file (.gguf) to run locally. We recommend **Llama-3-8B-Instruct**.

1.  **Install HuggingFace CLI**:
    ```bash
    pip install huggingface_hub
    ```

2.  **Download the Model**:
    
    **Mac/Linux**:
    ```bash
    huggingface-cli download Meta-Llama/Meta-Llama-3-8B-Instruct-GGUF --include "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" --local-dir ./models
    ```
    
    **Windows**:
    ```cmd
    huggingface-cli download Meta-Llama/Meta-Llama-3-8B-Instruct-GGUF --include "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf" --local-dir .\models
    ```
    
    *This saves the model to the `models/` folder in your project.*

## 3. Running the Local Server

Once installed and downloaded, start the server in a separate terminal window:

**Mac/Linux**:
```bash
# Verify path to your llama-server binary
./path/to/llama.cpp/llama-server \
    -m ./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
    -c 2048 \
    --port 8080
```

**Windows**:
```cmd
# Example with pre-built binary
C:\llama.cpp\bin\llama-server.exe -m .\models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -c 2048 --port 8080

# Or if compiled from source
.\build\bin\Release\llama-server.exe -m .\models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -c 2048 --port 8080
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