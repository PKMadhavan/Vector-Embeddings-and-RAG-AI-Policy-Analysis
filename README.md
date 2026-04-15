# Vector Embeddings and RAG – AI Policy Analysis

This project implements a multi-document Retrieval-Augmented Generation (RAG) system over AI policy and regulatory documents using LangChain, Chroma, and HuggingFace sentence-transformer embeddings. It evaluates how different embedding models affect retrieval quality and answer correctness on policy-focused questions.

## Features

- Loads multiple AI policy PDFs and text files from remote URLs  
- Splits documents into overlapping chunks with `RecursiveCharacterTextSplitter`  
- Builds Chroma vector stores for different embedding models  
- Runs a reusable RAG pipeline with a retriever and an LLM (ChatOpenAI)  
- Evaluates retrieval and answer quality on a 1–3 scale across several query types  
- Prints summary tables comparing models by average retrieval and answer scores

## Tech Stack

- Python  
- LangChain (documents, text splitting, chains)  
- Chroma (vector database)  
- HuggingFaceEmbeddings (e.g., `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `multi-qa-MiniLM-L6-cos-v1`)  
- OpenAI ChatCompletion models (for generation)

## Experiment Design

The system uses a small set of hand-designed queries that cover:

- Single-document factual lookup  
- Multi-document search and combine  
- Thematic synthesis across documents  
- Comparative questions (e.g., US vs EU approaches)  
- Specific definition/scope questions (e.g., “high-risk AI”)

For each embedding model, the notebook:

1. Builds a fresh Chroma store from the same processed chunks  
2. Runs the RAG pipeline on all queries  
3. Manually scores retrieval and answer quality (1–3 scale)  
4. Computes average scores per model and prints a compact results table

## How to Run

1. Install dependencies (in Colab or locally):

```bash
pip install -qU requests chromadb langchain langchain-chroma langchain-huggingface langchain-openai langchain-community sentence-transformers tiktoken openai pypdf
```

2. Open the notebook:

- `vector_embeddings_starter.ipynb`

3. Set your API keys in the `main()` function:

```python
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HF_TOKEN"
```

4. Run the notebook cells in order:

- Install libraries  
- Define core functions (loading, preprocessing, vector store, RAG chain, evaluation)  
- Run `main()` to execute the full experiment  
- Run the final “Print Results” cell to see the summary table

## Key Parameters

- Chunk size: typically around 1000 characters  
- Chunk overlap: around 200 characters  
- Retriever `k`: 4 documents per query  
- Embedding models: configurable via the `embeddingmodels` list in `main()`

These parameters can be tuned to explore trade-offs between retrieval granularity, context preservation, and overall answer quality.

## Outputs

- Console logs showing per-query retrieval and answer scores for each model  
- Final summary table with average retrieval and answer scores per embedding model  
- Qualitative insights about when retrieval or generation becomes the bottleneck in a policy-heavy RAG setup
