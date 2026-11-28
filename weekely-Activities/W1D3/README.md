# W1D3 – Data Loading, Cleaning & Inspection

## Project Overview

This project demonstrates a data processing pipeline using Python, Pandas, NumPy, LangChain, Ollama and ChromaDB.

The project uses COVID-19 state-wise data to demonstrate data loading, cleaning, inspection, embedding generation and semantic question answering.

## Objectives

- Load CSV datasets using Pandas.
- Clean missing values and inconsistent text.
- Inspect dataset structure and statistical information.
- Validate dataset quality.
- Convert state-wise records into LangChain documents.
- Generate embeddings using Ollama.
- Store embeddings in ChromaDB.
- Perform semantic question answering using a local LLM.

## Technology Stack

- Python
- Pandas
- NumPy
- LangChain
- Ollama
- ChromaDB
- Jupyter Notebook

## Project Structure

```text
w1d3-data-pipeline/
├── data/
│   ├── raw/
│   │   └── states.csv
│   └── cleaned/
│       └── states_cleaned.csv
├── notebooks/
│   └── 01_data_loading_cleaning.ipynb
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── clean_data.py
│   ├── inspect_data.py
│   ├── validate_data.py
│   ├── embed_index.py
│   └── query_demo.py
├── tests/
│   └── test_query_demo.py
├── main.py
├── requirements.txt
└── README.md
```

## Data Flow Architecture

The project follows a pipeline that transforms raw COVID-19 data into cleaned records, vector embeddings and AI-generated answers.

```text
             RAW COVID-19 CSV
                     |
                     v
              load_data.py
                     |
                     v
              clean_data.py
                     |
                     v
             validate_data.py
                     |
                     v
             inspect_data.py
                     |
                     v
             CLEANED CSV FILE
                     |
                     v
              embed_index.py
                     |
              +------+------+
              |             |
              v             v
        LangChain       Ollama
        Documents    Embeddings
              |             |
              +------+------+
                     |
                     v
                 ChromaDB
                     |
                     v
                USER QUERY
                     |
                     v
              query_demo.py
                     |
             +-------+-------+
             |               |
             v               v
        Numerical         Semantic
         Queries          Queries
             |               |
             v               v
           Pandas        ChromaDB
                             |
                             v
                      Retrieved Context
                             |
                             v
                        Ollama LLM
                         llama3.2
                             |
                             v
                       AI RESPONSE
```

### Input
- Raw COVID-19 state-wise CSV dataset.
- User questions about the dataset.

### Processing
- Pandas performs data loading, cleaning, inspection and validation.
- LangChain converts the latest state records into documents.
- Ollama generates embeddings using `nomic-embed-text`.
- ChromaDB stores embeddings and retrieves relevant documents.

### Output
- Cleaned CSV dataset.
- Persistent vector database.
- Numerical answers using Pandas.
- Context-based AI answers using Ollama.

## Local LLM Inference with Ollama

This project uses Ollama to run language models locally instead of depending on a hosted LLM API.

Two models are used:

| Model | Purpose |
|---|---|
| `nomic-embed-text` | Generates vector embeddings |
| `llama3.2:3b` | Generates natural-language answers |

### How Local Inference Works

1. The user submits a question.
2. ChromaDB retrieves relevant documents using semantic similarity search.
3. The retrieved documents are passed as context to the language model.
4. Ollama runs `llama3.2:3b` locally.
5. The model generates an answer based on the retrieved context.

### Prompt Instructions

The application uses LangChain's `ChatPromptTemplate` to provide instructions and context to the LLM.

The current prompt instructs the model to:

- Answer using only the supplied context.
- Clearly indicate when the answer is unavailable.
- Use the user's question and retrieved records to generate a response.

These instructions help reduce unsupported answers.

### Conversational Context

The current implementation passes the retrieved documents and the current question to the model.

It does not implement persistent conversation memory or chat history. Each query is processed independently.

Retrieved documents act as temporary context for generating the answer.

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama and download the required models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Run the Pipeline

Run the data loading, cleaning, inspection and validation pipeline:

```bash
python main.py
```

## Generate Embeddings

```bash
python src/embed_index.py
```

This process converts the latest available record for each state into a LangChain Document and stores its embedding in ChromaDB.

The index excludes national aggregate and unassigned records.

## Query the Dataset

```bash
python src/query_demo.py
```

Example questions:

- Which state has the highest confirmed cases?
- Which state has the lowest confirmed cases?
- What is the average confirmed cases?
- What is the latest available record for Kerala?

Numerical questions are calculated using Pandas. General questions use ChromaDB retrieval and Ollama.

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Data Cleaning Operations

- Remove completely empty rows and columns.
- Remove extra whitespace from text columns.
- Convert dates into datetime format.
- Handle missing numeric values using column means.
- Handle missing text values using the most frequent value.

**Limitation:** Mean imputation is a simple demonstration technique. It may not be appropriate for cumulative time-series values such as COVID-19 testing counts.

## Embedding Strategy

The current implementation creates one document per state using its latest available record. It does not embed every historical row or perform text chunking.

Ollama's `nomic-embed-text` model generates embeddings, which are stored in ChromaDB.


## Project Outcome

The project demonstrates a complete introductory workflow combining traditional data processing with local embedding generation, vector storage and retrieval-augmented question answering.
