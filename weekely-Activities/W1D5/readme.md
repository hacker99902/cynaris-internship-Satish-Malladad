# W1D5 – Data Visualisation & Local AI Assistant

## Overview

W1D5 focuses on exploring and understanding datasets through data visualisation using **Matplotlib, Seaborn, and Pandas**. It also introduces a local AI assistant powered by **Ollama**, which answers questions about the loaded dataset.

The goal is to convert raw data into meaningful insights and interact with the dataset using a locally running Large Language Model (LLM).

## Objectives

- Load and explore datasets using Pandas.
- Visualise data distributions using Matplotlib and Seaborn.
- Identify relationships and correlations between numerical features.
- Generate insights from the dataset.
- Build a local AI assistant using Ollama.
- Ask questions about the dataset through the command line.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |
| Requests | Communicating with the Ollama API |
| Ollama | Running the local LLM |
| Llama 3.2 (3B) | Answering dataset-related questions |

## Project Structure

```text
W1D5/
│
├── data/
│   └── states.csv
│
├── src/
│   ├── visualize_data.py
│   └── project_task_day5.py
│
├── docs/
│   └── week1_output.png
│
├── results/
│   └── week1_results.csv
│
├── README.md
└── requirements.txt
```

## Features

### 1. Data Visualisation
- Histograms for understanding numerical distributions.
- Correlation heatmaps for identifying relationships between numerical columns.
- Count plots for categorical data.
- Box plots for identifying outliers and understanding data spread.

### 2. Local AI Assistant
- Loads a CSV dataset using Pandas.
- Generates a dataset summary and sample records.
- Sends dataset context to the local Ollama API.
- Accepts questions through the command line.
- Generates responses using the Llama 3.2 model.

## Dataset

The project uses `states.csv`, containing **21,675 rows and 7 columns**.

The dataset is loaded and processed using Pandas. Its summary and sample records are provided as context to the local AI assistant.

## Setup and Installation

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and run Ollama

Download Ollama from https://ollama.com/

Pull the required model:

```bash
ollama pull llama3.2:3b
```

Make sure the Ollama server is running.

### 4. Run the visualisation script

```bash
python src/visualize_data.py
```

### 5. Run the local AI assistant

```bash
python src/project_task_day5.py
```

Ask at least three questions about the dataset through the command-line interface.

## Expected Output

- Visualisations showing dataset distributions and relationships.
- Dataset summary and sample records.
- Responses from the local AI assistant.
- Saved output evidence in the `docs/` or `results/` directory.

## Learning Outcomes

- Understood the importance of data visualisation in Exploratory Data Analysis (EDA).
- Practised using Matplotlib and Seaborn.
- Learned how to load and inspect datasets using Pandas.
- Integrated a locally running LLM using Ollama.
- Learned how to provide dataset context to an AI model for question answering.

## Future Improvements

- Add more advanced visualisations.
- Improve the AI assistant's ability to answer questions about specific columns.
- Add interactive visualisations.
- Integrate more datasets and support larger data files.

---

**Week 1 – Day 5 | AI/ML Internship**

*Learning data visualisation and building a local AI-powered dataset assistant.*