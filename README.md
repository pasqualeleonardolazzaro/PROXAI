# Overview

PROXAI (PROvenance for eXplainable AI) is a library capable of capturing provenance in a preprocessing pipeline. PROXAI provides a clear and clean interface for capturing provenance, at various levels of granularity, without the need to invoke additional functions. The data scientist simply needs to implement the pipeline and, after executing it, can analyze the corresponding graph using Neo4j.

## Table of Contents

- [Installation](#installation)
- [Neo4j (Docker)](#neo4j-docker)
- [Pipeline Assets](#pipeline-assets)
  - [Pipelines](pipelines)
  - [Datasets](datasets)
- [LLM Prompts](LLM)
- [Provenance Exploitation](NaturalLanguageGraphExploitation)
- [Provenance Explanation](Why+Narratives)
- [Getting Started](#getting-started)


## Installation

To use the tool, follow these steps:

1. **Clone the repository**:
   ```
   git clone https://github.com/pasqualeleonardolazzaro/PROXAI.git
   ```

2. **Navigate to the project directory**:
   ```
   cd PROXAI
   ```

3. **Create and activate a virtual environment (optional but recommended)**:
   On Unix or MacOS:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

   On Windows:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

4. **Install the requirements**:
   ```
   pip install -r requirements.txt
   ```
## Groq
1. **Create an account and generate an API key on Groq**  
   Visit [Groq Console](https://console.groq.com/keys) to create your API key.

2. **Create a Python file named `KEY.py` in the PROXAI folder**  
   Inside the file, define a variable to store your API key as follows:
   ```python
   MY_KEY = "YOUR_NEWLY_CREATED_KEY"
   
## Neo4j (Docker)


- **Start Neo4j in the background by executing the following command:**:
    
      cd neo4j
      docker compose up -d

- **To stop Neo4j, run the following command:**:

      cd neo4j
      docker compose down -v

### Access the Neo4j Web Interface

To access the Neo4j web interface, open the following URL in your web browser:

http://localhost:7474/browser/

#### Default Credentials

- **User**: `neo4j`
- **Password**: `adminadmin`

## Getting Started

To get started with the tool, follow these simple steps.

### How to run it

The main file for running the tool is **prolit_run.py**. Below is a description of the arguments you can pass via the command line.

### Example commands

Run PROXAI by specifying the dataset and pipeline you want to use, along with other options:

```bash
python prolit_run.py --dataset datasets/car_data.csv --pipeline pipelines/car_pipeline.py --frac 0.1 --granularity_level 3
```
### Arguments

- `--dataset`: Relative path to the dataset file. For example: `datasets/car_data.csv`
- `--pipeline`: Relative path to the pipeline file. For example: `pipelines/car_pipeline.py`
- `--frac`: Fraction of the dataset to use for sampling. Value between `0.0` and `1.0`. For example, `0.1` uses 10% of the dataset.
- `--granularity_level`: Granularity level. Can be `1`, `2`, `3` or `4` with specific meanings for each level of detail.
  - `1`: Sketch Level
  - `2`: Derivation Level
  - `3`: Full Level
  - `4`: Only Columns Level

## How to Use the GUI (Streamlit)

1. Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

2. Launch the GUI with Streamlit:

```bash
streamlit run streamlit_app.py
```

3. Once running, your browser should open automatically. If not, you can manually navigate to:

```
http://localhost:8501
```

4. The GUI provides five main sections, accessible from the sidebar:

   - **Provenance Analysis** – Select a dataset and a pipeline, choose the granularity level (Sketch, Only Columns, Detailed, Full), and run PROXAI to generate the provenance graph. An advanced editor lets you inspect and modify `extracted_code.py` before execution.
   - **Graph Chat** – Ask questions in natural language about the provenance graph using a RAG-based chatbot connected to Neo4j.
   - **Provenance Explorer** – Open the Neo4j Browser directly to explore and query the provenance graph.
   - **Global Analysis** – Evaluate model behaviour at a global level using SHAP feature importance and influence functions (FirstOrder, TracIn). Results include the most influential training samples (proponents and opponents) with direct links to the provenance graph.
   - **Local Analysis** – Inspect individual predictions: view correct/incorrect classifications (or regression residuals), select a specific sample, and compute its local influence scores (FirstOrder or TracIn) to identify which training points most affected that prediction.
