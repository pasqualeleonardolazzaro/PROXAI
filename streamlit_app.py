# streamlit_app.py
import os
import sys
import shutil
import traceback
import subprocess
import platform
from pathlib import Path
import streamlit as st
import json
import pandas as pd
import urllib.parse
import uuid
import numpy as np
import pickle
import tempfile


# ====================== CONFIG BASE ======================
st.set_page_config(page_title="PROXAI Console", layout="wide")

BASE_DIR = Path(__file__).parent.resolve()
os.chdir(BASE_DIR)

EXTRACTED = BASE_DIR / "extracted_code.py"
EXTRACTED_BAK = BASE_DIR / "extracted_code_llm.py"  # backup del codice generato dall'LLM

ENV_INFLUENCIAE_DIR = BASE_DIR / ".influenciae_env"

# Stato persistente Streamlit
ss = st.session_state
ss.setdefault("last_rc", None)
ss.setdefault("stdout", "")
ss.setdefault("stderr", "")
ss.setdefault("use_manual", True)     # di default usa il codice manuale se presente
ss.setdefault("reload_nonce", 0)      # cambia per forzare reload editor

# ====================== NAVIGAZIONE ======================
st.sidebar.title("PROXAI")
page = st.sidebar.radio("Navigazione", ["Provenance Analysis", "Graph Chat", "Provenance Explorer","Global Analysis","Local Analysis"], index=0)
#st.sidebar.caption(f"Working dir: {BASE_DIR}")

# ====================== HELPERS COMUNI ======================
def ensure_backup():
    if EXTRACTED.exists() and not EXTRACTED_BAK.exists():
        shutil.copy2(EXTRACTED, EXTRACTED_BAK)

def save_user_code(text: str):
    ensure_backup()
    EXTRACTED.write_text(text, encoding="utf-8")

def run_prolit(dataset: str, pipeline: str, frac: str, gran_level: int, use_manual: bool):
    """Esegue prolit_run.py come da CLI (get_args() lato script gestisce gli argomenti)."""
    cmd = [
        sys.executable, "prolit_run.py",
        "--dataset", dataset,
        "--pipeline", pipeline,
        "--frac", str(frac),
        "--granularity_level", str(gran_level),
    ]
    if use_manual:
        cmd.append("--use_manual_code")  # richiede la patch in get_args() di prolit_run.py

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    ss.last_rc = proc.returncode
    ss.stdout = proc.stdout or ""
    ss.stderr = proc.stderr or ""
    return cmd, proc.returncode

def run_shap(model: str, pipeline: str, dataset: str):
    """Esegue run_shap.py come da CLI (get_args() lato script gestisce gli argomenti)."""
    cmd = [
        sys.executable, "run_shap2.py",
        "--model", model,
        "--pipeline", pipeline,
        "--dataset", dataset,
    ] 

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    ss.last_rc = proc.returncode
    ss.stdout = proc.stdout or ""
    ss.stderr = proc.stderr or ""
    return cmd, proc.returncode

def run_global_analysis(dataset_path, pipeline_path, model_path):
    """
    Calls the subprocess script to calculate metrics.
    Returns the loaded results dictionary.
    """
    # Create a temp file to store results
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        output_file = tmp.name

    cmd = [
        sys.executable, # Use the same python env
        "backend_metrics.py",
        "--model", str(model_path),
        "--pipeline", str(pipeline_path),
        "--dataset", str(dataset_path),
        "--output_file", output_file
    ]

    try:
        with st.spinner("Running preprocessing and model inference..."):
            process = subprocess.run(cmd, capture_output=True, text=True)
            
        if process.returncode != 0:
            st.error("Error in analysis subprocess:")
            st.code(process.stderr)
            return None

        # Load results back
        with open(output_file, "rb") as f:
            data = pickle.load(f)
        
        return data

    finally:
        # Cleanup temp file
        if os.path.exists(output_file):
            os.remove(output_file)

def get_influenciae_python():
    if platform.system() == "Windows":
        return ENV_INFLUENCIAE_DIR / "Scripts" / "python.exe"
    return ENV_INFLUENCIAE_DIR / "bin" / "python"

# Ensure env exists
def ensure_influenciae_env():
    if not ENV_INFLUENCIAE_DIR.exists():
        from influenciae.create_influenciae_env import create_env
        create_env()

def run_FirstOrder(model_path, dataset_path, pipeline_path, sample_to_predict):
    """
    Calls the FirstOrder script inside the influenciae env.
    sample_to_predict can be a Pandas row (Series) or dict-like.
    """
    ensure_influenciae_env()

    python_path = get_influenciae_python()

    script_path = BASE_DIR / "influenciae" / "run_firstOrder.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Influence script not found at {script_path}")

    # Convert sample row to JSON-friendly dict
    if hasattr(sample_to_predict, "to_dict"):
        sample_dict = sample_to_predict.to_dict()
    else:
        sample_dict = sample_to_predict

    # Create payload
    data = {
        "model_path": str(model_path),
        "dataset": str(dataset_path),
        "pipeline_path": str(pipeline_path),
        "sample_to_predict": sample_dict,
    }

    # Call subprocess
    result = subprocess.run(
        [str(python_path), str(script_path)],
        input=json.dumps(data),
        text=True,
        capture_output=True
    )

    # Error handling 
    if result.returncode != 0:
        raise RuntimeError(
            f"Error running influenciae script:\n"
            f"STDOUT: {result.stdout}\n\n"
            f"STDERR: {result.stderr}"
        )

    return json.loads(result.stdout)


def run_TracIn(model_path, dataset_path, pipeline_path, sample_to_predict):
    """
    Calls the FirstOrder script inside the influenciae env.
    sample_to_predict can be a Pandas row (Series) or dict-like.
    """
    ensure_influenciae_env()

    python_path = get_influenciae_python()

    script_path = BASE_DIR / "influenciae" / "run_TracIn.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Influence script not found at {script_path}")

    # Convert sample row to JSON-friendly dict
    if hasattr(sample_to_predict, "to_dict"):
        sample_dict = sample_to_predict.to_dict()
    else:
        sample_dict = sample_to_predict

    # Create payload
    data = {
        "model_path": str(model_path),
        "dataset": str(dataset_path),
        "pipeline_path": str(pipeline_path),
        "sample_to_predict": sample_dict,
    }

    # Call subprocess
    result = subprocess.run(
        [str(python_path), str(script_path)],
        input=json.dumps(data),
        text=True,
        capture_output=True
    )

    # Error handling 
    if result.returncode != 0:
        raise RuntimeError(
            f"Error running influenciae script:\n"
            f"STDOUT: {result.stdout}\n\n"
            f"STDERR: {result.stderr}"
        )

    return json.loads(result.stdout)



def run_global_influence(script_filename, model_path, dataset_path, pipeline_path):
    """
    Dynamically calls any influence script inside the influenciae env.
    """
    ensure_influenciae_env()
    python_path = get_influenciae_python()
    
    script_path = BASE_DIR / "influenciae" / script_filename

    if not script_path.exists():
        raise FileNotFoundError(f"Influence script not found at {script_path}")

    # Create payload
    data = {
        "model_path": str(model_path),
        "dataset": str(dataset_path),
        "pipeline_path": str(pipeline_path),
    }

    # Call subprocess
    result = subprocess.run([str(python_path), str(script_path)],
        input=json.dumps(data),
        text=True,
        capture_output=True
    )

    # Error handling 
    if result.returncode != 0:
        raise RuntimeError(
            f"Error running {script_filename}:\n"
            f"STDOUT: {result.stdout}\n\n"
            f"STDERR: {result.stderr}"
        )

    return json.loads(result.stdout)
    
# Helper to format the raw JSON list into a clean DataFrame
def format_influence_df(data_list):
    """
    Generates links based on features, flattens the JSON, and formats the table.
    """
    if not data_list:
        return pd.DataFrame()
    
    # generate the Link for each item before flattening
    processed_list = []
    for item in data_list:
        new_item = item.copy()
        
        # Generate URL
        features = item.get('features', {})
        new_item['Graph Link'] = create_fingerprint_url(features)
        
        processed_list.append(new_item)
    
    # Flatten nested dictionaries
    df = pd.json_normalize(processed_list)
    
    # Clean column names
    df.columns = [col.replace("features.", "") for col in df.columns]
    
    # Reorder Columns 
    priority_cols = ['training_index', 'Graph Link', 'mean_influence_score', 'label']
    
   
    final_cols = [c for c in priority_cols if c in df.columns] + \
                 [c for c in df.columns if c not in priority_cols]
    
    return df[final_cols]

#query for neo4j most influent point retrival
def create_fingerprint_url(row_features):
    """
    Creates a URL that finds a specific row in Neo4j based on the
    intersection of all its feature names and values.
    """
    base_url = "http://localhost:7474/browser/"
    
    # Create a list of conditions for the WHERE clause
    conditions = []
    
    for feature, value in row_features.items():
        # json.dumps ensures strings have quotes ("val") and numbers don't (10)
        val_str = json.dumps(value) 
        conditions.append(f"(n.feature_name = '{feature}' AND n.value = {val_str})")
    
    if not conditions:
        return base_url # Return empty if row has no features

    where_clause = " OR ".join(conditions)

    # Construct the Cypher query
    # Logic: 
    #   - Find ALL nodes that match ANY feature-value pair.
    #   - Group them by their 'index' (n.index).
    #   - Count how many matches exist for that index (match_count).
    #   - Order by the count descending (biggest match first).
    #   - Take the top 1 result.
    query = (
        f"MATCH (n:Entity) "
        f"WHERE {where_clause} "
        f"WITH n.index as idx, collect(n) as row_nodes, count(n) as match_count "
        f"ORDER BY match_count DESC "
        f"LIMIT 1 "
        f"RETURN row_nodes"
    )
    
    # URL Encode
    encoded_query = urllib.parse.quote(query)
    return f"{base_url}?cmd=edit&arg={encoded_query}"
    
# Initialize the pipeline once using Streamlit's caching
@st.cache_resource
def initialize_rag():
    """
    Initialize RAG pipeline once and cache it.
    This runs only once per Streamlit server session.
    """
    from rag_system.rag_pipeline2 import get_pipeline
    from rag_system.llm import generator
    from rag_system.indexing import graph_indexer
    return get_pipeline()


def ask_rag(query, session_id="default_session"):
    """
    Query the RAG system.
    """
    pipeline = initialize_rag()
    answer, context = pipeline.query(query, session_id)
    return answer#, context #debugging


def build_rag_index():
    """
    Rebuild the FAISS indexes from the graph.
    This is called from the UI when needed.
    """
    from rag_system.indexing import graph_indexer
    graph_indexer.build_indexes()
    # Clear the cached pipeline so it reloads with new indexes
    st.cache_resource.clear()

# ====================== PAGINA: RUN PROLIT ======================
if page == "Provenance Analysis":
    st.title("Provenance Analysis")

    # ---- Scansione cartelle per menu a tendina ----
    datasets_dir = BASE_DIR / "datasets"
    pipelines_dir = BASE_DIR / "pipelines"

    dataset_options = sorted(
        [str(p.relative_to(BASE_DIR)).replace("\\", "/") for p in datasets_dir.glob("*.csv")]
    ) if datasets_dir.exists() else []
    pipeline_options = sorted(
        [str(p.relative_to(BASE_DIR)).replace("\\", "/") for p in pipelines_dir.glob("*.py")]
    ) if pipelines_dir.exists() else []

    # fallback se vuoti
    if not dataset_options:
        dataset_options = ["datasets/generated_dataset.csv"]
    if not pipeline_options:
        pipeline_options = ["pipelines/orders_pipeline.py"]

    # ---- Granularity con etichette umane ----
    granularity_labels = ["Sketch", "Only columns", "Detailed", "Full"]
    #granularity_map = {"Sketch": 0, "Only columns": 1, "Detailed": 2, "Full": 3}
    granularity_map = {"Sketch": 1, "Only columns": 4, "Detailed": 2, "Full": 3} #giusto per provare CORREGGERE
    default_gran_label = "Only columns"

    colA, colB = st.columns(2)
    with colA:
        dataset = st.selectbox(
            "Dataset",
            dataset_options,
            index=dataset_options.index("datasets/generated_dataset.csv")
            if "datasets/generated_dataset.csv" in dataset_options else 0,
        )
        frac = st.text_input("Frac", "1")
    with colB:
        pipeline = st.selectbox(
            "Pipeline",
            pipeline_options,
            index=pipeline_options.index("pipelines/orders_pipeline.py")
            if "pipelines/orders_pipeline.py" in pipeline_options else 0,
        )
        granularity_label = st.selectbox(
            "Granularity level",
            granularity_labels,
            index=granularity_labels.index(default_gran_label),
        )
        granularity = granularity_map[granularity_label]

    st.divider()
    left, right = st.columns([2, 1])
    with left:
        st.checkbox(
            "Use manual code  (extracted_code.py)",
            value=ss.use_manual,
            key="use_manual",
            help="Se attivo, passa --use_manual_code a prolit_run.py (gestito da get_args()).",
        )
    with right:
        if EXTRACTED_BAK.exists():
            if st.button("Restore LLM code "):
                shutil.copy2(EXTRACTED_BAK, EXTRACTED)
                st.success("Restored extracted_code.py from backup LLM.")
                ss.reload_nonce += 1
                st.rerun()

    # ---- Editor nascosto finché non lo espandi ----
    with st.expander("advanced editor: `extracted_code.py` (click to expand)", expanded=False):
        # ricarica SEMPRE il contenuto da file a ogni rerun
        if EXTRACTED.exists():
            try:
                mtime_ns = EXTRACTED.stat().st_mtime_ns
            except Exception:
                mtime_ns = 0
            editor_key = f"editor_{mtime_ns}_{ss.reload_nonce}"
            try:
                file_text = EXTRACTED.read_text(encoding="utf-8")
            except Exception:
                file_text = ""
        else:
            mtime_ns = 0
            editor_key = f"editor_{mtime_ns}_{ss.reload_nonce}"
            file_text = ""

        edited = st.text_area(
            "Contenuto file",
            value=file_text,
            height=350,
            key=editor_key,  # la chiave cambia quando cambia mtime o nonce -> forzato reload
        )

        ecol1, ecol2, ecol3 = st.columns([1,1,2])
        if ecol1.button("💾 Salva"):
            save_user_code(edited)
            st.success("Saved `extracted_code.py`.")
            ss.reload_nonce += 1
            st.rerun()

        if ecol2.button("↻ Reload from file"):
            ss.reload_nonce += 1
            st.rerun()

        ecol3.caption(f"Ultima modifica: {mtime_ns}")

    # ---- Esecuzione ----
    run_clicked = st.button("▶️ Run Provenance Analysis", type="primary")

    if run_clicked:
        # forza anche il reload editor al prossimo rerun
        ss.reload_nonce += 1

        # se usi manuale, salva PRIMA di eseguire (se hai l'editor aperto e modifiche non salvate)
        # qui non possiamo leggere il valore del text_area senza chiave fissa;
        # quindi confidiamo che l'utente abbia premuto "Salva".
        # (scelta intenzionale: vogliamo reload da file a ogni run)
        with st.status("Running analysis…", expanded=True) as status:
            cmd, rc = run_prolit(dataset, pipeline, frac, granularity, ss.use_manual)
            st.code(" ".join(cmd), language="bash")

            # Fallback automatico se non hai ancora patchato get_args() con --use_manual_code
            if rc != 0 and ss.use_manual and ("unrecognized arguments" in ss.stderr.lower()):
                status.update(
                    label="⚠️ `prolit_run.py` non supporta --use_manual_code (patch get_args() mancante). Rilancio senza flag…",
                    state="running",
                )
                cmd2 = [
                    sys.executable, "prolit_run.py",
                    "--dataset", dataset,
                    "--pipeline", pipeline,
                    "--frac", str(frac),
                    "--granularity_level", str(granularity),
                ]
                proc2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=BASE_DIR)
                ss.last_rc = proc2.returncode
                ss.stdout = proc2.stdout or ""
                ss.stderr = proc2.stderr or ""
                st.info("Rilanciato senza flag; verifica che `prolit_run.py` importi/usi `extracted_code.py` manuale.")

            st.subheader("STDOUT")
            st.code(ss.stdout or "(vuoto)")
            st.subheader("STDERR")
            st.code(ss.stderr or "(vuoto)")

            if ss.last_rc == 0:
                status.update(label="✅ Execution succeeded", state="complete")
            else:
                status.update(label=f"❌ Exit code {ss.last_rc}", state="error")
                st.error("If necessary, expand the editor, save changes to file and rerun")


# ====================== PAGINA: GRAPH CHAT ======================
elif page == "Graph Chat":
    st.title("Graph Chat (Neo4j)")

    # --- Memory Initialization ---

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    # We create a list in session_state to hold the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Build Index Button ---
    with st.expander("⚙️ Graph loader", expanded=False):
        st.write("Build index for the first use")
        if st.button("Load graph if it's the first use"):
            with st.spinner("Building index... (might take a while)"):
                build_rag_index()
                st.success("success!")

    # --- Display Chat History ---
    # Iterate through the history and display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input ---
    if prompt := st.chat_input("Ask RAG..."):
        
        # Add User message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        #  Generate Response
        with st.chat_message("assistant"):
            with st.spinner("analyzing the graph..."):
                # Call the subprocess function
                response_text = ask_rag(prompt, st.session_state.session_id)
                
                st.markdown(response_text)

        # Add Assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# ====================== PAGINA: PROVENANCE EXPLORER ======================
elif page == "Provenance Explorer":
    st.title("Provenance Explorer")

    st.write("Open Neo4j Browser:")
    # link cliccabile
    st.markdown("[🌐 Neo4j Browser](http://localhost:7474/browser/)")

    # pulsante che apre in nuova scheda
    if st.button("Open Neo4j Browser"):
        st.components.v1.html(
            "<script>window.open('http://localhost:7474/browser/', '_blank');</script>",
            height=0,
            width=0,
        )

# ====================== PAGINA: GLOBAL ANALYSIS ======================

if page == "Global Analysis":
    st.title("Global analysis")

    # ---- Scansione cartelle per menu a tendina ----
    datasets_dir = BASE_DIR / "datasets"
    pipelines_dir = BASE_DIR / "pipelines"
    models_dir=BASE_DIR / "trained_models"

    dataset_options = sorted(
        [str(p.relative_to(BASE_DIR)).replace("\\", "/") for p in datasets_dir.glob("*.csv")]
    ) if datasets_dir.exists() else []
    pipeline_options = sorted(
        [str(p.relative_to(BASE_DIR)).replace("\\", "/") for p in pipelines_dir.glob("*.py")]
    ) if pipelines_dir.exists() else []
    models_options = sorted(
        [str(p.relative_to(BASE_DIR)).replace("\\", "/") for p in models_dir.glob("*")]
    ) if models_dir.exists() else []

    # fallback se vuoti
    if not dataset_options:
        dataset_options = ["datasets/titanic.csv"]
    if not pipeline_options:
        pipeline_options = ["pipelines/titanic_influence_preprocess_test.py"]
    if not models_options:
        pipeline_options = ["trained_models/test_titanic_model"]

    # --- Helper function to create the Neo4j Browser URL ---
    def create_neo4j_url(feature_name):
        """
        Creates a clickable URL to open a query in the Neo4j Browser.

        Args:
            feature_name (str): The name of the feature to query.

        Returns:
            str: The formatted and URL-encoded link.
        """
        base_url = "http://localhost:7474/browser/"
        

        query = f"MATCH (n: Column {{name: '{feature_name}'}}) RETURN n"
        
        # URL-encode the query to handle special characters
        encoded_query = urllib.parse.quote(query)
        
        # Construct the final URL with the pre-filled query
        return f"{base_url}?cmd=edit&arg={encoded_query}"
    # --- Interfaccia Utente ---

    st.header("1. Configuration")

    selected_dataset = st.selectbox(
        "Select dataset:",
        dataset_options
    )

    selected_pipeline = st.selectbox(
        "Select pipeline:",
        pipeline_options
    )

    selected_model = st.selectbox(
        "Select model:",
        models_options
    )
    st.header("2. Choose Explanation Method")
    
    st.markdown("### Feature Analysis")
    
    if st.button("calculate with SHAP"):
        if selected_dataset and selected_pipeline and selected_model:
            with st.spinner("Calculating global feature SHAP values (this may take a while)..."):
                cmd, returncode = run_shap(selected_model, selected_pipeline, selected_dataset)

                if returncode == 0:
                    st.success("success!")
                    st.header("3. Results")
                    try:
                        output_json = st.session_state.get('stdout', '').strip()
                        if not output_json:
                            st.warning("No output returned.")
                        else:
                            results = json.loads(output_json)
                            top_features_data = results.get("top_features")
                            if top_features_data:
                                st.subheader("Most influential Features")
                                df = pd.DataFrame(top_features_data)
                                feature_column_name = "Feature" 

                                if feature_column_name in df.columns:
                                    df["Open in Neo4j Browser"] = df[feature_column_name].apply(create_neo4j_url)
                                    st.dataframe(
                                        df,
                                        column_config={
                                            "Open in Neo4j Browser": st.column_config.LinkColumn(
                                                "Neo4j Query",
                                                display_text="Find in graph",
                                                help="Click to open the query in the Neo4j Browser"
                                            )
                                        },
                                        hide_index=True,
                                    )
                                else:
                                    st.warning(f"The specified feature column '{feature_column_name}' was not found.")
                                    st.dataframe(df)

                            image_path_str = results.get("plot_path")
                            if image_path_str:
                                image_path = Path(image_path_str)
                                if image_path.exists():
                                    st.subheader("SHAP graph")
                                    st.image(str(image_path), caption="Grafico di riepilogo SHAP")
                                else:
                                    st.warning(f"file not found at path: {image_path_str}")
                            else:
                                st.warning("'plot_path' not found")

                    except json.JSONDecodeError:
                        st.error("Error: JSON not valid.")
                        st.text_area("Output  (stdout)", st.session_state.get('stdout', ''), height=150)
                    except Exception as e:
                        st.error(f"Error visualizing the results: {e}")
                        st.text_area("Output (stdout)", st.session_state.get('stdout', ''), height=150)
        else:
            st.warning("Select dataset, pipeline and model.")

    # --- DYNAMIC INFLUENCE METHOD DISCOVERY ---
    st.markdown("### Training Data Influence")
    
    # 1. Scan the directory for scripts matching the naming convention "run_*Global.py"
    influence_scripts_dir = BASE_DIR / "influenciae"
    influence_options = {}
    
    if influence_scripts_dir.exists():
        for script_file in influence_scripts_dir.glob("*_global.py"):
            if script_file.name == "base_analyzer_global.py":
                continue
            display_name = script_file.stem.replace("_global", "")
            display_name = display_name[:1].upper() + display_name[1:]
            influence_options[display_name] = script_file.name

    # Fallback just in case directory scan fails
    if not influence_options:
        influence_options = {
            "FirstOrder": "first_order_global.py", 
            "TracIn": "tracIn_global.py"
        }

    # 2. Setup the dynamic UI Selector
    col_sel, col_btn = st.columns([2, 1])
    with col_sel:
        selected_method_name = st.selectbox("Select Influence Funcion:", list(influence_options.keys()))
    
    with col_btn:
        st.write("") # Padding to align with selectbox
        st.write("") 
        run_influence = st.button(f"Calculate {selected_method_name}")

    # 3. Execution logic
    if run_influence:
        if selected_dataset and selected_pipeline and selected_model:
            selected_script = influence_options[selected_method_name]
            
            with st.spinner(f"Calculating global {selected_method_name} influence (this may take a while)..."):
                try:
                    # Single generic runner call
                    globalInfluenceResults = run_global_influence(
                        selected_script,
                        selected_model, 
                        selected_dataset, 
                        selected_pipeline
                    )
                    
                    if globalInfluenceResults.get("status") == "success":
                        stats = globalInfluenceResults["global_statistics"]
                        top_influential = globalInfluenceResults["most_influential_samples"]
                        potential_poison = globalInfluenceResults["potential_mislabeled_samples"]
                        debug_info = globalInfluenceResults.get("debug_info", {})

                        st.success(f"{selected_method_name} Analysis Complete")
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Test Set Size", stats.get('n_test_samples_analyzed', 'N/A'))
                        c2.metric("Train Set Size", debug_info.get('X_train', 'N/A'))
                        c3.metric("Mean Influence", f"{stats['mean_influence']:.5f}")
                        c4.metric("Max Influence", f"{stats['max_influence']:.5f}")
                        
                        st.markdown("---")

                        table_config = {
                            "Graph Link": st.column_config.LinkColumn(
                                "Graph Context",
                                display_text="Find in Graph", 
                                help="Click to open this data point in Neo4j Browser"
                            ),
                            "mean_influence_score": st.column_config.NumberColumn(
                                "Influence Score",
                                format="%.5f"
                            )
                        }

                        # TOP INFLUENTIAL 
                        st.subheader("Proponents")
                        st.caption("Training examples that contributed most positively.")
                        
                        df_top = format_influence_df(top_influential)
                        st.dataframe(
                            df_top,
                            use_container_width=True,
                            column_config=table_config,
                            hide_index=True
                        )

                        # POTENTIAL POISONING 
                        st.subheader("Opponents")
                        st.caption("Training examples with strong negative influence.")
                        
                        df_poison = format_influence_df(potential_poison)
                        st.dataframe(
                            df_poison.style.background_gradient(cmap="Reds_r", subset=["mean_influence_score"]),
                            use_container_width=True,
                            column_config=table_config,
                            hide_index=True
                        )

                        with st.expander("View Raw JSON Output"):
                            st.json(potential_poison)

                    else:
                        st.error(f"Analysis failed: {globalInfluenceResults.get('error')}")

                except RuntimeError as e:
                    st.error("Execution Error")
                    st.code(str(e))
                except Exception as e:
                    st.error(f"Unexpected Error: {e}")
        else:
            st.warning("Please select a dataset, pipeline, and model before running the analysis.")

# ====================== PAGINA: Local ANALYSIS ======================
if page == "Local Analysis":
    st.title("Local Analysis")

    # ---- SIDEBAR / SETUP ----
    st.sidebar.header("Configuration")

    datasets_dir = BASE_DIR / "datasets"
    pipelines_dir = BASE_DIR / "pipelines"
    models_dir = BASE_DIR / "trained_models"

    dataset_options = sorted(
        [str(p.relative_to(BASE_DIR)).replace("\\", "/") for p in datasets_dir.glob("*.csv")]
    ) if datasets_dir.exists() else []
    pipeline_options = sorted(
        [str(p.relative_to(BASE_DIR)).replace("\\", "/") for p in pipelines_dir.glob("*.py")]
    ) if pipelines_dir.exists() else []
    models_options = sorted(
        [str(p.relative_to(BASE_DIR)).replace("\\", "/") for p in models_dir.glob("*")]
    ) if models_dir.exists() else []

    # fallback se vuoti
    if not dataset_options:
        dataset_options = ["datasets/titanic.csv"]
    if not pipeline_options:
        pipeline_options = ["pipelines/titanic_influence_preprocess_test.py"]
    if not models_options:
        pipeline_options = ["trained_models/test_titanic_model"]

    selected_dataset = st.sidebar.selectbox("Dataset", dataset_options)
    selected_pipeline = st.sidebar.selectbox("Pipeline", pipeline_options)
    selected_model = st.sidebar.selectbox("Model", models_options)



    # Initialize Session State
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "selected_sample_index" not in st.session_state:
        st.session_state.selected_sample_index = None

    #  EXECUTION BUTTON 
    if st.sidebar.button("Model Prediction Overview"):
        
        res = run_global_analysis(
            BASE_DIR / selected_dataset,
            BASE_DIR / selected_pipeline,
            BASE_DIR / selected_model
        )
        st.session_state.analysis_results = res
        st.session_state.selected_sample_index = None # Reset selection on new run

    # RESULTS DISPLAY
    if st.session_state.analysis_results:
        data = st.session_state.analysis_results
        X_test = data["X_test"]
        y_test = data["y_test"]
        y_pred = data["y_pred"]
        task_type = data["task_type"]

        # Combine for easy filtering
        df_display = X_test.copy()
        df_display["True Label"] = y_test
        df_display["Predicted"] = y_pred
        
        st.divider()
        
       
        # Overview: Filter the data based on errors
        
        st.subheader("Prediction Overview: Filter by Performance")
        
        filtered_df = df_display

        if task_type == "classification":
            # Create a confusion category column
            df_display["Outcome"] = df_display.apply(
                lambda x: "Correct" if x["True Label"] == x["Predicted"] else "Error", axis=1
            )
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Simple Bar Chart of Errors
                outcome_counts = df_display["Outcome"].value_counts()
                st.bar_chart(outcome_counts)

            with col2:
                # Filter interaction
                filter_option = st.radio("Show rows:", ["All", "Only Errors", "Only Correct"], horizontal=True)
                
                if filter_option == "Only Errors":
                    filtered_df = df_display[df_display["Outcome"] == "Error"]
                elif filter_option == "Only Correct":
                    filtered_df = df_display[df_display["Outcome"] == "Correct"]
                
                st.info(f"Showing {len(filtered_df)} rows based on filter.")

        else: # Regression
            df_display["Residual"] = df_display["True Label"] - df_display["Predicted"]
            df_display["Abs Error"] = df_display["Residual"].abs()
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.scatter_chart(
                    df_display,
                    x="True Label",
                    y="Predicted",
                    color="Abs Error" 
                )
            
            with col2:
                min_err = st.slider("Filter by Minimum Absolute Error", 
                                    float(df_display["Abs Error"].min()), 
                                    float(df_display["Abs Error"].max()), 
                                    0.0)
                filtered_df = df_display[df_display["Abs Error"] >= min_err]
                st.info(f"Showing {len(filtered_df)} rows with Error > {min_err:.2f}")

        
        # LOCAL SELECTION: Pick the sample for Influence
        
        st.subheader("Select a Sample for Local Explanation")
        st.markdown("Select a row in the table below to analyze its influence.")

        # dataframe selection
        event = st.dataframe(
            filtered_df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        if event.selection.rows:
            # index relative to the filtered dataframe
            selected_row_idx = event.selection.rows[0]
            # actual index from the original dataset
            original_index = filtered_df.index[selected_row_idx]
            
            st.session_state.selected_sample_index = original_index
            
            # Display selected data
            selected_row_data = df_display.loc[original_index]
            
            st.success(f"Selected Sample Index: {original_index}")
            
            #  First Order
            
            st.markdown("Influence Analysis")
        
            col_btn, col_info = st.columns([1, 4])
            
            with col_btn:
                # Renamed variable for clarity
                run_fo_clicked = st.button("Run FirstOrder on Selection") 
                # New button
                run_tracin_clicked = st.button("Run TracIn on Selection") 

            # Logic for FirstOrder
            if run_fo_clicked:
                with st.spinner("Calculating FirstOrder Influence..."):
                    st.session_state.influence_result = run_FirstOrder(
                        BASE_DIR / selected_model,
                        BASE_DIR / selected_dataset,
                        BASE_DIR / selected_pipeline,
                        selected_row_data
                    )
                    st.toast("Calculated FirstOrder Influence!", icon="🚀")

            # Logic for TracIn (The new functionality)
            if run_tracin_clicked:
                with st.spinner("Calculating TracIn Influence..."):
                    st.session_state.influence_result = run_TracIn(
                        BASE_DIR / selected_model,
                        BASE_DIR / selected_dataset,
                        BASE_DIR / selected_pipeline,
                        selected_row_data
                    )
                    st.toast("Calculated TracIn Influence!", icon="🚀")

            with col_info:
                with st.expander("View Selected Row Details"):
                    st.json(selected_row_data.to_dict(), expanded=False)

            
            # DISPLAY RESULTS
            
            if "influence_result" in st.session_state and st.session_state.influence_result:
                res = st.session_state.influence_result
                
                # Check if successful
                if res.get('status') == 'success':
                    st.divider()
                    st.subheader("📊 Analysis Results")

                    # Display Prediction
                    preds = res.get('prediction', [[0]])[0] 
                    formatted_preds = ", ".join([f"{p:.4f}" for p in preds])
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Status", "Success", delta_color="normal")
                    c2.metric("Predicted Output", formatted_preds)
                    c3.metric("Outcome", res['sample_analyzed'].get('Outcome', 'N/A'))

                    # Process Top Influential Points
                    influential_points = res.get('top_influential_points', [])
                    
                    if influential_points:
                        flattened_data = []
                        for point in influential_points:
                            row = point['features'].copy()
                            
                            # --- GENERATE LINK ---
                            neo4j_url = create_fingerprint_url(row)
                            
                            # Add data to the row for display
                            row['Influence Score'] = point['influence_score']
                            row['Label'] = str(point['label'])
                            
                            # Add the generated URL to a specific column
                            row['Graph Query'] = neo4j_url
                            
                            flattened_data.append(row)
                        
                        df_results = pd.DataFrame(flattened_data)

                        # Reorder columns: Score, Label, Graph Link, then Features
                        # Exclude 'Graph Query' from the loop so we can place it manually
                        feature_cols = [c for c in df_results.columns if c not in ['Influence Score', 'Label', 'Graph Query']]
                        cols = ['Influence Score', 'Label', 'Graph Query'] + feature_cols
                        df_results = df_results[cols]

                        st.write("Most Influential Training Points")
                        
                        st.dataframe(
                            df_results,
                            use_container_width=True,
                            column_config={
                                "Influence Score": st.column_config.ProgressColumn(
                                    "Influence",
                                    format="%.4f",
                                    min_value=df_results['Influence Score'].min(),
                                    max_value=df_results['Influence Score'].max(),
                                ),
                                "Label": st.column_config.TextColumn("True Label"),
                                
                                # --- CONFIGURE THE BUTTON ---
                                "Graph Query": st.column_config.LinkColumn(
                                    "Graph Analysis",
                                    display_text="Find entity in graph", 
                                    help="Click to run a Cypher query finding this exact row based on its values"
                                )
                            },
                            hide_index=True
                        )
                        
                    else:
                        st.warning("No influential points returned.")
                        
                    # Raw JSON Viewer (Hidden by default for cleanliness)
                    with st.expander("View Raw JSON Output"):
                        st.json(res)
                else:
                    st.error("Analysis returned a non-success status.")
                    st.json(res)

        else:
            st.info("👆 Please select a row from the table above.")