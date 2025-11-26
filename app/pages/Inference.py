# app/pages/Inference.py

import json
import tempfile
import pandas as pd
import streamlit as st
from pathlib import Path

# Project imports
from aee.agents.extractor import UniversalExtractor
from aee.llm import setup_student
from aee.tasks import TASK_REGISTRY
from aee.ingestion import DoclingParser, TextCleaner
from aee.utils.io import load_ground_truth
from aee.eval import ExperimentMatcher

st.set_page_config(page_title="Inference", page_icon="🧪", layout="wide")

st.title("Inference Playground")
st.markdown("Test the agent on **new files** or existing files from the library.")

# --- Context Setup ---
task_name = st.session_state.get("current_task", "nanozymes")
task_conf = TASK_REGISTRY[task_name]

# --- Sidebar: Model Configuration ---
with st.sidebar:
    st.header("Model Configuration")
    uploaded_agent = st.file_uploader("Load Optimized Agent (.json)", type=["json"])
    
    if uploaded_agent:
        st.success("Custom weights loaded")
    else:
        st.info("ℹUsing Zero-Shot (Baseline)")

# --- 1. Input Source Selection ---
st.subheader("1. Input Source")
input_method = st.radio("Select Source:", ["Upload New PDF", "Select from Library"], horizontal=True)

processed_text = None
filename = "unknown.pdf"

# Initialize cache for uploaded text
if "temp_text" not in st.session_state:
    st.session_state["temp_text"] = None

if input_method == "Upload New PDF":
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf:
        filename = uploaded_pdf.name
        if st.button("Parse Document", key="parse_new"):
            with st.spinner("Parsing PDF with Docling..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    tmp_path = tmp.name
                
                try:
                    parser = DoclingParser()
                    doc = parser.parse(tmp_path)
                    clean_txt = TextCleaner.clean_docling_markdown(doc.text_content)
                    st.session_state["temp_text"] = clean_txt # Cache result
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
    
    # Retrieve from cache
    if st.session_state["temp_text"]:
        processed_text = st.session_state["temp_text"]

elif input_method == "Select from Library":
    proc_dir = Path("data/processed")
    if proc_dir.exists():
        files = list(proc_dir.glob("*.json"))
        if files:
            selected = st.selectbox("Select File", [f.name for f in files])
            if selected:
                filename = selected
                with open(proc_dir / selected) as f:
                    data = json.load(f)
                    processed_text = data.get("text_content")
        else:
            st.warning("Library is empty. Go to 'Training Studio' to ingest files.")

# Document Preview
if processed_text:
    with st.expander("View Parsed Content"):
        st.markdown(processed_text[:2000] + "...")

# --- 2. Execution & Analysis ---
st.subheader("2. Extraction & Analysis")

if processed_text:
    if st.button("🚀 Run Agent", type="primary"):
        with st.spinner(f"Running extraction for '{task_name}'..."):
            try:
                # Initialize LLM & Agent
                setup_student()
                agent = UniversalExtractor(task_conf["signature"])
                
                # Load optimized weights if provided
                if uploaded_agent:
                    # Fix: DSPy requires .json suffix to load correctly
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                        tmp.write(uploaded_agent.read())
                        tmp_path = tmp.name
                    
                    try:
                        agent.load(tmp_path)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                
                # Run Inference
                pred = agent(document_text=processed_text)
                exps = pred.extracted_data.experiments
                
                # --- Display Results ---
                st.divider()
                st.markdown("### Results")
                
                # 1. Reasoning
                with st.expander("Chain of Thought", expanded=True):
                    st.write(pred.reasoning)
                
                # 2. Data Table
                if exps:
                    df = pd.DataFrame([x.model_dump() for x in exps]).dropna(axis=1, how='all')
                    st.dataframe(df, use_container_width=True)
                    
                    # 3. Single-Document Evaluation (If available)
                    # Only possible if file is from library (known filename) and GT exists
                    if input_method == "Select from Library":
                        gt_path = Path("data/ground_truth") / f"{task_name}.csv"
                        
                        if gt_path.exists():
                            gt_dict = load_ground_truth(gt_path, task_conf["row_converter"])
                            file_stem = Path(filename).stem
                            
                            if file_stem in gt_dict:
                                st.markdown("#### Performance vs Ground Truth")
                                ground_truth = gt_dict[file_stem]
                                
                                matcher = ExperimentMatcher(task_conf["compare_fields"])
                                metrics = matcher._compute_metrics([exps], [ground_truth])
                                
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Precision", f"{metrics['precision']:.2f}")
                                m2.metric("Recall", f"{metrics['recall']:.2f}")
                                m3.metric("F1-Score", f"{metrics['f1']:.2f}")
                                
                                with st.expander("View Ground Truth Data"):
                                    gt_df = pd.DataFrame([x.model_dump() for x in ground_truth]).dropna(axis=1, how='all')
                                    st.dataframe(gt_df, use_container_width=True)

                    # Download Button
                    st.download_button(
                        "Download Result JSON",
                        data=pred.extracted_data.model_dump_json(indent=2),
                        file_name=f"{Path(filename).stem}_result.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No experiments found in the text.")
                    
            except Exception as e:
                st.error(f"Inference Error: {e}")
else:
    st.info("Please select or upload a document to proceed.")