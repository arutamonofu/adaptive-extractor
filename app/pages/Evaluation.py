# app/pages/Evaluation.py

import streamlit as st
import pandas as pd
import tempfile
import json
from pathlib import Path

# Project imports
from aee.eval import ExperimentMatcher
from aee.agents.extractor import UniversalExtractor
from aee.llm import setup_student
from aee.tasks import TASK_REGISTRY
from aee.utils.io import load_ground_truth

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")

st.title("Model Evaluation")
st.markdown("Run a benchmark on the **Test Set** to calculate Precision, Recall, and F1-Score.")

task_name = st.session_state.get("current_task", "nanozymes")
task_conf = TASK_REGISTRY[task_name]

# --- Sidebar: Model & Settings ---
with st.sidebar:
    st.header("Benchmark Settings")
    uploaded_agent = st.file_uploader("Load Optimized Agent (.json)", type=["json"])
    
    st.divider()
    
    # Split Selection
    split_path = Path("data/splits.json")
    target_split = "test"
    
    if split_path.exists():
        with open(split_path) as f:
            splits = json.load(f)
        split_options = list(splits.keys())
        # Filter out metadata
        split_options = [s for s in split_options if isinstance(splits[s], list)]
        target_split = st.selectbox("Select Data Split", split_options, index=split_options.index("test") if "test" in split_options else 0)
        
        file_ids = splits[target_split]
        st.caption(f"Found {len(file_ids)} documents in '{target_split}' split.")
    else:
        st.error("No splits.json found! Run 'create_splits.py' or generate in Training Studio.")
        file_ids = []

    limit = st.slider("Limit documents (for speed)", 5, 100, 20)

# --- Main Logic ---

if st.button("Run Benchmark", type="primary", disabled=not file_ids):
    
    # 1. Prepare Data
    gt_path = Path("data/ground_truth") / f"{task_name}.csv"
    proc_dir = Path("data/processed")
    
    if not gt_path.exists():
        st.error("Ground Truth CSV missing.")
        st.stop()
        
    status_box = st.status("Running Evaluation...", expanded=True)
    
    try:
        with status_box:
            st.write("🔹 Loading Ground Truth...")
            gt_dict = load_ground_truth(gt_path, task_conf["row_converter"])
            
            # Filter files that exist in Library AND in the selected Split
            valid_files = []
            for fid in file_ids:
                if (proc_dir / f"{fid}.json").exists() and fid in gt_dict:
                    valid_files.append(fid)
            
            # Apply limit
            valid_files = valid_files[:limit]
            
            if not valid_files:
                st.error("No valid files found intersecting Split, Library, and Ground Truth.")
                st.stop()
                
            st.write(f"🔹 Evaluating on {len(valid_files)} documents...")
            
            # Init Model
            setup_student()
            agent = UniversalExtractor(task_conf["signature"])
            
            if uploaded_agent:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                    tmp.write(uploaded_agent.read())
                    tmp_path = tmp.name
                try:
                    agent.load(tmp_path)
                    st.write("🔹 Loaded Optimized Agent.")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                st.write("🔹 Using Zero-Shot Agent.")

            # Run Batch Inference
            batch_preds = []
            batch_gts = []
            
            prog_bar = st.progress(0)
            
            for i, fid in enumerate(valid_files):
                # Load Doc
                with open(proc_dir / f"{fid}.json") as f:
                    doc_data = json.load(f)
                    text = doc_data["text_content"]
                
                # Predict
                pred = agent(document_text=text)
                
                # Collect
                batch_preds.append(pred.extracted_data.experiments)
                batch_gts.append(gt_dict[fid])
                
                prog_bar.progress((i + 1) / len(valid_files))
            
            st.write("🔹 Calculating Metrics...")
            matcher = ExperimentMatcher(task_conf["compare_fields"])
            metrics = matcher.evaluate_dataset(batch_preds, batch_gts)
            
            status_box.update(label="Benchmark Complete!", state="complete", expanded=False)

        # --- Display Results ---
        st.divider()
        st.subheader(f"Results on '{target_split}' split")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{metrics['precision']:.4f}")
        m2.metric("Recall", f"{metrics['recall']:.4f}")
        m3.metric("F1-Score", f"{metrics['f1']:.4f}")
        
        st.caption(f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}")

    except Exception as e:
        st.error(f"Evaluation failed: {e}")