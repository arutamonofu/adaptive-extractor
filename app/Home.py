# app/Home.py

import streamlit as st
import pandas as pd
from pathlib import Path

from aee.core.config import settings
from aee.tasks import TASK_REGISTRY

st.set_page_config(
    page_title="AutoEvoExtractor",
    page_icon="🧬",
    layout="wide"
)

st.title("AutoEvoExtractor")
st.markdown("### Dashboard & Configuration")

# --- Global Sidebar ---
with st.sidebar:
    st.header("Credentials")
    
    api_key = st.text_input("Gemini API Key", type="password", value=settings.student_api_key)
    if api_key:
        settings.student_api_key = api_key
        st.session_state["api_key"] = api_key
        st.success("API Key set!")
    else:
        st.warning("Please enter API Key.")

    st.divider()
    st.info("Use the sidebar to navigate.")

# --- Status ---
st.subheader("System Status")

col1, col2, col3 = st.columns(3)

raw_dir = Path("data/raw")
raw_count = len(list(raw_dir.glob("*.pdf"))) if raw_dir.exists() else 0

proc_dir = Path("data/processed")
proc_count = len(list(proc_dir.glob("*.json"))) if proc_dir.exists() else 0

task_name = st.selectbox("Active Task Domain", list(TASK_REGISTRY.keys()))
st.session_state["current_task"] = task_name
gt_path = Path("data/ground_truth") / f"{task_name}.csv"
gt_rows = 0
if gt_path.exists():
    try:
        gt_rows = len(pd.read_csv(gt_path))
    except Exception: 
        pass

with col1:
    st.metric("Raw PDFs", raw_count)
with col2:
    st.metric("Processed Docs (Library)", proc_count)
with col3:
    st.metric("Ground Truth Examples", gt_rows)

st.divider()

st.markdown("""
### Workflow Guide

1.  **Training Studio:** A unified workspace to build your dataset and train the agent.
    *   *Tab 1 (Library):* Upload and parse PDFs to build the training corpus.
    *   *Tab 2 (Optimizer):* Run evolutionary algorithms on the library data.
2.  **Playground:** Test your agent (Zero-shot or Optimized) on any document.
""")