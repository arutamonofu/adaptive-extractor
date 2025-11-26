# app/pages/Training_Studio.py

import streamlit as st
import contextlib
from io import StringIO
from pathlib import Path

# Ingestion imports
from aee.ingestion import DoclingParser, TextCleaner

# Optimizer imports
from dspy.teleprompt import MIPROv2
from aee.eval import TaskMetric
from aee.agents.extractor import UniversalExtractor
from aee.llm import setup_student, setup_teacher
from aee.tasks import TASK_REGISTRY
from aee.utils.io import load_ground_truth
from aee.utils.dataset import create_training_set

st.set_page_config(page_title="Training Studio", page_icon="🧠", layout="wide")

st.title("Training Studio")
st.markdown("Manage your training corpus and optimize the agent's prompts.")

task_name = st.session_state.get("current_task", "nanozymes")
task_conf = TASK_REGISTRY[task_name]

# --- Tabs for Logical Separation ---
tab_library, tab_optimizer = st.tabs(["Library Manager", "Prompt Optimizer"])

# ==========================================
# TAB 1: LIBRARY MANAGER (Ingestion)
# ==========================================
with tab_library:
    st.subheader("1. Build Training Corpus")
    st.markdown("""
    Upload PDFs here. They will be parsed and saved to `data/processed`.
    These files serve as the knowledge base for the Optimizer.
    """)
    
    uploaded_files = st.file_uploader(
        "Upload Scientific Articles (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button(f"Process {len(uploaded_files)} Files", type="primary"):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            raw_dir = Path("data/raw")
            proc_dir = Path("data/processed")
            raw_dir.mkdir(parents=True, exist_ok=True)
            proc_dir.mkdir(parents=True, exist_ok=True)
            
            # Simple Parser Init
            parser = DoclingParser()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}...")
                
                # Save Raw
                file_path = raw_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Parse
                    doc = parser.parse(file_path)
                    doc.text_content = TextCleaner.clean_docling_markdown(doc.text_content)
                    
                    # Save Processed
                    json_path = proc_dir / f"{file_path.stem}.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        f.write(doc.model_dump_json(indent=2))
                        
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.text("Ingestion Complete!")
            st.success(f"Added {len(uploaded_files)} documents to Library.")
            # Rerun to update counts/lists if needed
            st.rerun()

    # Library Preview
    st.divider()
    proc_dir = Path("data/processed")
    if proc_dir.exists():
        files = list(proc_dir.glob("*.json"))
        st.write(f"**Current Library Size:** {len(files)} documents")
        
        if files:
            with st.expander("View Library Content"):
                selected_file = st.selectbox("Select file to preview:", [f.name for f in files])
                if selected_file:
                    import json
                    with open(proc_dir / selected_file, "r") as f:
                        st.json(json.load(f), expanded=False)
    else:
        st.info("Library is empty.")


# ==========================================
# TAB 2: OPTIMIZER (MIPROv2)
# ==========================================
with tab_optimizer:
    st.subheader("2. Evolutionary Optimization")
    st.markdown("Use the Library data to improve the agent's prompts via MIPROv2.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Target Task: **{task_name}**")
        
        # Check Data Availability
        gt_path = Path("data/ground_truth") / f"{task_name}.csv"
        proc_dir = Path("data/processed")
        
        gt_available = gt_path.exists()
        proc_count = len(list(proc_dir.glob("*.json"))) if proc_dir.exists() else 0
        
        st.write(f"• Ground Truth: **{'Available' if gt_available else 'Missing'}**")
        st.write(f"• Library Docs: **{proc_count}**")
        
        train_size = st.slider("Max Training Examples", 3, 50, 5)

    with col2:
        trials = st.slider("Optimization Trials", 5, 30, 10)
        candidates = st.slider("Prompt Candidates", 1, 7, 3)

    st.divider()

    if not gt_available or proc_count == 0:
        st.warning("Prerequisites not met. Please upload data in the 'Library Manager' tab and ensure Ground Truth exists.")
    else:
        if st.button("Start Optimization", type="primary"):
            log_stream = StringIO()
            status = st.status("Running Optimization...", expanded=True)
            
            try:
                with status:
                    st.write("🔹 Loading Datasets...")
                    gt_data = load_ground_truth(gt_path, task_conf["row_converter"])
                    trainset = create_training_set(proc_dir, gt_data, task_conf, train_size)
                    
                    if not trainset:
                        st.error("No intersection found between Ground Truth and Library files.")
                        st.stop()
                        
                    st.write(f"🔹 Prepared {len(trainset)} valid examples.")
                    st.write("🔹 Initializing LLMs & Agent...")
                    
                    teacher = setup_teacher()
                    student = setup_student()
                    agent = UniversalExtractor(task_conf["signature"])
                    
                    teleprompter = MIPROv2(
                        prompt_model=teacher,
                        task_model=student,
                        metric=TaskMetric(task_conf),
                        num_candidates=candidates,
                        init_temperature=0.5,
                        verbose=True,
                        auto=None
                    )
                    
                    st.write("🔹 Running Evolution (this may take time)...")
                    with contextlib.redirect_stdout(log_stream):
                        optimized_agent = teleprompter.compile(
                            agent,
                            trainset=trainset,
                            num_trials=trials,
                            max_bootstrapped_demos=2,
                            max_labeled_demos=2,
                            minibatch=False
                        )
                    
                    # Save
                    save_path = Path("data/artifacts") / f"optimized_{task_name}.json"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    optimized_agent.save(str(save_path))
                    
                    status.update(label="Complete!", state="complete", expanded=False)
                    st.success(f"Agent saved to `{save_path}`")
                    
                    # Download
                    with open(save_path, "r") as f:
                        st.download_button(
                            "💾 Download Optimized Agent", 
                            f, 
                            file_name="optimized_agent.json",
                            mime="application/json"
                        )

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(f"Error: {e}")
            
            with st.expander("View Execution Logs"):
                st.code(log_stream.getvalue())