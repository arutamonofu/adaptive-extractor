#!/usr/bin/env python3
"""
Batch inference script for running extraction predictions on processed documents.

This script loads processed documents, runs extraction predictions using a UniversalExtractor agent,
and saves the results to output files.
"""

import os
os.environ["DSP_CACHEBOOL"] = "0"

import argparse
import json
import logging
import pprint
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm

import dspy
from aee import setup_logging, settings as default_settings, ProcessedDocument, UniversalExtractor
from aee.llm import setup_student
from aee.tasks import TASK_REGISTRY

logger = setup_logging()


def save_dspy_log(doc_id: str, logs_dir: Path) -> bool:
    """
    Save the DSPy interaction log for a document.
    
    Args:
        doc_id: Document identifier
        logs_dir: Directory to save logs
        
    Returns:
        bool: True if log was saved successfully, False otherwise
    """
    log_file = logs_dir / f"{doc_id}_predict.log"
    
    try:
        history = dspy.settings.lm.history
        last_interaction = history[-1] if history else "HISTORY IS EMPTY"
        
        with open(log_file, "w", encoding="utf-8") as f:
            formatted_interaction = pprint.pformat(last_interaction, width=120)
            f.write(formatted_interaction)
            
        return True
    except Exception as e:
        logger.error(f"Failed to write prediction log for {doc_id}: {e}")
        return False


def process_document(
    json_file: Path, 
    agent: UniversalExtractor, 
    out_dir: Path, 
    logs_dir: Path
) -> bool:
    """
    Process a single document file.
    
    Args:
        json_file: Path to the input JSON file
        agent: The extraction agent
        out_dir: Output directory for predictions
        logs_dir: Directory for logs
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    doc_id = json_file.stem
    
    try:
        # Load document
        doc_content = json_file.read_text(encoding="utf-8")
        doc = ProcessedDocument.model_validate_json(doc_content)
        
        if not doc.text_content:
            logger.warning(f"Document {doc_id} has no text content, skipping")
            return False

        # Run prediction
        prediction = agent(document_text=doc.text_content)

        if prediction.extracted_data is None:
            raise ValueError("Model returned None for extracted_data")

        # Prepare output data
        output_data = {
            "source_metadata": doc.metadata.model_dump(),
            "extraction": prediction.extracted_data.model_dump(mode='json'),
            "reasoning": getattr(prediction, 'reasoning', "No reasoning provided")
        }

        # Save prediction
        res_path = out_dir / f"{doc_id}_prediction.json"
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Save DSPy log
        save_dspy_log(doc_id, logs_dir)
        
        return True

    except Exception as e:
        logger.error(f"Error processing {doc_id}: {e}")
        save_dspy_log(doc_id, logs_dir)
        return False


def load_agent(agent_path: Optional[Path], task_signature: Any, logs_dir: Path) -> Optional[UniversalExtractor]:
    """
    Load or create an extraction agent.
    
    Args:
        agent_path: Path to a pre-trained agent (optional)
        task_signature: Task signature for the agent
        logs_dir: Directory for logs
        
    Returns:
        UniversalExtractor: Configured agent or None if failed
    """
    try:
        agent = UniversalExtractor(task_signature)
        
        if agent_path is not None:
            if agent_path.exists():
                agent.load(str(agent_path))
                logger.info(f"Loaded optimized agent from {agent_path}")
            else:
                logger.warning(f"Agent file not found at {agent_path}. Running in Zero-Shot mode.")
        else:
            logger.info("Running in Zero-Shot mode with default agent.")
            
        return agent
    except Exception as e:
        logger.error(f"Failed to load agent: {e}")
        return None


def validate_task_configuration(task_name: str) -> Optional[dict]:
    """
    Validate and retrieve task configuration.
    
    Args:
        task_name: Name of the task to validate
        
    Returns:
        dict: Task configuration or None if invalid
    """
    task_conf = TASK_REGISTRY.get(task_name)
    if not task_conf:
        logger.error(f"Task '{task_name}' not found in registry.")
        return None
    
    return task_conf


def setup_directories(in_dir: Path, out_dir: Path, logs_dir: Path) -> bool:
    """
    Set up required directories.
    
    Args:
        in_dir: Input directory
        out_dir: Output directory
        logs_dir: Logs directory
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Check if input directory exists
        if not in_dir.exists():
            logger.error(f"Input directory does not exist: {in_dir}")
            return False
            
        # Create output and logs directories
        out_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up directories: {e}")
        return False


def main() -> None:
    """Main entry point for the prediction script."""
    parser = argparse.ArgumentParser(description="Run batch inference with an extraction agent.")
    parser.add_argument("--config", type=str, help="Path to custom configuration YAML.")
    parser.add_argument("--input", type=str, help="Override directory of processed JSON documents.")
    parser.add_argument("--output", type=str, help="Override base directory for predictions.")
    parser.add_argument("--agent_path", type=str, help="Path to an optimized agent JSON file.")
    args = parser.parse_args()

    # Load custom config if provided
    current_settings = default_settings
    if args.config:
        try:
            current_settings = current_settings.load(args.config)
            logger.info(f"Loaded custom config from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load custom config from {args.config}: {e}")
            return

    # Validate task configuration
    task_name = current_settings.task.name
    task_conf = validate_task_configuration(task_name)
    if not task_conf:
        return

    # Set up LLM
    try:
        setup_student(config=current_settings)
    except Exception as e:
        logger.error(f"Failed to set up LLM: {e}")
        return
    
    # Create logs directory
    logs_dir = current_settings.paths.logs_dir / "predict"
    
    # Load agent
    agent_path = Path(args.agent_path) if args.agent_path else None
    agent = load_agent(agent_path, task_conf["signature"], logs_dir)
    
    if agent is None:
        logger.error("Failed to initialize extraction agent.")
        return
    
    # Set up input/output directories
    in_dir = Path(args.input) if args.input else current_settings.paths.parsed_dir
    out_dir = Path(args.output) if args.output else current_settings.paths.predictions_dir
    
    # Setup directories
    if not setup_directories(in_dir, out_dir, logs_dir):
        return
    
    # Find files to process
    files_to_process = list(in_dir.glob("*.json"))
    if not files_to_process:
        logger.warning(f"No JSON files found in {in_dir}")
        return
        
    logger.info(f"Found {len(files_to_process)} documents in {in_dir}")

    # Process documents
    successful = 0
    for json_file in tqdm(files_to_process, desc=f"Predicting [{task_name}]"):
        if process_document(json_file, agent, out_dir, logs_dir):
            successful += 1

    logger.info(f"Prediction complete. {successful}/{len(files_to_process)} documents processed successfully. "
                f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()