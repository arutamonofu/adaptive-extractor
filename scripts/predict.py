# scripts/predict.py

import os
os.environ["DSP_CACHEBOOL"] = "1"

import argparse
import json
import logging
import pprint
from pathlib import Path
from tqdm import tqdm

import dspy
from aee.core.logging import setup_logging
from aee.core.config import settings
from aee.core.types import ProcessedDocument
from aee.llm import setup_student
from aee.agents import UniversalExtractor
from aee.tasks import TASK_REGISTRY

logger = setup_logging()

def save_dspy_log(doc_id: str):
    logs_dir = settings.paths.logs_dir / "predict"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = logs_dir / f"{doc_id}_predict.log"
    
    try:
        history = dspy.settings.lm.history
        last_interaction = history[-1] if history else "HISTORY IS EMPTY"
        
        with open(log_file, "w", encoding="utf-8") as f:
            formatted_interaction = pprint.pformat(last_interaction, width=120)
            f.write(formatted_interaction)
            
    except Exception as e:
        logger.error(f"Failed to write prediction log for {doc_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run batch inference with an extraction agent.")
    parser.add_argument("--config", type=str, help="Path to custom configuration YAML.")
    parser.add_argument("--input", type=str, help="Override directory of processed JSON documents.")
    parser.add_argument("--output", type=str, help="Override base directory for predictions.")
    parser.add_argument("--agent_path", type=str, help="Path to an optimized agent JSON file.")
    args = parser.parse_args()

    if args.config:
        global settings
        settings = settings.load(args.config)

    task_name = settings.task.name
    task_conf = TASK_REGISTRY.get(task_name)
    if not task_conf:
        logger.error(f"Task '{task_name}' not found in registry.")
        return

    setup_student()
    agent = UniversalExtractor(task_conf["signature"])
    
    if args.agent_path:
        agent_path = Path(args.agent_path)
        if agent_path.exists():
            agent.load(str(agent_path))
            logger.info(f"Loaded optimized agent from {agent_path}")
        else:
            logger.warning(f"Agent file not found at {agent_path}. Running in Zero-Shot mode.")

    in_dir = Path(args.input) if args.input else settings.paths.parsed_dir
    out_dir = (Path(args.output) if args.output else settings.paths.predictions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_process = list(in_dir.glob("*.json"))
    logger.info(f"Found {len(files_to_process)} documents in {in_dir}")

    for json_file in tqdm(files_to_process, desc=f"Predicting [{task_name}]"):
        doc_id = json_file.stem 
        
        try:
            doc = ProcessedDocument.model_validate_json(json_file.read_text(encoding="utf-8"))
            if not doc.text_content:
                continue

            prediction = agent(document_text=doc.text_content)

            if prediction.extracted_data is None:
                raise ValueError("Model returned None for extracted_data")

            output_data = {
                "source_metadata": doc.metadata.model_dump(),
                "extraction": prediction.extracted_data.model_dump(mode='json'),
                "reasoning": getattr(prediction, 'reasoning', "No reasoning provided")
            }

            res_path = out_dir / f"{doc_id}_prediction.json"
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            save_dspy_log(doc_id)

        except Exception as e:
            logger.error(f"Error processing {doc_id}: {e}")
            save_dspy_log(doc_id)
            continue

    logger.info(f"Prediction complete. Results saved to: {out_dir}")

if __name__ == "__main__":
    main()