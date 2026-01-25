# scripts/predict.py

import os
os.environ["DSP_CACHEBOOL"] = "1"

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import logging
import pprint

from aee.core.logging import setup_logging
from aee.core import ProcessedDocument
from aee.llm import setup_student
from aee.agents import UniversalExtractor
from aee.tasks import TASK_REGISTRY

logger = setup_logging()

import dspy

def save_dspy_log(doc_id: str):
    """
    Сохраняет ошибку и историю в простой текстовый файл .log.
    Использует pprint, который не падает на сложных объектах.
    """
    log_dir = Path("logs/predict")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем как .log (обычный текст)
    log_file = log_dir / f"{doc_id}_predict.log"
    
    try:
        history = dspy.settings.lm.history
        last_interaction = history[-1] if history else "HISTORY IS EMPTY"
        
        with open(log_file, "w", encoding="utf-8") as f:
            formatted_interaction = pprint.pformat(last_interaction, width=120)
            f.write(formatted_interaction)
            
        print(f"\nRaw log saved to: {log_file}")
            
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to write simple log: {e}")


def main():
    """Main function to handle batch inference."""
    parser = argparse.ArgumentParser(
        description="Run batch inference with an extraction agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--task", type=str, default="nanozymes", choices=TASK_REGISTRY.keys(), help="The extraction task to perform.")
    parser.add_argument("--input", type=str, default="data/parsed/test", help="Directory of processed JSON documents.")
    parser.add_argument("--output", type=str, default="data/predictions", help="Base directory to save prediction JSON files.")
    parser.add_argument("--agent_path", type=str, default=None, help="Path to an optimized agent JSON file to load.")
    args = parser.parse_args()

    # --- 1. Setup ---
    task_conf = TASK_REGISTRY[args.task]
    
    # Initialize LLM and Agent
    setup_student()
    agent = UniversalExtractor(task_conf["signature"])
    
    # Load optimized agent if provided
    if args.agent_path:
        agent_path = Path(args.agent_path)
        if agent_path.exists():
            agent.load(str(agent_path))
            logger.info(f"Loaded optimized agent from {agent_path}")
        else:
            logger.error(f"Agent file not found at {agent_path}. Running in Zero-Shot mode.")
    else:
        logger.info("No agent path provided. Running in Zero-Shot mode.")

    in_dir = Path(args.input)
    out_dir = Path(args.output) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_process = list(in_dir.glob("*.json"))
    logger.info(f"Found {len(files_to_process)} documents to process for task '{args.task}'.")

    # --- 2. Inference Loop ---
    for json_file in tqdm(files_to_process, desc=f"Predicting on '{args.task}'"):
        # ОПРЕДЕЛЯЕМ ID СРАЗУ, чтобы он был виден в except
        doc_id = json_file.stem 
        
        try:
            # Load and validate input document
            doc = ProcessedDocument.model_validate_json(json_file.read_text(encoding="utf-8"))
            
            if not doc.text_content:
                logger.warning(f"Skipping empty document: {json_file.name}")
                continue

            # Run inference
            # Используем doc.text_content, так как это содержимое файла
            prediction = agent(document_text=doc.text_content)

            # Проверка на None (если модель вернула мусор, который не распарсился)
            if prediction.extracted_data is None:
                raise ValueError("DSPy returned None for extracted_data (Model output validation failed)")
            
            # Prepare output data
            output_data = {
                "source_metadata": doc.metadata.model_dump(),
                "extraction": prediction.extracted_data.model_dump(mode='json'),
                "reasoning": prediction.reasoning
            }
            
            # Save results
            res_path = out_dir / f"{doc_id}_result.json"
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            save_dspy_log(doc_id)

        except json.JSONDecodeError:
            logger.error(f"Failed to parse source JSON file: {json_file.name}")
        except Exception as e:
            logger.error(f"An unexpected error occurred with {json_file.name}: {e}")
            # Теперь doc_id точно определен
            save_dspy_log(doc_id)
            continue

if __name__ == "__main__":
    main()