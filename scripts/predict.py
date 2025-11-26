# scripts/predict.py

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from aee.core.logging import setup_logging
from aee.core.types import ProcessedDocument
from aee.llm import setup_student
from aee.agents.extractor import UniversalExtractor
from aee.tasks import TASK_REGISTRY

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Run Batch Inference.")
    parser.add_argument("--task", type=str, default="nanozymes")
    parser.add_argument("--input", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="data/results")
    parser.add_argument("--agent_path", type=str, help="Path to optimized JSON")
    args = parser.parse_args()

    task_conf = TASK_REGISTRY.get(args.task)
    if not task_conf:
        logger.error(f"Unknown task: {args.task}")
        return

    # Setup
    setup_student() # Init LLM
    agent = UniversalExtractor(task_conf["signature"])
    
    if args.agent_path and Path(args.agent_path).exists():
        agent.load(args.agent_path)
        logger.info(f"Loaded optimized agent from {args.agent_path}")
    else:
        logger.info("Running in Zero-Shot mode.")

    in_dir = Path(args.input)
    out_dir = Path(args.output) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(in_dir.glob("*.json"))
    
    for json_file in tqdm(files, desc="Predicting"):
        try:
            with open(json_file, "r") as f:
                doc = ProcessedDocument.model_validate_json(f.read())

            if not doc.text_content:
                continue

            # Inference
            prediction = agent(document_text=doc.text_content)
            
            # Save
            res_path = out_dir / f"{json_file.stem}_result.json"
            output_data = {
                "source": doc.metadata.filename,
                "extraction": prediction.extracted_data.model_dump(mode='json'),
                "reasoning": prediction.reasoning
            }
            
            with open(res_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed {json_file.name}: {e}")

if __name__ == "__main__":
    main()