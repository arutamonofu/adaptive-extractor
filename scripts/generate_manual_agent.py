import json
import argparse
from pathlib import Path
import dspy

from aee import settings, setup_logging, ProcessedDocument, UniversalExtractor
from aee.tasks import TASK_REGISTRY
from aee.utils import load_ground_truth

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Create a manual agent from train_manual split.")
    parser.add_argument("--output", type=str, help="Override output path.")
    args = parser.parse_args()

    task_name = settings.task.name
    task_conf = TASK_REGISTRY.get(task_name)
    if not task_conf:
        logger.error(f"Task {task_name} not found in registry.")
        return

    splits_path = settings.paths.splits_file
    if not splits_path.exists():
        logger.error(f"Splits file not found at {splits_path}")
        return

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    
    manual_ids = splits.get("train_manual", [])
    if not manual_ids:
        logger.warning("No IDs found in 'train_manual' split.")
        return
    
    logger.info(f"Found {len(manual_ids)} manual IDs: {manual_ids}")

    gt_path = settings.paths.ground_truth_dir / f"{task_name}.csv"
    gt_data = load_ground_truth(gt_path, task_conf["row_converter"])

    agent = UniversalExtractor(task_conf["signature"])
    manual_demos = []
    proc_dir = settings.paths.parsed_dir / "train" / "manual"
    OutputModel = task_conf["output_model"]

    for doc_id in manual_ids:
        key = doc_id.lower()
        json_path = proc_dir / f"{doc_id}.json"
        
        if not json_path.exists():
            logger.warning(f"File {json_path} not found, skipping.")
            continue
            
        if key not in gt_data:
            logger.warning(f"ID {key} not found in Ground Truth, skipping.")
            continue

        try:
            doc = ProcessedDocument.model_validate_json(json_path.read_text(encoding="utf-8"))
            
            example = dspy.Example(
                document_text=doc.text_content,
                extracted_data=OutputModel(experiments=gt_data[key])
            ).with_inputs("document_text")
            
            manual_demos.append(example)
            logger.info(f"Added demo: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to process {doc_id}: {e}")

    if not manual_demos:
        logger.error("No valid demos collected. Agent not saved.")
        return

    agent.prog.predict.demos = manual_demos
    
    default_output = settings.paths.agents_dir / f"manual_{task_name}.json"
    output_path = Path(args.output) if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    agent.save(str(output_path))
    logger.info(f"Manual agent saved to: {output_path.absolute()}")

if __name__ == "__main__":
    main()