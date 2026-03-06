import json
import argparse
from pathlib import Path
import dspy

from aee import settings, setup_logging
from aee.infrastructure.agents import UniversalExtractor
from aee.domain.tasks import get_task
from aee.infrastructure.storage import GroundTruthRepository

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Create a manual agent from train_manual split.")
    parser.add_argument("--output", type=str, help="Override output path.")
    args = parser.parse_args()

    task_name = settings.task.name

    # Get task definition using new task system
    try:
        task = get_task(task_name)
    except Exception as e:
        logger.error(f"Task {task_name} not found: {e}")
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

    # Load ground truth using new repository
    gt_path = settings.paths.ground_truth_dir / f"{task_name}.csv"
    gt_repo = GroundTruthRepository()
    gt_data = gt_repo.load(gt_path, task.row_converter)

    # Use task properties
    agent = UniversalExtractor(task.signature)
    manual_demos = []
    proc_dir = settings.paths.parsed_dir / "train" / "manual"
    OutputModel = task.output_model

    for doc_id in manual_ids:
        key = doc_id.lower()
        md_path = proc_dir / f"{doc_id}.md"

        if not md_path.exists():
            logger.warning(f"File {md_path} not found, skipping.")
            continue

        if key not in gt_data:
            logger.warning(f"ID {key} not found in Ground Truth, skipping.")
            continue

        try:
            text = md_path.read_text(encoding="utf-8")

            example = dspy.Example(
                document_text=text,
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