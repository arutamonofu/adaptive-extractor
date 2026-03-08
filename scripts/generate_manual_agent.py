"""Generate manual agent from train_manual split examples.

This script creates a manual agent by loading examples from the train_manual
split and saving them as few-shot demonstrations in a DSPy agent.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import dspy

from aee.domain.tasks import load_task_with_instruction
from aee.infrastructure.config.settings import Settings
from aee.infrastructure.llm import setup_student
from aee.infrastructure.storage import GroundTruthRepository

logger = logging.getLogger("aee")


def setup_language_models(config: Settings) -> None:
    """Setup student language model for manual agent generation.

    Args:
        config: Settings object containing LLM configuration.
    """
    setup_student(config, enable_cache=False)
    logger.info(f"Configured student LM: {config.llm.student.model}")


def main() -> int:
    """Main entry point for manual agent generation.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        parser = argparse.ArgumentParser(
            description="Create manual agent from train_manual split.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--config",
            type=Path,
            required=True,
            help="Path to configuration file (required)",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Override output path for agent JSON",
        )
        args = parser.parse_args()

        # Load settings from config file
        try:
            custom_settings = Settings.load(config_path=args.config)
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {args.config}")
            return 1
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            return 1

        # Setup logging with custom settings
        from aee import setup_logging

        setup_logging(custom_settings)

        # Setup LLM for agent generation
        setup_language_models(custom_settings)

        # Get task name from settings
        task_name = custom_settings.task.name
        logger.info(f"Starting manual agent generation for task '{task_name}'")

        # Load task definition with instruction from YAML config
        try:
            task, instruction_metadata = load_task_with_instruction(task_name, custom_settings)
            logger.info(f"Task loaded: {task['config'].name}")
        except FileNotFoundError as e:
            logger.error(f"Task configuration not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Task {task_name} not found: {e}")
            return 1

        # Validate task has signature for agent creation
        if task.get("signature") is None:
            logger.error("Task signature not found - required for manual agent creation")
            return 1

        # Check if splits file exists
        splits_path = custom_settings.paths.splits_file
        if not splits_path.exists():
            logger.error(f"Splits file not found at {splits_path}")
            return 1

        # Load data splits
        try:
            with open(splits_path, "r", encoding="utf-8") as f:
                splits = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in splits file: {e}")
            return 1

        # Get manual IDs from train_manual split
        manual_ids = splits.get("train_manual", [])
        if not manual_ids:
            logger.warning("No IDs found in 'train_manual' split.")
            return 1

        logger.info(f"Found {len(manual_ids)} manual IDs: {manual_ids}")

        # Load ground truth using row_converter from task dict
        gt_path = custom_settings.paths.ground_truth_dir / f"{task_name}.csv"
        gt_repo = GroundTruthRepository()

        try:
            gt_data = gt_repo.load(gt_path, task["row_converter"])
        except Exception as e:
            logger.error(f"Failed to load ground truth: {e}")
            return 1

        # Process each manual ID to collect demos
        manual_demos = []
        proc_dir = custom_settings.paths.parsed_dir
        OutputModel = task["output_model"]

        # Validate parsed_dir exists
        if not proc_dir.exists():
            logger.error(f"Parsed directory not found: {proc_dir}")
            return 1

        # Process each manual ID
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
                continue

        # Check if we collected any valid demos
        if not manual_demos:
            logger.error("No valid demos collected. Agent not saved.")
            return 1

        # Create agent with demos using AgentManager (consistent architecture)
        from aee.application.services import AgentManager
        from aee.infrastructure.storage import AgentRepository

        agent_repo = AgentRepository(agents_dir=custom_settings.paths.agents_dir)
        agent_manager = AgentManager(agent_repo=agent_repo)

        agent = agent_manager.create_agent_with_demos(
            signature_class=task["signature"],
            demos=manual_demos,
        )

        # Determine output path with path resolution
        default_output = custom_settings.paths.agents_dir / f"manual_{task_name}.json"
        output_path = args.output if args.output else default_output

        # Resolve relative path if needed
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

        # Ensure parent directory exists (consistent with extract.py and optimize.py)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Warn if overwriting existing agent
        if output_path.exists():
            logger.warning(f"Overwriting existing agent: {output_path}")

        agent.save(str(output_path))
        logger.info(f"Manual agent saved to: {output_path.absolute()}")

        return 0

    except KeyboardInterrupt:
        logger.warning("Manual agent generation interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
