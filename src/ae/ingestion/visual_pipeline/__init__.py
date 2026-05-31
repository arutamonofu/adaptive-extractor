from __future__ import annotations

import logging
from pathlib import Path

import dspy

from .config import load_config, prepare_inputs
from .paths import build_paths, ensure_paths
from .stages import (
    build_report,
    create_targets,
    crop_figures,
    extract_chart_tables,
    insert_visual_tables,
    locate_bboxes,
    manifest,
    md_anchors,
    render_pages,
)

logger = logging.getLogger(__name__)


def run_visual_pipeline(
    initial_md: str,
    pdf_path: Path,
    out_dir: Path,
    pipeline_config: Path | dict[str, Any],
    task_config_path: Path,
    lm: dspy.LM | None = None,
    force: bool = False,
) -> str:
    """Run the visual extraction pipeline in-process.

    Args:
        initial_md: Initial markdown content extracted by GeminiParser.
        pdf_path: Path to the input PDF file.
        out_dir: Output directory for the pipeline work artifacts.
        pipeline_config: Path or dictionary configuration for pipeline.
        task_config_path: Path to task YAML configuration (defines extraction schema).
        lm: The core LLM client/provider to use for multimodal queries.
        force: Force execution of all stages (bypass caching).

    Returns:
        str: Enriched markdown content containing extracted tables/charts.
    """
    logger.info("[run_visual_pipeline] Preparing pipeline configuration and inputs...")

    # 1. Build and load the pipeline configuration dictionary
    overrides = {
        "pdf": pdf_path,
        "task_config": task_config_path,
        "out_dir": out_dir,
        "force": force,
    }
    cfg = load_config(pipeline_config, overrides)
    paths = build_paths(cfg)

    # Ensure all directories exist and copy input files
    ensure_paths(paths)

    # Write the initial markdown to paths.input_markdown so stages can read/manipulate it
    paths.input_markdown.parent.mkdir(parents=True, exist_ok=True)
    paths.input_markdown.write_text(initial_md, encoding="utf-8")

    prepare_inputs(cfg, paths)

    # 2. Run each stage in sequence, passing the core LM provider where needed
    logger.info("[run_visual_pipeline] Stage 1/9: Running manifest stage...")
    manifest.run(cfg, paths, lm=lm)

    logger.info("[run_visual_pipeline] Stage 2/9: Running md_anchors stage...")
    md_anchors.run(cfg, paths)

    logger.info("[run_visual_pipeline] Stage 3/9: Running create_targets stage...")
    create_targets.run(cfg, paths)

    logger.info("[run_visual_pipeline] Stage 4/9: Running render_pages stage...")
    render_pages.run(cfg, paths)

    logger.info("[run_visual_pipeline] Stage 5/9: Running locate_bboxes stage...")
    locate_bboxes.run(cfg, paths, lm=lm)

    logger.info("[run_visual_pipeline] Stage 6/9: Running crop_figures stage...")
    crop_figures.run(cfg, paths)

    logger.info("[run_visual_pipeline] Stage 7/9: Running extract_chart_tables stage...")
    extract_chart_tables.run(cfg, paths, lm=lm)

    logger.info("[run_visual_pipeline] Stage 8/9: Running insert_visual_tables stage...")
    # This stage updates the markdown and returns the paths to the updated files
    inserted_paths = insert_visual_tables.run(cfg, paths)
    enriched_md_path = inserted_paths[0]

    logger.info("[run_visual_pipeline] Stage 9/9: Running build_report stage...")
    build_report.run(cfg, paths)

    logger.info("[run_visual_pipeline] Pipeline execution finished successfully.")

    # 3. Read and return the enriched markdown content
    return enriched_md_path.read_text(encoding="utf-8")
