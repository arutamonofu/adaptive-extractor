from __future__ import annotations

import argparse
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def _resolve(path: str | Path | None, base: Path = PROJECT_ROOT) -> Path | None:
    if path is None or path == "":
        return None
    value = Path(path)
    return value if value.is_absolute() else base / value


def _set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value not in (None, ""):
        target[key] = value


def _normalize_model(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": section.get("provider") or os.environ.get("VISUAL_MODEL_PROVIDER"),
        "model": section.get("model"),
        "temperature": float(section.get("temperature", 0.0) or 0.0),
        "max_output_tokens": section.get("max_output_tokens"),
        "thinking_level": section.get("thinking_level"),
        "thinking_budget": section.get("thinking_budget"),
    }


def load_config(config_path_or_dict: Path | dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(config_path_or_dict, dict):
        raw = config_path_or_dict
    else:
        raw = _read_yaml(_resolve(config_path_or_dict) or config_path_or_dict)
    cfg = deepcopy(raw)
    overrides = overrides or {}
    cfg.setdefault("inputs", {})
    cfg.setdefault("run", {})
    cfg.setdefault("models", {})
    cfg.setdefault("runtime", {})
    _set_if_present(cfg["inputs"], "pdf", overrides.get("pdf"))
    _set_if_present(cfg["inputs"], "markdown", overrides.get("markdown"))
    _set_if_present(cfg["inputs"], "task_config", overrides.get("task_config"))
    _set_if_present(cfg["run"], "out_dir", overrides.get("out_dir"))
    if overrides.get("force") is not None:
        cfg["run"]["force"] = bool(overrides["force"])
    _set_if_present(cfg["run"], "target_id", overrides.get("target_id"))
    _set_if_present(cfg["run"], "max_targets", overrides.get("max_targets"))
    pdf = _resolve(cfg["inputs"].get("pdf"))
    markdown = _resolve(cfg["inputs"].get("markdown"))
    task_config = _resolve(cfg["inputs"].get("task_config"))
    out_dir = _resolve(cfg["run"].get("out_dir"))
    if pdf is None or task_config is None or out_dir is None:
        raise ValueError("Config requires inputs.pdf, inputs.task_config, and run.out_dir")
    cfg["inputs"]["pdf"] = pdf
    cfg["inputs"]["markdown"] = markdown
    cfg["inputs"]["task_config"] = task_config
    cfg["run"]["out_dir"] = out_dir
    cfg["run"]["force"] = bool(cfg["run"].get("force", False))
    if cfg["run"].get("max_targets") not in (None, ""):
        cfg["run"]["max_targets"] = int(cfg["run"]["max_targets"])
    else:
        cfg["run"]["max_targets"] = None
    models = cfg["models"]
    for key in ("manifest", "bbox", "chart_extraction"):
        models[key] = _normalize_model(models.get(key, {}) or {})
    runtime = cfg["runtime"]
    runtime["dpi"] = int(runtime.get("dpi", 200) or 200)
    runtime["margin_ratio"] = float(runtime.get("margin_ratio", 0.02) or 0.02)
    if runtime.get("bbox_margin_px") not in (None, ""):
        runtime["bbox_margin_px"] = int(runtime["bbox_margin_px"])
    else:
        runtime["bbox_margin_px"] = None
    if runtime.get("gemini_retries") not in (None, ""):
        os.environ.setdefault("GEMINI_RETRIES", str(runtime["gemini_retries"]))
    if runtime.get("gemini_retry_base_seconds") not in (None, ""):
        os.environ.setdefault("GEMINI_RETRY_BASE_SECONDS", str(runtime["gemini_retry_base_seconds"]))
    return cfg


def model_options(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    model_cfg = cfg["models"][name]
    return {
        "model": model_cfg.get("model"),
        "provider": model_cfg.get("provider"),
        "temperature": model_cfg.get("temperature", 0.0),
        "max_output_tokens": model_cfg.get("max_output_tokens"),
        "thinking_level": model_cfg.get("thinking_level"),
        "thinking_budget": model_cfg.get("thinking_budget"),
    }


def prepare_inputs(cfg: dict[str, Any], paths: Any) -> None:
    paths.input.mkdir(parents=True, exist_ok=True)
    pairs = [
        (cfg["inputs"]["pdf"], paths.input_pdf),
        (cfg["inputs"]["task_config"], paths.input_task_config),
    ]
    if cfg["inputs"].get("markdown") is not None:
        pairs.append((cfg["inputs"]["markdown"], paths.input_markdown))
    for src, dst in pairs:
        if src.resolve() == dst.resolve():
            continue
        if dst.exists() and not cfg["run"]["force"]:
            continue
        shutil.copy2(src, dst)


def parse_config(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--task-config", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--force", action="store_true", default=None)
    parser.add_argument("--target-id")
    parser.add_argument("--max-targets", type=int)
    args = parser.parse_args(argv)
    return load_config(
        args.config,
        {
            "pdf": args.pdf,
            "markdown": args.markdown,
            "task_config": args.task_config,
            "out_dir": args.out_dir,
            "force": args.force,
            "target_id": args.target_id,
            "max_targets": args.max_targets,
        },
    )


def build_schema(task_config: dict[str, Any]) -> list[dict[str, Any]]:
    fields_in = task_config.get("fields") or {}
    fields: list[dict[str, Any]] = []
    if isinstance(fields_in, dict):
        for name, spec in fields_in.items():
            spec = spec or {}
            fields.append(
                {
                    "name": str(name),
                    "type": spec.get("type"),
                    "description": spec.get("description"),
                    "required": bool(spec.get("required", False)),
                    "choices": spec.get("choices"),
                }
            )
    elif isinstance(fields_in, list):
        for item in fields_in:
            if isinstance(item, dict) and item.get("name"):
                fields.append(
                    {
                        "name": str(item.get("name")),
                        "type": item.get("type"),
                        "description": item.get("description"),
                        "required": bool(item.get("required", False)),
                        "choices": item.get("choices"),
                    }
                )

    return fields
