from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    root: Path

    @property
    def input(self) -> Path:
        return self.root / "input"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def service(self) -> Path:
        return self.root / "service"

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def input_pdf(self) -> Path:
        return self.input / "article.pdf"

    @property
    def input_task_config(self) -> Path:
        return self.input / "task.yaml"

    @property
    def input_markdown(self) -> Path:
        return self.input / "article.md"

    @property
    def manifest(self) -> Path:
        return self.work / "manifest.json"

    @property
    def md_anchor_index(self) -> Path:
        return self.work / "md_anchor_index.json"

    @property
    def pages(self) -> Path:
        return self.work / "pages.json"

    @property
    def targets(self) -> Path:
        return self.work / "targets"

    @property
    def bboxes(self) -> Path:
        return self.work / "bboxes"

    @property
    def crops(self) -> Path:
        return self.work / "crops"

    @property
    def extraction(self) -> Path:
        return self.work / "extraction"

    @property
    def page_assets(self) -> Path:
        return self.assets / "pages"

    @property
    def crop_assets(self) -> Path:
        return self.assets / "crops"

    @property
    def overlays(self) -> Path:
        return self.assets / "overlays"

    def service_dir(self, stage: str) -> Path:
        return self.service / stage


def build_paths(cfg: dict) -> PipelinePaths:
    return PipelinePaths(Path(cfg["run"]["out_dir"]))


def ensure_paths(paths: PipelinePaths) -> None:
    for path in (
        paths.input,
        paths.work,
        paths.service,
        paths.assets,
        paths.targets,
        paths.bboxes,
        paths.crops,
        paths.extraction,
        paths.page_assets,
        paths.crop_assets,
        paths.overlays,
    ):
        path.mkdir(parents=True, exist_ok=True)
