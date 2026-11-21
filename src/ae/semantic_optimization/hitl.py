"""Human-in-the-Loop (HITL) module for Reverse Engineering.

Manages:
1. GTEditor: Safe editing and persistence of ground_truth_corrected.csv.
2. MarkdownEditor: Reading, inspecting unparsed visuals, and patching ingested .md documents.
3. RulingsStore: Storing and formatting human expert rulings (expert_rulings.json).
4. InteractiveConflictResolver: Terminal UI for resolving conflicts and ambiguity in-flight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd

from ae.core.storage.documents import DocumentRepository
from ae.core.utils import normalize_document_key
from ae.ingestion.parsers.gemini.stages.render_markdown import extract_unparsed_visual_ids
from ae.semantic_optimization.models import ExpertRuling, REConflict, ResolutionOption

logger = logging.getLogger(__name__)


@dataclass
class BatchResolutionResult:
    """Summary of resolving a batch of conflicts in the unified HITL digest."""

    rulings: List[ExpertRuling] = field(default_factory=list)
    recompute_scopes: Set[str] = field(default_factory=set)
    recompute_doc_ids: Set[str] = field(default_factory=set)

    @property
    def needs_doc_recompute(self) -> bool:
        """True if any document requires re-running 1A/1B (e.g. .md patched or new row added)."""
        return len(self.recompute_doc_ids) > 0

    @property
    def needs_rule_recompute(self) -> bool:
        """True if any scope (rows or column:<field>) needs rule synthesis recomputation."""
        return len(self.recompute_scopes) > 0


def _get_id_column(df: pd.DataFrame) -> str:
    """Identify the document ID column in Ground Truth DataFrame."""
    for col in ["pdf", "filename", "source", "doi", "document"]:
        if col in df.columns:
            return col
    # Fallback to the first column
    if len(df.columns) > 0:
        return df.columns[0]
    raise ValueError(f"No valid document ID column found in DataFrame with columns: {list(df.columns)}")


class GTEditor:
    """Manages ground_truth_corrected.csv with atomic edits and reload."""

    def __init__(self, source_gt_path: Path, corrected_gt_path: Path, resume: bool = False):
        self.source_gt_path = Path(source_gt_path)
        self.corrected_gt_path = Path(corrected_gt_path)
        self.corrected_gt_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.corrected_gt_path.exists() or not resume:
            if self.source_gt_path.exists():
                shutil.copyfile(self.source_gt_path, self.corrected_gt_path)
                logger.info(f"Initialized corrected GT from {self.source_gt_path} -> {self.corrected_gt_path}")
            else:
                self.corrected_gt_path.write_text("doi\n", encoding="utf-8")
                logger.info(f"Created fallback empty corrected GT at {self.corrected_gt_path}")
        else:
            logger.info(f"Resuming with existing corrected GT at {self.corrected_gt_path}")

        self._df = self._load_df()

    def _load_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.corrected_gt_path)
        df.columns = df.columns.str.lower().str.strip()
        return df

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _save_df(self) -> None:
        df_to_save = self._df.drop(columns=["_re_row_id"], errors="ignore")
        df_to_save.to_csv(self.corrected_gt_path, index=False)
        logger.info(f"Saved updated Ground Truth table to {self.corrected_gt_path}")

    def _ensure_doc_row_ids(self, norm_doc: str) -> None:
        """Lazily assign stable 1-based `_re_row_id` labels (`row_1`..) for `norm_doc`."""
        id_col = _get_id_column(self._df)
        if "_re_row_id" not in self._df.columns:
            self._df["_re_row_id"] = None
        doc_mask = self._df[id_col].astype(str).apply(normalize_document_key) == norm_doc
        doc_indices = self._df[doc_mask].index.tolist()
        if not doc_indices:
            return
        if self._df.loc[doc_mask, "_re_row_id"].isna().all():
            self._df.loc[doc_mask, "_re_row_id"] = [f"row_{i + 1}" for i in range(len(doc_indices))]

    def reindex_doc_rows(self, doc_id: Optional[str] = None) -> None:
        """Reset `_re_row_id` labels to sequential `row_1..row_N` after batch edits & recompute."""
        if "_re_row_id" not in self._df.columns:
            return
        if doc_id is None:
            self._df["_re_row_id"] = None
            return
        id_col = _get_id_column(self._df)
        norm_doc = normalize_document_key(doc_id)
        doc_mask = self._df[id_col].astype(str).apply(normalize_document_key) == norm_doc
        doc_indices = self._df[doc_mask].index.tolist()
        if doc_indices:
            self._df.loc[doc_mask, "_re_row_id"] = [f"row_{i + 1}" for i in range(len(doc_indices))]

    def _resolve_target_idx(self, doc_id: str, row_id_or_index: str | int) -> Optional[int]:
        """Resolve a DataFrame index for `doc_id` using stable `_re_row_id` first, then position."""
        id_col = _get_id_column(self._df)
        norm_doc = normalize_document_key(doc_id)
        self._ensure_doc_row_ids(norm_doc)

        doc_mask = self._df[id_col].astype(str).apply(normalize_document_key) == norm_doc
        doc_indices = self._df[doc_mask].index.tolist()
        if not doc_indices:
            return None

        if isinstance(row_id_or_index, int):
            if 0 <= row_id_or_index < len(doc_indices):
                return doc_indices[row_id_or_index]
            return None

        if isinstance(row_id_or_index, str):
            if row_id_or_index.startswith("row_") and "_re_row_id" in self._df.columns:
                doc_row_ids = self._df.loc[doc_mask, "_re_row_id"]
                if doc_row_ids.notna().any():
                    stable_matches = self._df[
                        doc_mask & (self._df["_re_row_id"] == row_id_or_index)
                    ].index.tolist()
                    return stable_matches[0] if stable_matches else None
            if row_id_or_index.startswith("row_"):
                try:
                    num = int(row_id_or_index.split("_")[1]) - 1
                    if 0 <= num < len(doc_indices):
                        return doc_indices[num]
                except ValueError:
                    pass
        return None

    def update_cell(self, doc_id: str, row_id_or_index: str | int, field_name: str, new_value: Any) -> bool:
        """Update a specific field for a given document and row."""
        id_col = _get_id_column(self._df)
        norm_doc = normalize_document_key(doc_id)
        field = field_name.lower().strip()
        if field not in self._df.columns:
            self._df[field] = None
        elif self._df[field].dtype != object:
            self._df[field] = self._df[field].astype(object)

        doc_mask = self._df[id_col].astype(str).apply(normalize_document_key) == norm_doc
        doc_indices = self._df[doc_mask].index.tolist()
        if not doc_indices:
            logger.warning(f"Document {doc_id} not found in GT table")
            return False

        target_idx = self._resolve_target_idx(doc_id, row_id_or_index)
        if target_idx is None:
            logger.warning(
                f"Target row {row_id_or_index} could not be resolved for document {doc_id} (valid indices: 0..{len(doc_indices)-1})"
            )
            return False

        self._df.at[target_idx, field] = new_value
        self._save_df()
        logger.info(f"Updated GT cell for {doc_id} {row_id_or_index} [{field}] -> {new_value}")
        return True

    def drop_row(self, doc_id: str, row_id_or_index: str | int) -> bool:
        """Drop a row for a given document."""
        id_col = _get_id_column(self._df)
        norm_doc = normalize_document_key(doc_id)
        doc_mask = self._df[id_col].astype(str).apply(normalize_document_key) == norm_doc
        doc_indices = self._df[doc_mask].index.tolist()
        if not doc_indices:
            return False

        target_idx = self._resolve_target_idx(doc_id, row_id_or_index)
        if target_idx is None:
            logger.warning(
                f"Target row {row_id_or_index} could not be resolved for document {doc_id} (valid indices: 0..{len(doc_indices)-1})"
            )
            return False

        self._df = self._df.drop(index=target_idx).reset_index(drop=True)
        self._save_df()
        logger.info(f"Dropped GT row {row_id_or_index} for document {doc_id}")
        return True

    def add_row(self, doc_id: str, row_dict: Dict[str, Any]) -> bool:
        """Append a newly identified valid row (e.g. from gap analysis) to GT."""
        id_col = _get_id_column(self._df)
        norm_doc = normalize_document_key(doc_id)
        self._ensure_doc_row_ids(norm_doc)

        clean_row = {k.lower().strip(): v for k, v in row_dict.items()}
        clean_row[id_col] = doc_id

        doc_mask = self._df[id_col].astype(str).apply(normalize_document_key) == norm_doc
        max_num = 0
        if "_re_row_id" in self._df.columns:
            for existing_id in self._df.loc[doc_mask, "_re_row_id"].dropna().astype(str):
                if existing_id.startswith("row_"):
                    try:
                        max_num = max(max_num, int(existing_id.split("_")[1]))
                    except ValueError:
                        pass
            clean_row["_re_row_id"] = f"row_{max_num + 1}"

        for col in clean_row.keys():
            if col not in self._df.columns:
                self._df[col] = None

        new_row_df = pd.DataFrame([clean_row])
        self._df = pd.concat([self._df, new_row_df], ignore_index=True)
        self._save_df()
        logger.info(f"Appended new GT row for document {doc_id}: {clean_row}")
        return True


class MarkdownEditor:
    """Manages reading, visual inspection, and in-place patching of ingested .md files."""

    def __init__(self, ingestion_dir: Path, pdf_dir: Optional[Path] = None):
        self.ingestion_dir = Path(ingestion_dir)
        self.pdf_dir = Path(pdf_dir) if pdf_dir else None
        self.doc_repo = DocumentRepository(ingestion_dir=self.ingestion_dir)

    def get_md_path(self, doc_id: str) -> Optional[Path]:
        norm_key = normalize_document_key(doc_id)
        return self.doc_repo._find_document_path(norm_key, self.ingestion_dir)

    def get_unparsed_visuals_summary(self, doc_id: str) -> list[str]:
        """Extract unparsed visual IDs from document text."""
        norm_key = normalize_document_key(doc_id)
        text = self.doc_repo.get(norm_key)
        if not text:
            return []
        return extract_unparsed_visual_ids(text)

    def _patch_file_with_backup(self, file_path: Path, old_value: str, new_value: str, doc_id: str) -> None:
        """Apply substring replacement or append patch block to file_path, backing up original as .md.bak."""
        bak_path = file_path.with_suffix(".md.bak")
        if not bak_path.exists():
            shutil.copyfile(file_path, bak_path)
            logger.info(f"Created backup of ingested markdown: {bak_path}")

        content = file_path.read_text(encoding="utf-8")
        header_match = re.match(r"^(<!--\s*document_id:[^>]*-->\s*\n*)", content)
        header = header_match.group(1) if header_match else ""
        body = content[len(header):]

        if old_value and len(old_value.strip()) >= 3 and old_value in body:
            updated = header + body.replace(old_value, new_value, 1)
        else:
            logger.warning(
                f"Target value '{old_value}' not found or too short in {doc_id} body ({file_path.name}). "
                "Appending patch block instead of in-place replace."
            )
            updated = content.rstrip() + f"\n\n<!-- Ingestion Patch: {new_value} -->\n{new_value}\n"

        file_path.write_text(updated, encoding="utf-8")

    def patch_markdown(self, doc_id: str, old_value: str, new_value: str) -> bool:
        """Replace old_value with new_value in doc.md (and Stage 02 manifest if present), backing up as .md.bak."""
        norm_key = normalize_document_key(doc_id)
        md_path = self.get_md_path(norm_key)
        if not md_path or not md_path.exists():
            logger.warning(f"Document markdown file not found for {doc_id}")
            return False

        self._patch_file_with_backup(md_path, old_value, new_value, doc_id)

        manifest_md_path = md_path.parent / "02_manifest_visuals" / "manifest_enriched_text.md"
        if manifest_md_path.is_file():
            self._patch_file_with_backup(manifest_md_path, old_value, new_value, doc_id)

        logger.info(f"Patched markdown for {doc_id}: replaced '{old_value}' -> '{new_value}'")
        return True

    def unpack_visual(self, doc_id: str, visual_ids: list[str]) -> bool:
        """Trigger on-demand visual parsing service for visual_ids."""
        norm_key = normalize_document_key(doc_id)
        try:
            from ae.core.config.settings import Settings
            from ae.ingestion.service import VisualParsingService
            from ae.ingestion.tool import make_visual_parsing_tool

            settings = Settings.load(load_env_file=True)
            service = VisualParsingService(
                config=settings.ingestion,
                pdf_dir=self.pdf_dir or settings.paths.pdf_dir,
                ingestion_dir=self.ingestion_dir,
            )
            logger.info(f"Unpacking visuals {visual_ids} for document {norm_key}...")
            tool = make_visual_parsing_tool(norm_key, service, mode="semantic_optimization")
            tool.invoke({"visual_ids": visual_ids})
            return True
        except Exception as e:
            logger.warning(f"Failed to unpack visual {visual_ids} for {doc_id}: {e}")
            return False


_TOPIC_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "into", "when", "where",
        "without", "while", "only", "each", "are", "not", "has", "have", "was", "were",
        "per", "all", "any", "value", "values", "rule", "rules", "baseline", "prompt",
        "ground", "truth", "text", "raw", "field", "document", "expert", "ruling",
        "policy", "conflict", "keep", "override", "authoritative", "ignore", "deviation",
        "conflicting", "shows", "requires", "mandates", "multiple", "rows", "across",
        "ignored", "ingestion", "case", "defect", "exception", "behavior", "empirical",
        "trust", "draft", "treat",
    }
)


def _is_descriptive_rule_text(text: Optional[str]) -> bool:
    """Return True if text looks like a descriptive rule rather than a short/numeric literal."""
    if not text:
        return False
    s = text.strip()
    if len(s) < 15:
        return False
    if re.fullmatch(r"[\d\s.,+\-/*%()\[\]=<>~±µ°]+", s):
        return False
    word_tokens = re.findall(r"[A-Za-z]{2,}", s)
    return len(word_tokens) >= 3


def _extract_topic_tokens(*texts: Optional[str]) -> set[str]:
    tokens: set[str] = set()
    for t in texts:
        if not t:
            continue
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            if len(tok) >= 3 and tok not in _TOPIC_STOPWORDS:
                tokens.add(tok)
    return tokens


def _conflict_topics_overlap(conflict: REConflict, ruling: ExpertRuling) -> bool:
    """Return True if a conflict and an existing ruling share the same rule topic."""
    c_tokens = _extract_topic_tokens(
        conflict.conflicting_value_or_rule,
        conflict.explanation,
        conflict.conflict_id,
    )
    r_tokens = _extract_topic_tokens(
        ruling.decision_text,
        ruling.conflict_id,
    )
    if not c_tokens or not r_tokens:
        return False
    shared = len(c_tokens & r_tokens)
    overlap_coeff = shared / min(len(c_tokens), len(r_tokens))
    return shared >= 2 and overlap_coeff >= 0.35


_baseline_conflict_topics_overlap = _conflict_topics_overlap


class RulingsStore:
    """Persists human expert decisions and ignored examples."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.rulings: Dict[str, ExpertRuling] = {}
        self.ignored_examples: Set[str] = set()  # set of f"{doc_id}:{row_id}:{field_name}"
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                for r_data in data.get("rulings", []):
                    r = ExpertRuling.model_validate(r_data)
                    self.rulings[r.conflict_id] = r
                self.ignored_examples = set(data.get("ignored_examples", []))
                logger.info(f"Loaded {len(self.rulings)} expert rulings from {self.storage_path}")
            except Exception as e:
                logger.warning(f"Failed to load expert rulings from {self.storage_path}: {e}")

    def save(self) -> None:
        data = {
            "rulings": [r.model_dump() for r in self.rulings.values()],
            "ignored_examples": sorted(list(self.ignored_examples)),
        }
        self.storage_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(self.rulings)} expert rulings to {self.storage_path}")

    def add_ruling(self, ruling: ExpertRuling) -> None:
        self.rulings[ruling.conflict_id] = ruling
        self.save()

    def has_ruling(self, conflict_id: str) -> bool:
        return conflict_id in self.rulings

    def get_ruling(self, conflict_id: str) -> Optional[ExpertRuling]:
        return self.rulings.get(conflict_id)

    def find_matching_ruling(self, conflict: REConflict) -> Optional[ExpertRuling]:
        """Find an existing ruling matching by exact conflict_id, semantic identity, or ignored status."""
        if conflict.conflict_id in self.rulings:
            return self.rulings[conflict.conflict_id]

        field_name = (
            conflict.scope.split("column:", 1)[1].lower().strip()
            if "column:" in conflict.scope
            else ""
        )
        norm_doc = normalize_document_key(conflict.document_id) if conflict.document_id else ""

        for ruling in self.rulings.values():
            r_doc = normalize_document_key(ruling.document_id) if ruling.document_id else ""
            if (
                ruling.scope == conflict.scope
                and ruling.action_type in ("KEEP_BASELINE", "OVERRIDE_BASELINE", "GENERALIZE", "ADD_CONDITION")
            ):
                if (
                    conflict.conflict_type == "BASELINE_POLICY_CONFLICT"
                    and ruling.conflict_type == "BASELINE_POLICY_CONFLICT"
                ):
                    if norm_doc and r_doc == norm_doc:
                        return ruling
                    if _conflict_topics_overlap(conflict, ruling):
                        return ruling
                    continue
                if norm_doc and r_doc == norm_doc and _conflict_topics_overlap(conflict, ruling):
                    return ruling

            if (
                norm_doc
                and r_doc == norm_doc
                and ruling.scope == conflict.scope
                and ruling.action_type in ("IGNORE_CASE", "IGNORE_INGESTION_DEFECT")
                and _conflict_topics_overlap(conflict, ruling)
            ):
                return ruling

            if (
                ruling.scope == conflict.scope
                and ruling.conflict_type == conflict.conflict_type
                and ((norm_doc and r_doc == norm_doc) or (not norm_doc and not r_doc and ruling.row_id == conflict.row_id and bool(ruling.row_id)))
                and (
                    ruling.row_id == conflict.row_id
                    or not conflict.row_id
                    or not ruling.row_id
                )
            ):
                return ruling

        if norm_doc and conflict.row_id and self.is_ignored(norm_doc, conflict.row_id, field_name):
            for ruling in self.rulings.values():
                r_doc = normalize_document_key(ruling.document_id) if ruling.document_id else ""
                r_field = (ruling.field_name or "").lower().strip()
                if (
                    r_doc == norm_doc
                    and ruling.row_id == conflict.row_id
                    and (not field_name or r_field == field_name)
                ):
                    return ruling
            return ExpertRuling(
                conflict_id=conflict.conflict_id,
                scope=conflict.scope,
                conflict_type=conflict.conflict_type,
                action_type="IGNORE_CASE",
                decision_text=f"Already ignored {norm_doc}:{conflict.row_id}:{field_name}",
                document_id=conflict.document_id,
                row_id=conflict.row_id,
                field_name=field_name,
            )

        return None

    def has_matching_ruling(self, conflict: REConflict) -> bool:
        """Return True if a ruling already covers this conflict by ID or semantic key."""
        return self.find_matching_ruling(conflict) is not None

    def mark_ignored(self, doc_id: str, row_id: str, field_name: str = "") -> None:
        key = f"{normalize_document_key(doc_id)}:{row_id}:{field_name.lower().strip()}"
        self.ignored_examples.add(key)
        self.save()

    def is_ignored(self, doc_id: str, row_id: str, field_name: str = "") -> bool:
        key = f"{normalize_document_key(doc_id)}:{row_id}:{field_name.lower().strip()}"
        wildcard_row_key = f"{normalize_document_key(doc_id)}:{row_id}:"
        return key in self.ignored_examples or wildcard_row_key in self.ignored_examples

    def format_rulings_for_prompt(self, scope: str, only_policy_rules: bool = False) -> str:
        """Format matching expert rulings as a clear bulleted list of constraints."""
        policy_actions = {"GENERALIZE", "ADD_CONDITION", "OVERRIDE_BASELINE", "KEEP_BASELINE"}
        relevant = [
            r for r in self.rulings.values()
            if (r.scope == scope or scope == "all")
            and (not only_policy_rules or r.action_type in policy_actions)
        ]
        if not relevant:
            return ""

        lines = ["EXPERT RULINGS AND AUTHORITATIVE DECISIONS (HARD CONSTRAINTS):"]
        for r in relevant:
            lines.append(f"- [{r.conflict_type} -> {r.action_type}] {r.decision_text}")
        return "\n".join(lines)


class InteractiveConflictResolver:
    """Interactively resolves RE conflicts in terminal (single or batch digest)."""

    def __init__(
        self,
        gt_editor: GTEditor,
        rulings_store: RulingsStore,
        md_editor: Optional[MarkdownEditor] = None,
        interactive: bool = True,
        input_fn: Optional[Callable[[str], str]] = None,
        artifacts: Optional[Any] = None,
    ):
        self.gt_editor = gt_editor
        self.rulings_store = rulings_store
        self.md_editor = md_editor
        self.interactive = interactive
        self.input_fn = input_fn or input
        self.artifacts = artifacts

    def patch_positive_cell_artifact(
        self, doc_id: str, row_id: str, field_name: str, new_value: Any
    ) -> bool:
        """In-place update of ground_truth in cached positive column JSON artifact to avoid full LLM recompute."""
        if self.artifacts is None:
            return False
        norm_doc = normalize_document_key(doc_id)
        col_path = self.artifacts.get_path("positive", "columns", f"{norm_doc}_{row_id}.json", iteration=None)
        if not col_path.exists():
            return False
        try:
            data = json.loads(col_path.read_text(encoding="utf-8"))
            updated = False
            field_lower = field_name.lower().strip()
            for f_entry in data.get("fields", []):
                if f_entry.get("field_name", "").lower().strip() == field_lower:
                    f_entry["ground_truth"] = str(new_value)
                    f_entry["Suspected_Ingestion_Or_GT_Issue"] = "None"
                    if str(f_entry.get("Transformation_description", "")).startswith("No valid transformation"):
                        f_entry["Transformation_description"] = "Direct extraction / Ground truth value corrected"
                    updated = True
            if updated:
                col_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info(f"Patched positive column artifact in-place: {col_path} [{field_name}] -> {new_value}")

            # Also synchronize negative column artifact to prevent contradictory candidate rules
            neg_col_path = self.artifacts.get_path("negative", "columns", f"{norm_doc}_{row_id}.json", iteration=None)
            if neg_col_path.exists():
                try:
                    neg_data = json.loads(neg_col_path.read_text(encoding="utf-8"))
                    neg_updated = False
                    for f_res in neg_data.get("fields", []):
                        if f_res.get("field_name", "").lower().strip() == field_lower:
                            f_res["candidates"] = [
                                c for c in f_res.get("candidates", [])
                                if str(c.get("candidate_value", "")).strip() != str(new_value).strip()
                            ]
                            for c in f_res["candidates"]:
                                c["ground_truth"] = str(new_value)
                            neg_updated = True
                    if neg_updated:
                        neg_col_path.write_text(json.dumps(neg_data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as neg_err:
                    logger.debug(f"Failed to patch negative column artifact in-place {neg_col_path}: {neg_err}")

            return updated
        except Exception as e:
            logger.warning(f"Failed to patch positive column artifact {col_path}: {e}")
            return False

    def patch_drop_row_artifact(self, doc_id: str, row_id: str) -> bool:
        """Mark dropped row as ignored and remove from positive row artifact in-place."""
        self.rulings_store.mark_ignored(doc_id, row_id, "")
        if self.artifacts is None:
            return False
        norm_doc = normalize_document_key(doc_id)
        row_path = self.artifacts.get_path("positive", "rows", f"{norm_doc}.json", iteration=None)
        if row_path.exists():
            try:
                data = json.loads(row_path.read_text(encoding="utf-8"))
                rows = [r for r in data.get("rows", []) if r.get("row_id") != row_id]
                data["rows"] = rows
                row_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to update positive row artifact {row_path}: {e}")
        return True

    def resolve_batch(self, conflicts: List[REConflict]) -> BatchResolutionResult:
        """Resolve a unified batch of conflicts across rows and all columns in a single digest session."""
        result = BatchResolutionResult()
        if not conflicts:
            return result

        # Deduplicate by conflict_id while preserving order
        seen_ids: Set[str] = set()
        unique_conflicts: List[REConflict] = []
        for c in conflicts:
            if c.conflict_id not in seen_ids:
                seen_ids.add(c.conflict_id)
                unique_conflicts.append(c)

        # Priority order: Baseline Prompt Conflicts first, then GT/Ingestion errors, then Cross-Doc
        priority_map = {
            "BASELINE_POLICY_CONFLICT": 0,
            "SUSPECTED_GT_ERROR": 1,
            "SUSPECTED_GT_OMISSION": 2,
            "TEXT_VS_GT_MISMATCH": 3,
            "SUSPECTED_INGESTION_DEFECT": 4,
            "CROSS_DOC_CONTRADICTION": 5,
            "CONDITIONAL_BRANCH_NEEDED": 6,
        }
        sorted_conflicts = sorted(
            unique_conflicts,
            key=lambda c: priority_map.get(c.conflict_type, 99),
        )

        pending = [c for c in sorted_conflicts if not self.rulings_store.has_matching_ruling(c)]
        if pending and self.interactive:
            row_count = sum(1 for c in pending if c.scope == "rows")
            col_count = len(pending) - row_count
            baseline_count = sum(1 for c in pending if c.conflict_type == "BASELINE_POLICY_CONFLICT")
            print("\n" + "#" * 70)
            print(
                f"📋 HITL CONFLICT DIGEST: {len(pending)} item(s) to review "
                f"(Rows: {row_count}, Columns: {col_count}, Baseline vs GT: {baseline_count})"
            )
            print("#" * 70)

        for idx, conflict in enumerate(sorted_conflicts, start=1):
            was_cached = self.rulings_store.has_matching_ruling(conflict)
            if not was_cached and self.interactive and len(pending) > 1:
                print(f"\n--- Digest Item [{idx}/{len(sorted_conflicts)}] ---")

            ruling, needs_doc_recompute = self.resolve(conflict)
            result.rulings.append(ruling)

            if not was_cached:
                # If the user made a new ruling, mark its scope for rule synthesis recomputation
                result.recompute_scopes.add(conflict.scope)
                if needs_doc_recompute and conflict.document_id:
                    result.recompute_doc_ids.add(normalize_document_key(conflict.document_id))

        return result

    def resolve(self, conflict: REConflict) -> tuple[ExpertRuling, bool]:
        """Resolves conflict.

        Returns:
            (ruling, needs_recompute_positive_flag).
        """
        existing_ruling = self.rulings_store.find_matching_ruling(conflict)
        if existing_ruling is not None:
            return existing_ruling, False

        # Check for unparsed visuals in document
        unparsed_visuals = []
        if self.md_editor:
            unparsed_visuals = self.md_editor.get_unparsed_visuals_summary(conflict.document_id)

        if not self.interactive:
            default_action = conflict.options[0].action_type if conflict.options else "GENERALIZE"
            default_text = conflict.options[0].proposed_rule_or_value if conflict.options else conflict.explanation
            ruling = ExpertRuling(
                conflict_id=conflict.conflict_id,
                scope=conflict.scope,
                conflict_type=conflict.conflict_type,
                action_type=default_action,
                decision_text=default_text,
                document_id=conflict.document_id,
                row_id=conflict.row_id,
                field_name=conflict.scope.split("column:", 1)[1] if "column:" in conflict.scope else "",
            )
            self.rulings_store.add_ruling(ruling)
            return ruling, False

        # Interactive terminal prompt
        print("\n" + "=" * 70)
        print(f"⚠️  CONFLICT DETECTED [{conflict.conflict_type}]")
        print(f"Scope: {conflict.scope} | Document: {conflict.document_id} | Row: {conflict.row_id}")
        print("-" * 70)
        print(f"Explanation: {conflict.explanation}")
        if conflict.raw_text_quote:
            print(f"Ingested Text Quote (.md): \"{conflict.raw_text_quote}\"")
        if conflict.gt_value:
            print(f"Ground Truth Value (GT):  \"{conflict.gt_value}\"")
        if conflict.conflicting_value_or_rule:
            print(f"Target of Conflict:        {conflict.conflicting_value_or_rule}")
        if unparsed_visuals:
            print(f"💡 Notice: Found unparsed visuals in {conflict.document_id}.md: {unparsed_visuals}")
        print("-" * 70)
        print("Choose an action:")

        options: list[ResolutionOption] = list(conflict.options)

        # 1. Option to patch GT cell (if error in GT)
        if not any(opt.action_type == "PATCH_GT_CELL" for opt in options) and conflict.scope.startswith("column:"):
            options.append(
                ResolutionOption(
                    action_type="PATCH_GT_CELL",
                    label="Patch Ground Truth Cell (GT is wrong, update GT with value from text)",
                    proposed_rule_or_value=conflict.conflicting_value_or_rule or "",
                )
            )

        # 2. Option to patch Ingested .md (if GT is correct and .md contains OCR/VLM hallucination)
        if not any(opt.action_type == "PATCH_INGESTED_MD" for opt in options) and self.md_editor:
            options.append(
                ResolutionOption(
                    action_type="PATCH_INGESTED_MD",
                    label=f"Patch Ingested Markdown (GT is correct, fix OCR/VLM error in {conflict.document_id}.md)",
                    proposed_rule_or_value=conflict.gt_value or "",
                )
            )

        # 3. Option to unpack visual in .md (if visual stubs exist)
        if unparsed_visuals and not any(opt.action_type == "UNPACK_VISUAL_IN_MD" for opt in options):
            options.append(
                ResolutionOption(
                    action_type="UNPACK_VISUAL_IN_MD",
                    label=f"Unpack Visual in {conflict.document_id}.md (data is in unparsed visual manifest)",
                    proposed_rule_or_value=unparsed_visuals[0] if len(unparsed_visuals) == 1 else "",
                )
            )

        # 4. Option to ignore ingestion defect (leave GT clean, skip rule synthesis)
        if not any(opt.action_type == "IGNORE_INGESTION_DEFECT" for opt in options):
            options.append(
                ResolutionOption(
                    action_type="IGNORE_INGESTION_DEFECT",
                    label="Ignore Ingestion Defect (GT is correct, but .md is corrupted; skip from rule synthesis)",
                    proposed_rule_or_value="Skip corrupted ingestion example",
                )
            )

        # 5. Drop GT row (only valid for concrete existing rows)
        if not any(opt.action_type == "PATCH_GT_DROP_ROW" for opt in options) and conflict.row_id.startswith("row_"):
            options.append(
                ResolutionOption(
                    action_type="PATCH_GT_DROP_ROW",
                    label="Drop Ground Truth Row (row should not exist in GT)",
                    proposed_rule_or_value="",
                )
            )

        # 6. Add Missing GT Row (if GT omission suspected)
        if conflict.conflict_type == "SUSPECTED_GT_OMISSION" and not any(opt.action_type == "PATCH_GT_ADD_ROW" for opt in options):
            options.append(
                ResolutionOption(
                    action_type="PATCH_GT_ADD_ROW",
                    label="Add Missing Row to Ground Truth (omission in GT confirmed)",
                    proposed_rule_or_value="",
                )
            )

        # 6. Baseline Policy Conflict options (Override vs Keep Baseline)
        if conflict.conflict_type == "BASELINE_POLICY_CONFLICT":
            explanation_rule = conflict.explanation if _is_descriptive_rule_text(conflict.explanation) else ""
            if not any(opt.action_type == "OVERRIDE_BASELINE" for opt in options):
                override_rule = (
                    conflict.gt_value
                    if _is_descriptive_rule_text(conflict.gt_value)
                    else explanation_rule
                )
                options.append(
                    ResolutionOption(
                        action_type="OVERRIDE_BASELINE",
                        label="Override Baseline Prompt (trust Ground Truth markup over baseline draft)",
                        proposed_rule_or_value=override_rule,
                    )
                )
            if not any(opt.action_type == "KEEP_BASELINE" for opt in options):
                keep_rule = (
                    conflict.conflicting_value_or_rule
                    if _is_descriptive_rule_text(conflict.conflicting_value_or_rule)
                    else explanation_rule
                )
                options.append(
                    ResolutionOption(
                        action_type="KEEP_BASELINE",
                        label="Keep Baseline Prompt Rule (trust baseline draft; treat GT deviation as exception)",
                        proposed_rule_or_value=keep_rule,
                    )
                )

        # 7. Custom rule
        options.append(
            ResolutionOption(
                action_type="ADD_CONDITION",
                label="Custom Rule (enter manual imperative instruction text)",
                proposed_rule_or_value="",
            )
        )

        for i, opt in enumerate(options, start=1):
            detail = f" -> {opt.proposed_rule_or_value}" if opt.proposed_rule_or_value else ""
            print(f"  [{i}] ({opt.action_type}) {opt.label}{detail}")

        needs_recompute_positive = False
        selected_opt: ResolutionOption
        while True:
            try:
                choice_str = self.input_fn(f"Select option [1-{len(options)}]: ").strip()
                choice_idx = int(choice_str) - 1
                if 0 <= choice_idx < len(options):
                    selected_opt = options[choice_idx]
                    break
                print(f"Invalid option. Enter a number between 1 and {len(options)}.")
            except (ValueError, IndexError):
                print(f"Please enter a valid number between 1 and {len(options)}.")

        decision_text = selected_opt.proposed_rule_or_value

        # Execute chosen action
        if selected_opt.action_type == "PATCH_GT_CELL":
            field_name = conflict.scope.split("column:", 1)[1] if "column:" in conflict.scope else ""
            proposed = (selected_opt.proposed_rule_or_value or "").strip()
            looks_like_prose = bool(
                proposed
                and (
                    len(proposed) > 40
                    or re.match(r"^(?:update|change|correct|replace|set|modify)\b", proposed, re.I)
                )
            )
            if not proposed or (self.interactive and looks_like_prose):
                prompt_msg = (
                    f"Proposed value looks like an instruction ({proposed!r}). "
                    f"Enter exact literal cell value for GT field '{field_name}': "
                    if looks_like_prose
                    else f"Enter corrected cell value for GT field '{field_name}': "
                )
                val_to_set = self.input_fn(prompt_msg).strip()
            else:
                val_to_set = proposed

            if val_to_set:
                self.gt_editor.update_cell(
                    doc_id=conflict.document_id,
                    row_id_or_index=conflict.row_id,
                    field_name=field_name,
                    new_value=val_to_set,
                )
                decision_text = f"Corrected GT [{field_name}] to '{val_to_set}'"
                # Patch artifact in-place if available; only require full doc recompute if in-place patch unavailable
                patched_in_place = self.patch_positive_cell_artifact(
                    conflict.document_id, conflict.row_id, field_name, val_to_set
                )
                needs_recompute_positive = not patched_in_place
            else:
                logger.warning(
                    f"Skipped PATCH_GT_CELL for {conflict.document_id}:{conflict.row_id}:{field_name}: empty value."
                )
                decision_text = f"Skipped empty GT patch for [{field_name}]"
                needs_recompute_positive = False

        elif selected_opt.action_type == "PATCH_INGESTED_MD":
            val_to_set = selected_opt.proposed_rule_or_value
            if not val_to_set:
                val_to_set = self.input_fn("Enter corrected value to place in .md: ").strip()

            old_val = ""
            if conflict.conflicting_value_or_rule and (
                not conflict.raw_text_quote or conflict.conflicting_value_or_rule in conflict.raw_text_quote
            ):
                old_val = conflict.conflicting_value_or_rule
            elif conflict.raw_text_quote:
                old_val = conflict.raw_text_quote
            else:
                old_val = conflict.conflicting_value_or_rule

            if self.interactive and old_val != val_to_set:
                user_old = self.input_fn(f"Exact substring to replace in .md [{old_val}]: ").strip()
                if user_old:
                    if len(user_old) < 4 and user_old.isdigit():
                        print(f"⚠️  Ignoring '{user_old}' (too short / looks like menu index).")
                    else:
                        old_val = user_old

            if self.md_editor:
                self.md_editor.patch_markdown(
                    doc_id=conflict.document_id,
                    old_value=old_val,
                    new_value=val_to_set,
                )
            decision_text = f"Patched {conflict.document_id}.md with '{val_to_set}'"
            needs_recompute_positive = True

        elif selected_opt.action_type == "UNPACK_VISUAL_IN_MD":
            vis_input = selected_opt.proposed_rule_or_value
            if not vis_input:
                avail_str = f" (available: {', '.join(unparsed_visuals)})" if unparsed_visuals else ""
                vis_input = self.input_fn(f"Enter visual ID(s) to unpack (e.g. Table_1){avail_str}: ").strip()
            vis_ids = [v.strip() for v in vis_input.split(",") if v.strip()]
            if not vis_ids and unparsed_visuals:
                vis_ids = [unparsed_visuals[0]]
            if self.md_editor and vis_ids:
                self.md_editor.unpack_visual(doc_id=conflict.document_id, visual_ids=vis_ids)
            decision_text = f"Unpacked visual(s) {vis_ids} in {conflict.document_id}.md"
            needs_recompute_positive = True

        elif selected_opt.action_type in ("IGNORE_INGESTION_DEFECT", "IGNORE_CASE"):
            field_name = conflict.scope.split("column:", 1)[1] if "column:" in conflict.scope else ""
            self.rulings_store.mark_ignored(conflict.document_id, conflict.row_id, field_name)
            decision_text = f"Ignored ingestion/GT case {conflict.document_id}:{conflict.row_id}:{field_name}"

        elif selected_opt.action_type == "PATCH_GT_DROP_ROW":
            self.gt_editor.drop_row(doc_id=conflict.document_id, row_id_or_index=conflict.row_id)
            decision_text = f"Dropped row {conflict.row_id} for doc {conflict.document_id}"
            id_col = _get_id_column(self.gt_editor.df)
            norm_doc = normalize_document_key(conflict.document_id)
            doc_rows_left = len(self.gt_editor.df[self.gt_editor.df[id_col].astype(str).apply(normalize_document_key) == norm_doc])
            if doc_rows_left == 0:
                patched_in_place = self.patch_drop_row_artifact(conflict.document_id, conflict.row_id)
                needs_recompute_positive = not patched_in_place
            else:
                needs_recompute_positive = True

        elif selected_opt.action_type == "PATCH_GT_ADD_ROW":
            row_data = {}
            if selected_opt.proposed_rule_or_value:
                try:
                    row_data = json.loads(selected_opt.proposed_rule_or_value)
                except Exception:
                    pass
            if not row_data and self.interactive:
                entered = self.input_fn("Enter row data as JSON (e.g. {\"field\": \"val\"}): ").strip()
                if entered:
                    try:
                        row_data = json.loads(entered)
                    except Exception as e:
                        print(f"Warning: could not parse row JSON '{entered}': {e}")
            if row_data:
                self.gt_editor.add_row(doc_id=conflict.document_id, row_dict=row_data)
                decision_text = f"Added valid row from gap for doc {conflict.document_id}: {row_data}"
            else:
                decision_text = f"Added valid row from gap for doc {conflict.document_id}"
            needs_recompute_positive = True

        elif selected_opt.action_type == "OVERRIDE_BASELINE":
            rule_detail = (
                selected_opt.proposed_rule_or_value
                if _is_descriptive_rule_text(selected_opt.proposed_rule_or_value)
                else (
                    conflict.explanation
                    if _is_descriptive_rule_text(conflict.explanation)
                    else selected_opt.label
                )
            )
            decision_text = (
                f"Override baseline prompt rule with empirical Ground Truth behavior: {rule_detail}"
                if rule_detail
                else "Override baseline prompt rule with empirical Ground Truth behavior"
            )

        elif selected_opt.action_type == "KEEP_BASELINE":
            field_name = conflict.scope.split("column:", 1)[1] if "column:" in conflict.scope else ""
            if conflict.document_id and conflict.row_id:
                self.rulings_store.mark_ignored(conflict.document_id, conflict.row_id, field_name)
            rule_detail = (
                selected_opt.proposed_rule_or_value
                if _is_descriptive_rule_text(selected_opt.proposed_rule_or_value)
                else (
                    conflict.explanation
                    if _is_descriptive_rule_text(conflict.explanation)
                    else selected_opt.label
                )
            )
            decision_text = (
                f"Keep authoritative baseline prompt rule (ignore conflicting GT deviation): {rule_detail}"
                if rule_detail
                else "Keep authoritative baseline prompt rule (ignore conflicting GT deviation)"
            )

        elif selected_opt.action_type == "ADD_CONDITION" and not selected_opt.proposed_rule_or_value:
            custom_rule = self.input_fn("Enter custom rule instruction text: ").strip()
            decision_text = custom_rule or "Custom conditional rule applied"

        ruling = ExpertRuling(
            conflict_id=conflict.conflict_id,
            scope=conflict.scope,
            conflict_type=conflict.conflict_type,
            action_type=selected_opt.action_type,
            decision_text=decision_text,
            document_id=conflict.document_id,
            row_id=conflict.row_id,
            field_name=conflict.scope.split("column:", 1)[1] if "column:" in conflict.scope else "",
        )
        self.rulings_store.add_ruling(ruling)
        print("✓ Decision saved.")
        print("=" * 70 + "\n")
        return ruling, needs_recompute_positive
