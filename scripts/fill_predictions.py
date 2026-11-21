"""Populate ground truth predicted columns from extracted JSON files.

Matches extracted records from data/processed/extracted/*.json to Ground Truth rows
using RecordMatcher (Hungarian algorithm) and canonicalizes SMILES with rdkit.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel
from rdkit import Chem

from ae.core.evaluation.matcher import RecordMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COMPARE_FIELDS = [
    "compound_id",
    "target_type",
    "target_relation",
    "target_value",
    "target_units",
    "bacteria",
]


class Entity(BaseModel):
    pdf: Optional[str] = None
    compound_id: Optional[str] = None
    smiles: Optional[str] = None
    target_type: Optional[str] = None
    target_relation: Optional[str] = None
    target_value: Optional[str] = None
    target_units: Optional[str] = None
    bacteria: Optional[str] = None
    bacteria_unified: Optional[str] = None
    original_idx: Optional[int] = None


def canonicalize_smiles(smiles: Optional[str]) -> str:
    """Canonicalize SMILES string without chirality."""
    if not smiles:
        return ""
    try:
        return Chem.CanonSmiles(str(smiles).strip(), useChiral=0)
    except Exception:
        return ""


def fill_predicted_csv(
    csv_path: Path,
    extracted_dir: Path,
    output_path: Optional[Path] = None,
) -> None:
    if output_path is None:
        output_path = csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    json_files = sorted(list(extracted_dir.glob("*.json")))

    if not json_files:
        logger.warning(f"No JSON files found in {extracted_dir}")
        return

    logger.info(f"Found {len(json_files)} extracted JSON file(s) in {extracted_dir}")

    matcher = RecordMatcher(
        fields_to_compare=COMPARE_FIELDS,
        float_tolerance=0.05,
        enable_semantic_judge=False,
    )

    unmatched_preds_to_append: list[Entity] = []

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        doc_id = data.get("source_metadata", {}).get("document_id")
        if not doc_id:
            doc_id = jf.stem

        extracted_records = data.get("extraction", {}).get("records", [])
        logger.info(f"Processing doc '{doc_id}' with {len(extracted_records)} record(s)")

        preds = [
            Entity(
                pdf=doc_id,
                compound_id=str(r.get("compound_id")) if r.get("compound_id") is not None else None,
                smiles=str(r.get("smiles")) if r.get("smiles") is not None else None,
                target_type=str(r.get("target_type")) if r.get("target_type") is not None else None,
                target_relation=str(r.get("target_relation")) if r.get("target_relation") is not None else None,
                target_value=str(r.get("target_value")) if r.get("target_value") is not None else None,
                target_units=str(r.get("target_units")) if r.get("target_units") is not None else None,
                bacteria=str(r.get("bacteria")) if r.get("bacteria") is not None else None,
                bacteria_unified=str(r.get("bacteria_unified")) if r.get("bacteria_unified") is not None else None,
                original_idx=idx,
            )
            for idx, r in enumerate(extracted_records)
        ]

        gt_sub = df[df["pdf"] == doc_id]
        gts = [
            Entity(
                pdf=str(row["pdf"]) if row["pdf"] else None,
                compound_id=str(row["compound_id"]) if row["compound_id"] else None,
                smiles=str(row["smiles"]) if row["smiles"] else None,
                target_type=str(row["target_type"]) if row["target_type"] else None,
                target_relation=str(row["target_relation"]) if row["target_relation"] else None,
                target_value=str(row["target_value"]) if row["target_value"] else None,
                target_units=str(row["target_units"]) if row["target_units"] else None,
                bacteria=str(row["bacteria"]) if row["bacteria"] else None,
                bacteria_unified=str(row["bacteria_unified"]) if row["bacteria_unified"] else None,
                original_idx=idx,
            )
            for idx, row in gt_sub.iterrows()
        ]

        pairs = matcher.align_pairs(preds, gts)

        for p, g in pairs:
            if p and g:
                matches = sum(
                    1
                    for f in COMPARE_FIELDS
                    if matcher._is_match(getattr(p, f, None), getattr(g, f, None))
                )
                score = matches / len(COMPARE_FIELDS)
                cid_match = (
                    p.compound_id is not None
                    and g.compound_id is not None
                    and p.compound_id.strip().lower() == g.compound_id.strip().lower()
                )

                if score > 0 or cid_match:
                    row_idx = g.original_idx
                    assert row_idx is not None
                    df.at[row_idx, "pdf_predicted"] = p.pdf or ""
                    df.at[row_idx, "smiles_predicted"] = p.smiles or ""
                    df.at[row_idx, "smiles_canonical_predicted"] = canonicalize_smiles(p.smiles)
                    df.at[row_idx, "compound_id_predicted"] = p.compound_id or ""
                    df.at[row_idx, "target_type_predicted"] = p.target_type or ""
                    df.at[row_idx, "target_relation_predicted"] = p.target_relation or ""
                    df.at[row_idx, "target_value_predicted"] = p.target_value or ""
                    df.at[row_idx, "target_units_predicted"] = p.target_units or ""
                    df.at[row_idx, "bacteria_predicted"] = p.bacteria or ""
                    df.at[row_idx, "bacteria_unified_predicted"] = p.bacteria_unified or ""
                else:
                    unmatched_preds_to_append.append(p)
            elif p and not g:
                unmatched_preds_to_append.append(p)

    if unmatched_preds_to_append:
        logger.info(f"Appending {len(unmatched_preds_to_append)} unmatched prediction(s) as new row(s)...")
        fp_rows = [
            {
                "pdf": "",
                "pdf_predicted": p.pdf or "",
                "smiles": "",
                "smiles_predicted": p.smiles or "",
                "smiles_canonical": "",
                "smiles_canonical_predicted": canonicalize_smiles(p.smiles),
                "compound_id": "",
                "compound_id_predicted": p.compound_id or "",
                "target_type": "",
                "target_type_predicted": p.target_type or "",
                "target_relation": "",
                "target_relation_predicted": p.target_relation or "",
                "target_value": "",
                "target_value_predicted": p.target_value or "",
                "target_units": "",
                "target_units_predicted": p.target_units or "",
                "bacteria": "",
                "bacteria_predicted": p.bacteria or "",
                "bacteria_unified": "",
                "bacteria_unified_predicted": p.bacteria_unified or "",
            }
            for p in unmatched_preds_to_append
        ]
        df = pd.concat([df, pd.DataFrame(fp_rows)], ignore_index=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved populated CSV to '{output_path}'. Total rows: {len(df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill GT CSV with extracted predictions.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/ground_truth/Benzimidazole antibiotics_predicted.csv"),
        help="Path to the GT predicted CSV file",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("data/processed/extracted"),
        help="Directory containing extracted JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to output CSV (defaults to overwriting --csv)",
    )
    args = parser.parse_args()
    fill_predicted_csv(args.csv, args.extracted_dir, args.output)


if __name__ == "__main__":
    main()
