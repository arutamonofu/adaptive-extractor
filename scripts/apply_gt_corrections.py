"""Apply verified Ground Truth corrections to Benzimidazole antibiotics dataset.

Reads the original ground truth CSV and exported Val/Train extracted CSVs,
applies confirmed DELETE, UPDATE, and ADD operations from the 'Расхождения'
review sheets, validates all SMILES with RDKit, and writes both the local
corrected CSV and a Google-Sheets-ready CSV with canonical SMILES.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rdkit import Chem

LOCAL_COLUMNS = [
    "pdf",
    "smiles",
    "compound_id",
    "target_type",
    "target_relation",
    "target_value",
    "target_units",
    "bacteria",
    "bacteria_unified",
]

GS_COLUMNS = [
    "pdf",
    "smiles",
    "smiles_canonical",
    "compound_id",
    "target_type",
    "target_relation",
    "target_value",
    "target_units",
    "bacteria",
]

# Confirmed DELETE rules: (pdf_lower, compound_id, optional_bacteria_exact)
DELETE_RULES: List[Tuple[str, str, str | None]] = [
    ("rjc.2023.1638382", "3", "E.coli"),
    ("s41598-022-21435-6", "7h", None),
    ("j.molstruc.2021.130566", "20", None),
    ("j.molstruc.2021.130566", "21", None),
    ("j.molstruc.2021.130566", "39", None),
    ("antibiotics12071220", "95", None),
    ("antibiotics12071220", "96", None),
    ("antibiotics12071220", "97", None),
]

# Confirmed SMILES updates: (pdf_lower, compound_id) -> new_smiles
SMILES_UPDATES: Dict[Tuple[str, str], str] = {
    ("acsomega.2c06142", "5b"): (
        "C(C(C)C)NC1SC(C2C=CC(C3NC4C=C(C(=CC=4N=3)C)C)=CC=2)=NN=1"
    ),
    ("rjc.2023.1638382", "34"): (
        "CC1=NN(c2ccc(Cl)cc2)C(=O)C1=NNc1ccc(-c2ccc(-c3nc4ccccc4[nH]3)cc2)cc1"
    ),
    ("rjc.2023.1638382", "35"): (
        "CC1=NN(c2cccc(Cl)c2)C(=O)C1=NNc1ccc(-c2ccc(-c3nc4ccccc4[nH]3)cc2)cc1"
    ),
    ("antibiotics10081002", "31a"): (
        "Clc1ccc2[nH]c(-c3ccccc3)c(C3CC(c4nc5ccccc5[nH]4)=NN3)c2c1"
    ),
    ("antibiotics10081002", "31b"): (
        "Cc1ccc2[nH]c(-c3ccccc3)c(C3CC(c4nc5ccccc5[nH]4)=NN3)c2c1"
    ),
    ("antibiotics10081002", "31c"): (
        "c1ccc(-c2[nH]c3ccccc3c2C2CC(c3nc4ccccc4[nH]3)=NN2)cc1"
    ),
    ("antibiotics10081002", "32a"): (
        "Clc1ccc2[nH]c(-c3ccccc3)c(C3CC(c4nc5ccccc5[nH]4)=NN3c3ccccc3)c2c1"
    ),
    ("antibiotics10081002", "32b"): (
        "Cc1ccc2[nH]c(-c3ccccc3)c(C3CC(c4nc5ccccc5[nH]4)=NN3c3ccccc3)c2c1"
    ),
    ("antibiotics10081002", "32c"): (
        "c1ccc(-c2[nH]c3ccccc3c2C2CC(c3nc4ccccc4[nH]3)=NN2c2ccccc2)cc1"
    ),
    ("antibiotics10081002", "61a"): (
        "Cc1ccc(-n2nc(C)c(-c3c(C#N)c(N)n4c(nc5cc(C)ccc54)c3C#N)c2Oc2ccccc2)cc1"
    ),
    ("antibiotics10081002", "61b"): (
        "Cc1ccc(-n2nc(C)c(-c3c(C#N)c(N)n4c(nc5cc(C)ccc54)c3C#N)c2Oc2ccc(C)cc2)cc1"
    ),
    ("antibiotics10081002", "61c"): (
        "Cc1ccc(-n2nc(C)c(-c3c(C#N)c(N)n4c(nc5cc(C)ccc54)c3C#N)c2Oc2ccc(Cl)cc2)cc1"
    ),
    ("antibiotics10081002", "39i"): (
        "CC1=NN(C(=O)c2ccncc2)C(=O)C1=NNc1ccc(-c2nc3ccccc3[nH]2)cc1"
    ),
    ("antibiotics10081002", "74"): (
        "Cn1c(-c2c(S(=O)(=O)c3ccccc3)nnc3c(-c4ccc(Cl)cc4)cnn23)nc2ccccc21"
    ),
    ("antibiotics10081002", "88j"): (
        "C(=C\\C1CC(c2ccc(NCn3cnc4ccccc43)cc2)=NN1c1ccccc1)/c1ccccc1"
    ),
    ("antibiotics10081002", "98e"): (
        "CC1=CC(c2cc3nc(NC(=O)NCC)[nH]c3c(-n3cc(C)cn3)c2F)=CC(=O)C1Cc1ccccn1"
    ),
    ("antibiotics12071220", "88b"): (
        "Cc1nc2ccccc2n1Cc1nnc(S)n1-c1ccc(O)cc1"
    ),
    ("antibiotics12071220", "88c"): (
        "COc1ccc(-n2c(S)nnc2Cn2c(C)nc3ccccc32)cc1"
    ),
    ("antibiotics12071220", "37d"): (
        "O=c1n(Cc2ccccc2)c2ccccc2n1Cc1cn(Cc2ccccc2)nn1"
    ),
}


def canonicalize_smiles(smiles: str) -> str:
    """Validate and canonicalize a SMILES string using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def infer_bacteria_unified(bacteria: str) -> str:
    """Infer unified species name for local GT CSV."""
    low = bacteria.lower()
    if "coli" in low or "e.c." in low or "e. c." in low:
        return "Escherichia coli"
    if (
        "aureus" in low
        or "mrsa" in low
        or "vrsa" in low
        or "s.a." in low
        or "s. a." in low
    ):
        return "Staphylococcus aureus"
    return ""


def format_local_relation(rel: str) -> str:
    """Ensure '=' is quoted as \"'='\" in local CSV."""
    cleaned = rel.strip().strip("'")
    return "'='" if cleaned == "=" else cleaned


def load_csv(path: Path) -> List[Dict[str, str]]:
    """Load CSV into a list of row dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def should_delete(row: Dict[str, str]) -> bool:
    """Return True if row matches any confirmed DELETE rule."""
    pdf_low = row["pdf"].strip().lower()
    cid = row["compound_id"].strip()
    bac = row["bacteria"].strip()
    for d_pdf, d_cid, d_bac in DELETE_RULES:
        if pdf_low == d_pdf and cid == d_cid:
            if d_bac is None or bac == d_bac:
                return True
    return False


def apply_updates(row: Dict[str, str]) -> None:
    """Apply confirmed in-place UPDATE rules to a GT row."""
    pdf_low = row["pdf"].strip().lower()
    cid = row["compound_id"].strip()

    if (pdf_low, cid) in SMILES_UPDATES:
        row["smiles"] = SMILES_UPDATES[(pdf_low, cid)]

    if pdf_low == "rjc.2023.1638382" and cid == "15" and "aureus" in row["bacteria"].lower():
        row["target_value"] = "3,125"

    if pdf_low == "j.ejmech.2016.01.052" and cid == "14m" and "MRSA" in row["bacteria"]:
        row["target_value"] = "1"

    if pdf_low == "s41598-022-21435-6" and cid in {"11g", "11i", "12h"}:
        for old_code in ("ATCC 25926", "ATCC 25925", "ATCC 25924"):
            if old_code in row["bacteria"]:
                row["bacteria"] = row["bacteria"].replace(old_code, "ATCC 25923")

    if pdf_low == "acs.jafc.1c02545" and "aureus" in row["bacteria"].lower():
        if row["target_units"].strip() == "µmol/mL":
            row["target_units"] = "mM"

    if pdf_low == "antibiotics10081002" and cid == "67" and "aureus" in row["bacteria"].lower():
        row["target_value"] = "0,98"


def build_added_rows(
    gt_rows: List[Dict[str, str]],
    val_ext: List[Dict[str, str]],
    train_ext: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Build the 73 confirmed missing rows to add to GT."""
    gt_smiles_by_key: Dict[Tuple[str, str], str] = {
        (r["pdf"].strip().lower(), r["compound_id"].strip()): r["smiles"]
        for r in gt_rows
    }
    added: List[Dict[str, str]] = []

    def make_row(
        pdf: str,
        smiles: str,
        compound_id: str,
        target_type: str,
        target_relation: str,
        target_value: str,
        target_units: str,
        bacteria: str,
    ) -> Dict[str, str]:
        return {
            "pdf": pdf,
            "smiles": smiles,
            "compound_id": compound_id,
            "target_type": target_type,
            "target_relation": format_local_relation(target_relation),
            "target_value": target_value,
            "target_units": target_units,
            "bacteria": bacteria,
            "bacteria_unified": infer_bacteria_unified(bacteria),
        }

    # 1. RJC.2023.1638382 compounds 4, 24, and 28 (5 rows, MIC '=' from article text)
    for r in val_ext:
        if r["pdf"].strip().lower() == "rjc.2023.1638382" and r["compound_id"].strip() in {"4", "24", "28"}:
            added.append(
                make_row(
                    pdf="RJC.2023.1638382",
                    smiles=r["smiles"],
                    compound_id=r["compound_id"].strip(),
                    target_type=r["target_type"],
                    target_relation="=",
                    target_value=r["target_value"],
                    target_units=r["target_units"],
                    bacteria=r["bacteria"],
                )
            )

    # 3. j.ejmech.2016.01.052 E. coli JM109 (11 rows)
    jm109_cids = {"13d", "13h", "14h", "14n", "14p", "15c", "15d", "15e", "15k", "16a", "18b"}
    for r in val_ext:
        if (
            r["pdf"].strip().lower() == "j.ejmech.2016.01.052"
            and r["compound_id"].strip() in jm109_cids
            and "JM109" in r["bacteria"]
        ):
            cid = r["compound_id"].strip()
            added.append(
                make_row(
                    pdf="j.ejmech.2016.01.052",
                    smiles=gt_smiles_by_key[("j.ejmech.2016.01.052", cid)],
                    compound_id=cid,
                    target_type=r["target_type"],
                    target_relation=r["target_relation"],
                    target_value=r["target_value"],
                    target_units=r["target_units"],
                    bacteria=r["bacteria"],
                )
            )

    # 4. j.ejmech.2020.112996 12 clinical isolates MRSA/VRSA × 4 compounds (48 rows)
    for r in val_ext:
        if (
            r["pdf"].strip().lower() == "j.ejmech.2020.112996"
            and r["compound_id"].strip() in {"8b", "8c", "8h", "8i"}
            and "25922" not in r["bacteria"]
            and "29213" not in r["bacteria"]
        ):
            cid = r["compound_id"].strip()
            added.append(
                make_row(
                    pdf="j.ejmech.2020.112996",
                    smiles=gt_smiles_by_key[("j.ejmech.2020.112996", cid)],
                    compound_id=cid,
                    target_type=r["target_type"],
                    target_relation=r["target_relation"],
                    target_value=r["target_value"],
                    target_units=r["target_units"],
                    bacteria=r["bacteria"],
                )
            )

    # 5. antibiotics10081002 compound 35e (2 rows)
    for r in train_ext:
        if r["pdf"].strip().lower() == "antibiotics10081002" and r["compound_id"].strip() == "35e":
            added.append(
                make_row(
                    pdf="antibiotics10081002",
                    smiles=r["smiles"],
                    compound_id="35e",
                    target_type=r["target_type"],
                    target_relation=r["target_relation"],
                    target_value=r["target_value"],
                    target_units=r["target_units"],
                    bacteria=r["bacteria"],
                )
            )

    # 6. antibiotics12071220 7 missing E. coli rows
    ab12_cids = {"2d", "3a", "3f", "19a", "37b", "105a", "105b"}
    for r in train_ext:
        if (
            r["pdf"].strip().lower() == "antibiotics12071220"
            and r["compound_id"].strip() in ab12_cids
            and "coli" in r["bacteria"].lower()
        ):
            cid = r["compound_id"].strip()
            smiles = gt_smiles_by_key.get(("antibiotics12071220", cid), r["smiles"])
            added.append(
                make_row(
                    pdf="antibiotics12071220",
                    smiles=smiles,
                    compound_id=cid,
                    target_type=r["target_type"],
                    target_relation=r["target_relation"],
                    target_value=r["target_value"],
                    target_units=r["target_units"],
                    bacteria=r["bacteria"],
                )
            )

    return added


def write_csv(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, str]]) -> None:
    """Write rows to CSV with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply verified GT corrections.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/ground_truth/Benzimidazole antibiotics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/ground_truth/Benzimidazole antibiotics_corrected.csv"),
    )
    parser.add_argument(
        "--val-extracted",
        type=Path,
        default=Path("/tmp/val_extracted.csv"),
    )
    parser.add_argument(
        "--train-extracted",
        type=Path,
        default=Path("/tmp/train_extracted.csv"),
    )
    parser.add_argument(
        "--gs-output",
        type=Path,
        default=Path("/tmp/ground_truth_corrected_gs.csv"),
    )
    args = parser.parse_args()

    gt_rows = load_csv(args.input)
    val_ext = load_csv(args.val_extracted)
    train_ext = load_csv(args.train_extracted)

    kept_rows: List[Dict[str, str]] = []
    deleted_count = 0
    for row in gt_rows:
        if should_delete(row):
            deleted_count += 1
            continue
        apply_updates(row)
        kept_rows.append(row)

    added_rows = build_added_rows(gt_rows, val_ext, train_ext)
    all_rows = kept_rows + added_rows

    if deleted_count != 8:
        raise RuntimeError(f"Expected 8 deleted rows, got {deleted_count}")
    if len(added_rows) != 73:
        raise RuntimeError(f"Expected 73 added rows, got {len(added_rows)}")

    # Validate all SMILES and build Google Sheets rows
    gs_rows: List[Dict[str, str]] = []
    for idx, row in enumerate(all_rows, 1):
        canon = canonicalize_smiles(row["smiles"])
        gs_row = {
            "pdf": row["pdf"],
            "smiles": row["smiles"],
            "smiles_canonical": canon,
            "compound_id": row["compound_id"],
            "target_type": row["target_type"],
            "target_relation": row["target_relation"].strip().strip("'"),
            "target_value": row["target_value"],
            "target_units": row["target_units"],
            "bacteria": row["bacteria"],
        }
        gs_rows.append(gs_row)

    write_csv(args.output, LOCAL_COLUMNS, all_rows)
    write_csv(args.gs_output, GS_COLUMNS, gs_rows)

    print(
        f"Success: {len(gt_rows)} -> {len(all_rows)} rows "
        f"(deleted={deleted_count}, added={len(added_rows)}). "
        f"Saved {args.output} and {args.gs_output}."
    )


if __name__ == "__main__":
    main()
