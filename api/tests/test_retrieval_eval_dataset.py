from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.validate_retrieval_eval_dataset import (
    DEFAULT_COLLECTION_DIR,
    DEFAULT_DATASET_DIR,
    DatasetValidationError,
    EXPECTED_CATEGORY_COUNTS,
    validate_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_retrieval_eval_v1_matches_collection_and_split_contract() -> None:
    summary = validate_dataset()

    assert summary["questionCount"] == 250
    assert summary["categoryDistribution"] == EXPECTED_CATEGORY_COUNTS
    assert summary["judgmentDistribution"] == {
        "deterministic_field_gold": 100,
        "pooled_silver_pending_human_review": 137,
        "deterministic_evidence_boundary_no_relevant_objects": 13,
    }
    assert summary["qrelDistribution"] == {
        "deterministic_field_gold": 332,
        "pooled_silver_pending_human_review": 1644,
    }
    assert summary["objectsSha256"] == (
        "8a7cee291acc9e5beedd40cf12e30299049613cbb61b6954e1a4fb9a5758d55e"
    )


def test_validator_rejects_qrel_evidence_not_owned_by_frozen_object(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "retrieval_eval_v1"
    shutil.copytree(DEFAULT_DATASET_DIR, dataset_dir)
    qrels_path = dataset_dir / "qrels.jsonl"
    rows = qrels_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["supportingEvidenceIds"] = ["invented:evidence-id"]
    rows[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
    qrels_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    # Keep the file-integrity layer satisfied so this test reaches the stronger
    # object/evidence ownership check.
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["qrels"]["sha256"] = _sha256(qrels_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="unknown evidence IDs"):
        validate_dataset(dataset_dir, DEFAULT_COLLECTION_DIR)
