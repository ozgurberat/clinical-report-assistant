"""Unit tests for src.data.preprocess — run offline against a synthetic sample XML
that mirrors the real Open-i report schema, so CI doesn't depend on network access."""

from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from src.data.preprocess import build_splits, has_duplicate_artifact, parse_report

SAMPLE_XML = """<?xml version="1.0"?>
<eCitation>
  <MedlineCitation>
    <MeSH>
      <major>cardiomegaly</major>
      <major>normal</major>
    </MeSH>
    <Article>
      <Abstract>
        <AbstractText Label="COMPARISON">None.</AbstractText>
        <AbstractText Label="INDICATION">Chest pain.</AbstractText>
        <AbstractText Label="FINDINGS">The heart size is mildly enlarged. Lungs are clear.</AbstractText>
        <AbstractText Label="IMPRESSION">Mild cardiomegaly, no acute disease.</AbstractText>
      </Abstract>
    </Article>
  </MedlineCitation>
  <parentImage id="CXR1_IM-0001-1001"/>
  <parentImage id="CXR1_IM-0001-2001"/>
</eCitation>
"""


@pytest.fixture
def sample_xml_path(tmp_path: Path) -> Path:
    p = tmp_path / "CXR1.xml"
    p.write_text(SAMPLE_XML)
    return p


def test_parse_report_extracts_all_sections(sample_xml_path: Path):
    rec = parse_report(sample_xml_path)
    assert rec is not None
    assert rec["report_id"] == "CXR1"
    assert rec["comparison"] == "None."
    assert rec["indication"] == "Chest pain."
    assert "mildly enlarged" in rec["findings"]
    assert "cardiomegaly" in rec["impression"].lower()
    assert rec["mesh_major"] == ["cardiomegaly", "normal"]
    assert rec["image_ids"] == ["CXR1_IM-0001-1001", "CXR1_IM-0001-2001"]


def test_parse_report_handles_missing_sections(tmp_path: Path):
    minimal = """<?xml version="1.0"?>
    <eCitation>
      <MedlineCitation>
        <Article>
          <Abstract>
            <AbstractText Label="FINDINGS">Lungs are clear.</AbstractText>
          </Abstract>
        </Article>
      </MedlineCitation>
    </eCitation>
    """
    p = tmp_path / "CXR2.xml"
    p.write_text(minimal)
    rec = parse_report(p)
    assert rec is not None
    assert rec["comparison"] == ""
    assert rec["indication"] == ""
    assert rec["impression"] == ""
    assert rec["mesh_major"] == []
    assert rec["image_ids"] == []


def test_parse_report_returns_none_on_malformed_xml(tmp_path: Path):
    p = tmp_path / "bad.xml"
    p.write_text("<not><valid xml")
    assert parse_report(p) is None


def test_has_duplicate_artifact_flags_embedded_second_report():
    # Mirrors real report CXR2415: impression contains a second full report
    # (header + findings + impression) concatenated after the real impression.
    rec = {
        "findings": "The heart is mildly enlarged.",
        "impression": (
            "Hypoinflation with elevated left hemidiaphragm. "
            "IMPRESSION: Exam: CHEST 2V FRONTAL/LATERAL Date: XXXX "
            "FINDINGS: The heart is mildly enlarged. "
            "IMPRESSION: Hypoinflation with elevated left hemidiaphragm."
        ),
    }
    assert has_duplicate_artifact(rec) is True


def test_has_duplicate_artifact_ignores_clean_reports():
    rec = {
        "findings": "The lungs are clear. No focal consolidation.",
        "impression": "No acute cardiopulmonary process.",
    }
    assert has_duplicate_artifact(rec) is False


def test_build_splits_partitions_all_ids_without_overlap():
    cfg = {"split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42}}
    ids = [f"CXR{i}" for i in range(100)]
    splits = build_splits(ids, cfg)

    all_split_ids = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_split_ids) == sorted(ids)
    assert len(set(all_split_ids)) == len(ids)
    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10


def test_build_splits_is_deterministic_given_seed():
    cfg = {"split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42}}
    ids = [f"CXR{i}" for i in range(50)]
    assert build_splits(ids, cfg) == build_splits(ids, cfg)
