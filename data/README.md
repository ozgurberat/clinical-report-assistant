# Data card — Open-i / Indiana University Chest X-ray reports

## Source

- **Official:** NLM Open-i service — https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz (reports, XML) and `NLMCXR_png.tgz` (images).
- **Mirror (recommended if the official host is slow/unreachable):** [Kaggle — raddar/chest-xrays-indiana-university](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
- **Mirror:** [Academic Torrents — XML reports](https://academictorrents.com/details/66450ba52ba3f83fbf82ef9c91f2bde0e845aba9), [PNG images](https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d)

Scale: ~3,955 radiology reports paired with ~7,470 chest X-ray images (frontal + lateral views for most studies).

## License / usage

Open-i content is made available by the NLM for research use; no PhysioNet-style CITI training/credentialing is required (unlike MIMIC-CXR). Always check the current terms on the [Open-i site](https://openi.nlm.nih.gov) before redistributing. This repo does not commit any raw or processed patient data — `data/raw/` and `data/processed/` are gitignored.

## Raw format

Each report is an XML file (`NLMCXR_reports/<id>.xml`) with a structure roughly like:

```xml
<eCitation>
  <pmcId .../>
  <MedlineCitation>
    <MeSH>
      <major>...</major>
    </MeSH>
    <Article>
      <Abstract>
        <AbstractText Label="COMPARISON">...</AbstractText>
        <AbstractText Label="INDICATION">...</AbstractText>
        <AbstractText Label="FINDINGS">...</AbstractText>
        <AbstractText Label="IMPRESSION">...</AbstractText>
      </Abstract>
    </Article>
  </MedlineCitation>
  <parentImage id="CXR1234_IM-0001-1001">...</parentImage>
</eCitation>
```

Field presence is inconsistent across reports (e.g. many are missing `COMPARISON` or `INDICATION`); the preprocessing script handles this defensively.

## Processed schema

`src/data/preprocess.py` parses the raw XML into `data/processed/reports.jsonl`, one JSON object per report:

```json
{
  "report_id": "CXR1234",
  "comparison": "None.",
  "indication": "Positive TB test",
  "findings": "The cardiomediastinal silhouette is within normal limits...",
  "impression": "No acute cardiopulmonary abnormality.",
  "mesh_major": ["normal"],
  "image_ids": ["CXR1234_IM-0001-1001", "CXR1234_IM-0001-2001"]
}
```

Plus a flattened `data/processed/reports.csv` for quick inspection, and `data/processed/splits.json` with train/val/test report-id splits (80/10/10, seeded) for downstream fine-tuning.

## Known limitations

- Small dataset (~4k reports) relative to modern LLM fine-tuning norms — mitigated via QLoRA (parameter-efficient) rather than full fine-tuning, and documented explicitly as a small-data regime.
- Reports vary widely in verbosity and section completeness.
- If more volume is needed for RAG indexing, this repo may supplement with clearly-labeled synthetic reports or n2c2/i2b2 data — any such addition will be documented here with provenance, never silently merged with real patient-derived text.
