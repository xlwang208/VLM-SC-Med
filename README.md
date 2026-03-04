# Anonymous Code Release: VLM-SC-Med

This repository contains an **anonymized, cleaned** implementation of the core pipeline described in the accompanying MICCAI submission:

**Vision-Language Model Guided Semantic Curation for Large-Scale Medical Data Cohorts**


## 🏗 Pipeline Overview

![Pipeline](figures/pipeline.png)


## 📂 What is included

- A lightweight, dataset-agnostic **agentic reconciliation module** that fuses:
  - visual/VLM evidence (e.g., 2D projections),
  - segmentation-derived organ presence evidence,
  - DICOM metadata evidence,
  - report/NLP evidence,
  and outputs a single **anatomical coverage label**.
- An optional **study-level consistency** post-processing step.

## 🔒 What is NOT included 

Due to privacy restrictions and ongoing submissions of related work, we do **not** provide:
- any clinical CT volumes, DICOMs, or radiology reports,
- any derived outputs from private cohorts,
- any non-public embedding extraction code or embedding files used only for evaluation in the paper.

This repository provides a minimal and cleaned implementation of the core methodology described in the paper. Some dataset-specific preprocessing steps and intermediate utilities from the original research pipeline are not included because they depend on restricted datasets or internal processing workflows. The released code therefore focuses on the main methodological components while preserving the functional behavior of the original pipeline.

---

## Quickstart (no LLM required)

Run the toy example with a deterministic **mock** reconciler:

```bash
pip install -r requirements.txt
pip install -e .
python -m scripts.run_pipeline_mock \
  --in-jsonl data/example/input_example.jsonl \
  --out-jsonl output_example.jsonl
```

---

## Run with an OpenAI-compatible LLM endpoint

This code works with any OpenAI-compatible **Chat Completions** endpoint, including local servers (e.g., vLLM).

```bash
pip install -r requirements.txt
pip install -e .

python -m scripts.run_pipeline \
  --in-jsonl  /path/to/your_input.jsonl \
  --out-jsonl /path/to/curated_output.jsonl \
  --base-url  http://127.0.0.1:8001/v1 \
  --api-key   EMPTY \
  --model     YOUR_MODEL_NAME
```

### Input JSONL schema (one CT series per line)

Minimum fields expected by the pipeline:

```json
{
  "volume_id": "any-string-identifier",
  "visual_evidence": { "vlm_label": "...", "structures": [...], "confidence": "..." },
  "seg_evidence":    { "organs_present": [...] },
  "meta_evidence":   { "BodyPartExamined_raw": "...", "normalized_bodypart": "..." },
  "report_evidence": { "report_label": "...", "mentioned_organs": [...] }
}
```

You can omit any evidence block (set it to null / remove the key). The reconciler is instructed to handle missing modalities.

---

## Notes on reproducibility

Different institutions have different identifier conventions (patient/study/series IDs, date tokens, etc.).
The provided `vlm_sc_med/consistency.py` uses conservative heuristics; please adapt `extract_patient_id()` and `extract_study_date()` to your dataset.

---

## 📊 evaluation utilities

This release also includes **evaluation-only** utilities extracted from the original project:

- `vlm_sc_med.eval_manualgt`: evaluate predicted body-part labels against a *manual* ground truth table (CSV/JSONL input).
- `vlm_sc_med.bodypart_mlp`: a reference 5-label MLP trainer that consumes **user-provided** embedding `.npy` files.

**Not included:** any private embeddings, embedding extractors, or datasets. You can still run the code by providing your own:
- `cases.csv` with `volume_id, labels, embedding_path`
- `.npy` embeddings referenced by `embedding_path`

### Quick usage

Train an MLP on embeddings (requires `torch` installed):
```bash
python -m scripts.train_bodypart_mlp --cases-csv /path/to/cases.csv --out-model mlp.joblib
```

You can also run a toy evaluation demo:
```bash
python -m scripts.evaluate_manualgt --pred data/example/pred_labels.csv --manual-gt data/example/manual_gt.csv
```

Evaluate predictions vs manual GT:
```bash
python -m scripts.evaluate_manualgt --pred /path/to/pred.csv --manual-gt /path/to/manual_gt.csv
```

---

## License

MIT License (see `LICENSE`).
