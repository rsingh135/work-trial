# Researcher Work-Trial Reference Bundle

This is a candidate-safe materials bundle for a research exercise using the BPI
Challenge 2013 and 2020 logs, followed by a small AppWorld integration. It
contains raw data, source documentation, a narrow reading list, and faithfully
transcribed published reference results. It intentionally contains no preferred
model architecture, existing project model code, or internal experiment notes.

## Start here

1. Read `docs/LOG_CATALOG.md` for the scope and important dataset caveats.
2. Read `docs/DATA_USAGE_AND_CITATION.md` before redistributing any data.
3. Read `papers/READING_GUIDE.md`; the required reading is deliberately short.
4. Use `results/PUBLISHED_BPI2013_RESULTS.md` only for comparisons that match
   the paper's evaluation protocol.
5. Read `results/BPI2020_RESULTS_NOTE.md` before treating any BPI 2020 number as
   a benchmark target.
6. Follow `docs/APPWORLD_SETUP.md` to install AppWorld separately.

## Contents

```text
data/
  bpi2013/    Three official XES logs plus their 4TU metadata
  bpi2020/    Five official XES logs plus the common source README
docs/
  APPWORLD_SETUP.md
  DATA_USAGE_AND_CITATION.md
  LOG_CATALOG.md
  source_metadata/   Machine-readable 4TU article records
papers/
  bpi2013_challenge_summary.pdf
  bpi2013_dataset_guide.pdf
  bpi2020_attribute_explanation.pdf
  deep_learning_predictive_process_monitoring_benchmark_v4.pdf
  appworld_acl2024.pdf
  READING_GUIDE.md
results/
  PUBLISHED_BPI2013_RESULTS.md
  BPI2020_RESULTS_NOTE.md
tools/
  inspect_xes.py
CHECKSUMS.sha256
```

## Intentional exclusions

- No preprocessed graph, token, or learned representation is supplied.
- No internal project model code or internal model results are supplied.
- No fixed train/test split is supplied; the work-trial brief should state the
  expected protocol explicitly.
- AppWorld itself is not copied into this archive. Its maintainers distribute
  parts of the benchmark in protected bundles and request that unpacked content
  not be republished. Install it from the official package instead.
- The 77-page VINST application manual is linked from the reading guide but is
  not duplicated. The included 12-page dataset guide contains the documentation
  needed to interpret the BPI 2013 event fields.

## Quick integrity check

From the bundle root:

```bash
shasum -a 256 -c CHECKSUMS.sha256
python3 tools/inspect_xes.py --pretty data/bpi2013/*.xes.gz data/bpi2020/*.xes.gz
```

The task brief and the evaluator's private rubric should be distributed
separately from this reference bundle.
