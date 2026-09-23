# Reproducibility

Docker Compose is the only host requirement. Run `python -m mfcontrol reproduce --profile paper` inside the research service. The bind-mounted pipeline writes timestamped `logs/reproduce_<timestamp>.log`, append-only `logs/events.jsonl`, and atomic `artifacts/progress.json`. `results/result_manifest.json` records code/config/lock hashes; `results/paired_controller_comparisons.csv` stores paired effect evidence. Raw Azure data is gitignored.
