# PROGRESS (update at the end of every phase; this is the handoff file)

Repo: quantlens | Env: Windows, PowerShell, .venv, CPU torch
Current phase: All Core Phases (P0 through P6) Completed and Validated!

Completed Phases:
- P0: Repo scaffold, configuration, setuptools packaging, pytest configuration.
- P1: model.py and weights.py implemented; custom GPT-2 logits match HF oracle (test_logits_match_hf).
- P2: kvcache.py and generate.py; test_kvcache_equivalence passing (verified 4.2x-14.0x generation speedup).
- P3: eval/perplexity.py; full WikiText-2 baseline evaluated (V0 ppl: 25.17, 561 windows, results/v0_ppl.json).
- P4: quant/int8.py (V1 per-tensor ppl: 29.25, V2 per-channel ppl: 25.72, 51.1% memory reduction); test_quant_roundtrip passing.
- P5: quant/act.py (V3 naive W8A8 ppl: 36.25); eval/outliers.py; hypotheses tested and confirmed in docs/hypotheses.md.
- P6: quant/smooth.py (V4 SmoothQuant); eval/bench.py; test_smooth_and_bench passing;
      scripts/run_bench.py, scripts/run_outliers.py, scripts/make_figures.py;
      publication figures in reports/figures/; 3-4 page research report in reports/research_report.md;
      CI workflow .github/workflows/ci.yml; comprehensive README.md.

Gate status:
- pytest -m "not slow" -q: 32 passed, 2 deselected (100% green)
- All experimental numbers logged in results/*.json
- All report figures generated in reports/figures/*.png

Optional Next / Stretch:
- V5: Int4 group-wise quantization (stretch goal per SPEC.md).