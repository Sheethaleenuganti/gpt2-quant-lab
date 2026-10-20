# PROGRESS (update at the end of every phase; this is the handoff file)

Repo: quantlens | Env: Windows, PowerShell, .venv, CPU torch
Current phase: P4 done
Done:
- P0: Repo scaffold, configuration, directory structure.
- P1: model.py and weights.py implemented; logits match HF oracle.
- P2: kvcache.py and generate.py implemented; tests/test_kvcache_equivalence.py passing.
- P3: eval/perplexity.py implemented; smoke test verified (results/v0_ppl_partial.json).
- P4: quant/int8.py implemented (quantize_per_tensor, quantize_per_channel, QuantLinear,
  quantize_model); scripts/run_experiments.py updated with v1/v2 variants and fixed
  weight_bytes to include buffers; tests/test_quant_roundtrip.py all passing.
Next: P5 - quant/act.py (naive W8A8, V3), eval/outliers.py, per-block sensitivity.
Gate status:
- test_logits_match_hf: PASS (slow, skipped in fast run)
- test_kvcache_equivalence: PASS
- test_perplexity: PASS
- test_quant_roundtrip: PASS (11 new tests)
- pytest -m "not slow" -q: 15 passed, 2 deselected
Notes: Always use `.venv\Scripts\python.exe` or activate `.venv` in PowerShell before running commands.