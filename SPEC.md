# quantlens SPEC (every agent must read this first)

Project: GPT-2 small (124M) inference engine written from scratch in PyTorch (CPU), used to
measure where int8 quantization hurts and whether SmoothQuant-style rescaling repairs it.
Research question: Which layers and activation channels cause int8 quantization error in
GPT-2 small, and does a SmoothQuant-style rescaling recover the lost perplexity?

## Rules
- Python 3.12, PyTorch CPU only, Windows-safe paths (pathlib), fixed seeds.
- Engine code lives in src/quantlens and must NEVER import a Hugging Face model class.
  transformers is allowed only for: weights loading (weights.py reads the state dict),
  the tokenizer, and as the test oracle in tests/.
- Allowed primitives: torch.matmul, torch.nn.functional (softmax, gelu tanh-approx, layer_norm not
  allowed: write LayerNorm manually). Attention, LayerNorm, MLP, embeddings, KV cache,
  sampling and every quantizer are written by hand.
- Every number quoted anywhere must come from a file in results/.
- Do not commit weights, datasets, .venv.
- Work one phase at a time. Stop when the phase gate passes. Do not start the next phase.

## Interfaces (fixed so files written separately fit together)
- quantlens.model: GPT2Config (dataclass: vocab_size=50257, n_ctx=1024, n_embd=768,
  n_head=12, n_layer=12, eps=1e-5) and GPT2(nn.Module) with forward(input_ids [B,T] long)
  -> logits [B,T,vocab_size] float32. Output head is tied to the token embedding.
- quantlens.weights: load_pretrained(name="gpt2") -> GPT2 with HF weights copied in.

## Phases and gates
P0 setup: repo, env, SPEC, failing logits test.
P1 model.py + weights.py: tests/test_logits_match_hf.py passes (20 prompts, assert_close).
P2 kvcache.py + generate.py: cached and uncached outputs identical (test_kvcache_equivalence.py).
P3 eval/perplexity.py: WikiText-2 test, context 1024, fixed stride. Record V0 fp32 in results/.
P4 quant/int8.py: weight-only per-tensor (V1) and per-channel (V2); test_quant_roundtrip.py;
   memory + perplexity in results/. Minimum shippable point.
P5 quant/act.py (naive W8A8, V3), eval/outliers.py, per-block sensitivity.
P6 quant/smooth.py (V4), eval/bench.py, figures, README, 3-4 page report, CI.
Stretch (V5 int4 group-wise) only after the report draft exists.

## Experiment variants
V0 fp32 | V1 int8 weights per-tensor | V2 per-channel | V3 W8A8 naive | V4 W8A8 + smoothing | V5 int4 group-wise.
Metrics: perplexity (WikiText-2, sliding window 1024), weight bytes, tokens/sec at context 128/512/1024
with and without KV cache (median of 3 runs, batch 1).
Write hypotheses before running: naive W8A8 degrades more than weight-only; a few channels dominate
activation range. Report the result even if wrong.

## Known GPT-2 traps
Conv1D weights are stored transposed; GELU is the tanh approximation; LayerNorm eps 1e-5;
output head shares token embedding; attention mask is causal.