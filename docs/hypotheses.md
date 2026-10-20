# Hypotheses (written before running any quantized variant)

These are guesses, not results.

H1: Naive W8A8 (V3) degrades perplexity more than weight-only int8 (V1, V2).
H2: A small number of activation channels dominate the activation range in the inputs to
    linear layers, which would explain H1.

If either is wrong, the report says so.