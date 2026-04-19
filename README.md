# PRISM

PRISM is a zero-cost and largely data-free mixed-precision quantization pipeline with:

- Stage 0 meta-learner training
- Stage 1 zero-cost profiling
- Stage 2 budgeted LP-style bit assignment
- Stage 2.5 QUIC output-space correction
- Stage 3 RTN precomputation
- Stage 4 runtime assembly with Marlin preference and PyTorch GEMM fallback

## Demo

```bash
python -m pytest tests/prism -v
prism-demo --output-root artifacts/prism/demo
```
