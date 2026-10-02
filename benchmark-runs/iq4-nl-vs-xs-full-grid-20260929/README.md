# IQ4_NL vs IQ4_XS full-grid benchmark

This is a fully crossed head-to-head benchmark for the UD-IQ4_NL and UD-IQ4_XS
Qwen3.6 35B MoE quants. Every model runs each context, thread, parallelism,
and batch combination with the same settings.

| Axis | Values |
|---|---|
| Context | 65,536; 131,072; 204,800 |
| CPU threads | 6; 8; 12 |
| Parallel slots | 1; 2 |
| Batch | 256; 512; 2,048 |
| Ubatch | 512 (fixed) |
| Fit target | 256 MiB (fixed) |
| K/V cache | q5_1 / q5_1 (fixed) |
| KV pool | Unified; 204,800 tokens per slot cap |
| Cache reuse / FlashAttention | 0 / on (fixed) |
| Preserve thinking | off (fixed) |

The 108 settings each get one warm-up and seven counted repetitions in the same
server session. The schedule runs paired model settings adjacently. Total:
108 warm-ups plus 756 counted runs. The harness uses the standalone prompt
script and binds llama-server to port 1234.

Run or resume in Screen with `./run.sh`. `matrix.json` is the exact analysis
coverage manifest. After completion, create the audited analysis with:

```sh
bin/llama-bench-harness analyze \
  --min-runs 7 \
  --matrix-manifest benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/matrix.json \
  --out benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/analysis.md \
  benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/runs.csv
```
