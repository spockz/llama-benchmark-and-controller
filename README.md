# llama.cpp end-to-end benchmark harness

This Go harness benchmarks llama.cpp configurations using the same coding-agent
task for each run. Its primary measure is end-to-end task wall time; it also
records prompt and decode throughput, server metrics, GPU utilization, and fit
information. The hosted model is called through `standalone-prompt/run.sh`.

## Build and test

Use the Makefile to build and verify the harness:

```sh
make test
make build
make vet
```

The executable is `bin/llama-bench-harness`. The default llama.cpp checkout is
`../llama.cpp`; pass `--workdir /path/to/llama.cpp` when it lives elsewhere.

## Define a benchmark

The harness combines setting axes as a Cartesian product. `--threads` is an
axis; the `--*-values` options define other axes. Each `--*-values` list is
combined with its scalar option, so include the scalar value if you want it in
the matrix. For example, use `--batch 256 --batch-values 256,512,2048` to test
those three batch sizes. Fixed settings can be supplied as scalar options, or
as singleton value lists when you want the analysis manifest to audit them.

Available axes include models, contexts, parallel slots, batch and ubatch,
fit targets, K/V cache types, cache reuse, FlashAttention, and preserve-thinking.
`--kv-per-slot` and `--kv-unified` are fixed options rather than matrix axes.
Keep the same coding-agent prompt, sampling settings, and telemetry options
across the cells being compared.

Here is the full Q4_NL vs Q4_XS matrix used in this repository:

| Setting | Values |
|---|---|
| Model | `UD-IQ4_NL`, `UD-IQ4_XS` |
| Context | 65,536; 131,072; 204,800 |
| Threads | 6; 8; 12 |
| Parallel slots | 1; 2 |
| Batch | 256; 512; 2,048 |
| Ubatch | 512 (fixed) |
| Fit target | 256 MiB (fixed) |
| K/V cache | q5_1 / q5_1 (fixed) |

Unified KV, a 204,800 token per-slot limit, cache reuse 0, FlashAttention on,
and preserve-thinking off are also fixed. The matrix has 108 model/configuration/
thread cells. See [the matrix manifest](benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/matrix.json)
and [its launcher](benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/run.sh).

For a complete factorial matrix, list all values directly on the harness
command line. Each combination of model, context, parallelism, batch, and other
setting axes gets a separate server session for each thread count. The harness
starts the server, performs the requested warm-up runs, then records the
counted runs before moving to the next cell. A warm-up therefore applies to
that exact model/configuration/thread combination. With `R` counted runs and
`W` warm-ups, a fully completed matrix of `C` cells performs `C × (R + W)`
client runs. Large Cartesian products grow quickly; estimate the cell count
before starting.

For example, this command defines a two-model, two-context, two-parallelism,
two-batch matrix across three thread counts (48 cells):

```sh
bin/llama-bench-harness \
  --workdir /path/to/llama.cpp \
  --out benchmark-runs/example \
  --model 'unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL' \
  --model-values 'unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL,unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS' \
  --threads 6,8,12 \
  --ctx 65536 --ctx-values 65536,131072 \
  --parallel 1 --parallel-values 1,2 \
  --batch 256 --batch-values 256,512 \
  --ubatch 512 \
  --fit-target 256 --fit-target-values 256 \
  --kv-k q5_1 --kv-k-values q5_1 \
  --kv-v q5_1 --kv-v-values q5_1 \
  --kv-per-slot 131072 --kv-unified \
  --runs 7 --warmup-runs 1 --port 1234 --resume
```

`--kv-per-slot` should be large enough for the per-slot context you intend to
test. Use `--model-values` and the other axis flags to compare multiple models
under exactly the same settings. Keep a JSON analysis manifest with the same
values to verify coverage after the run.

When a full cross-product is too large, use the pairwise planner with an
analysis matrix manifest:

```sh
bin/llama-bench-harness plan \
  --matrix-manifest benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/matrix.json \
  --out benchmark-runs/pairwise-plan.json
```

Run that plan with `--plan benchmark-runs/pairwise-plan.json`. A pairwise plan
covers each pair of factor values at least once, but does not measure every
full-factorial combination; use the full matrix when direct head-to-head
results are required for every matching configuration.

## Run in Screen

Run long benchmarks in Screen so they are detached from the interactive shell
and continue if the coding agent or terminal disconnects. Build and test first,
then start the checked-in matrix from the repository root:

```sh
screen -dmS q4-matrix ./benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/run.sh
screen -ls
```

Reattach with `screen -r q4-matrix`; detach without stopping the benchmark with
`Ctrl-a`, then `d`. Follow the run with:

```sh
tail -f benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/screen.log
```

Screen protects the process from terminal or agent disconnection, but it does
not restart a command that exits. The checked-in launcher retries the harness
with `--resume` after errors and audits the matrix before it exits. For custom
benchmarks, use `--resume` with the same output directory and exactly the same
matrix arguments after interruption. The harness reuses successful matching
runs and retries failed or missing ones. Do not run two harnesses against the
same output directory or server port at the same time.

## Analyze results

The harness writes `runs.csv` for individual attempts, `summary.csv` and
`summary.md` for grouped results, and per-run logs and metadata under the
output directory. The analyzer ranks cells by median end-to-end wall time.
Require complete coverage against the exact manifest with:

```sh
bin/llama-bench-harness analyze \
  --min-runs 7 \
  --matrix-manifest benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/matrix.json \
  --out benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/analysis.md \
  benchmark-runs/iq4-nl-vs-xs-full-grid-20260929/runs.csv
```

`--min-runs` is the minimum number of successful counted runs required per
cell. The manifest audit checks the full Cartesian matrix and fails if cells
are missing or under-sampled. Compare medians and ranges: coding-agent task
length varies, so a single fast run is not a reliable winner.
