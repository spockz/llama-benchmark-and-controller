# Remote Implementation Handover — llama.cpp Benchmark & Profiling Harness

## Objective

Implement and harden a **single Go program** that can reproducibly benchmark my local `llama.cpp` inference workstation end-to-end.

The program must own the full experiment lifecycle:

1. Generate/run the desired llama.cpp server configuration.
2. Start `llama-server`.
3. Wait until `/health` is ready.
4. Optionally run `llama-fit-params` and capture fitted tensor placement.
5. Start AMD profiling:
   - `AMDuProfPcm` for DDR4 bandwidth.
   - `amd-smi` for GPU utilization, power, clocks/VRAM where available.
6. Start sampling llama.cpp `/metrics`.
7. Run the real client benchmark command.
8. Capture client + server + profiler logs.
9. Stop everything cleanly.
10. Parse all collected data.
11. Produce per-run and aggregate summaries (CSV + JSON + Markdown).
12. Repeat a configurable experiment matrix enough times to make medians meaningful.

The purpose is **not** merely to maximize raw llama.cpp tokens/sec. The primary metric is **end-to-end coding-agent task latency**. Server throughput, prompt-processing speed, generated-token speed, DDR bandwidth, GPU utilization, cancellations, cache reuse, and request overlap are explanatory metrics.

---

# Hardware

Inference workstation:

- CPU: AMD Ryzen 9 5900X
  - 12 cores / 24 threads
  - 2 CCDs, 6 physical cores per CCD
- RAM: 64 GiB DDR4, dual channel
- GPU: AMD Radeon RX 9070 XT
  - 16 GiB VRAM
  - RDNA4 / `gfx1201`
- PCIe:
  - PCIe 4.0 x16
  - `amd-smi metric --pcie` reports:
    - WIDTH: 16
    - SPEED: 16 GT/s

This is a **dedicated inference workstation**. The coding client/OpenCode normally runs elsewhere, although the benchmark harness will initially run both caller and server on this workstation for easier coordination.

---

# OS / Driver / Tooling

Current environment observed during testing:

```text
OS/kernel:
  CachyOS
  Linux 7.2.2-1-cachyos-bore-lto

ROCm:
  7.2.4

AMD SMI:
  package: amdsmi 7.2.4-1.1
  AMDSMI Tool: 26.2.2
  AMDSMI Library: 26.2.2

GPU driver:
  amdgpu

Vulkan:
  RADV
```

`amd-smi monitor --pcie` currently reports `N/A` for bandwidth counters. This is a telemetry limitation, not a bad PCIe link.

`AMDuProfPcm` 5.3.521 is available separately and works with Linux Perf / `amd_uncore`.

Before memory profiling:

```bash
sudo modprobe amd_uncore
```

uProf output confirmed:

```text
Data Collection Driver: Linux Perf Driver
NMI Watchdog Enabled: 0
```

---

# llama.cpp

`llama.cpp` is built locally.

Maintain two separate build trees:

```text
build-rocm/
build-vulkan/
```

Vulkan is materially faster than ROCm on this workload/GPU.

An earlier baseline with Qwen3.6 IQ3_S showed:

```text
ROCm:
  pp2048  ≈ 2701 tok/s
  pp8192  ≈ 2638 tok/s
  tg128   ≈ 88.6 tok/s

Vulkan:
  pp2048  ≈ 3724 tok/s
  pp8192  ≈ 3558 tok/s
  tg128   ≈ 129.8 tok/s
```

Use Vulkan as the primary backend for these experiments.

One observed llama.cpp build identifier during testing:

```text
ec9281505 (11042)
```

The repository may have moved since then; always record the current git commit in benchmark metadata.

---

# Models Investigated

Primary model family:

```text
Qwen3.6-35B-A3B
```

MoE model, approximately 35B stored / 3B active.

Quantizations tested:

```text
unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_S
unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q3_K_S
unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q3_K_M
unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_S
```

A dense Qwen3.8 experiment was also prepared:

```text
unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS
```

but the current benchmarking focus is **Qwen3.6 UD-Q3_K_M**.

---

# Current Primary Configuration

The current quality/performance candidate is:

```text
Model:       Qwen3.6-35B-A3B UD-Q3_K_M
Backend:     Vulkan
Context:     65536
KV K:        q5_1
KV V:        q5_1
Batch:       2048
UBatch:      256
Fit target:  512 MiB
Reasoning:   on
N CPU MoE:   0 explicitly
Threads:     being tested (6 / 8 / 12)
```

Relevant command shape:

```bash
HF=unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q3_K_M \
CTX=65536 \
KV_K=q5_1 \
KV_V=q5_1 \
BATCH=2048 \
UBATCH=256 \
THREADS=<6|8|12> \
FIT_TARGET=512 \
PARALLEL=<1|2 depending experiment> \
CACHE_REUSE=0 \
EXTRA_SERVER_ARGS="--kv-unified --kv-unified-per-slot 65536" \
make serve-vulkan-think
```

Do not assume `PARALLEL=2` is always beneficial. It increases aggregate throughput but can severely hurt task latency when one slot is doing a large prefill.

---

# Important Placement Results

`llama-fit-params` is essential. Do not infer CPU/GPU placement only from `-ngl`.

Example command:

```bash
./build-vulkan/bin/llama-fit-params \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q3_K_M \
  -c 65536 \
  -ctk q5_1 \
  -ctv q5_1 \
  -b 2048 \
  -ub 256 \
  -fa on \
  -fitt 512
```

## Q4_K_S + 64K + Q5_1 + fit target 768

Fitter placed FFN/MoE tensors on CPU from roughly blocks 27–40.

Observed generation:

```text
~45–47 tok/s
```

GPU utilization/power were low.

Interpretation: too much CPU expert spill.

## Q3_K_S + 64K + Q5_1 + fit target 768

CPU expert spill only around blocks 37–40.

Not yet fully benchmarked end-to-end.

## Q3_K_M + 64K + Q5_1 + fit target 768

CPU expert spill around blocks 34–40.

Observed:

```text
~56.5 tok/s
~70% GPU utilization
~80 W GPU power
```

## Q3_K_M + 64K + Q4_0 KV + fit target 768

CPU expert spill around blocks 35–40.

Observed:

```text
~59.4 tok/s
```

## Q3_K_M + 64K + Q5_1 KV + fit target 512

Fitter result:

```text
blk.35 ffn_down... = CPU
blk.36–40 experts/FFN = CPU
```

Observed:

```text
~59.5 tok/s
~70% GPU utilization
~79–86 W GPU power
```

This is currently preferable to Q4 KV because it achieves essentially the same speed while retaining Q5_1 KV precision.

---

# KV Cache Findings

128K Q8 KV caused much lower GPU power/utilization and lower generation speed.

Q5_1 is currently the preferred compromise.

Useful observed behavior:

```text
Q3_K_M + Q5_1 + fit 768  ≈ 56.5 tok/s
Q3_K_M + Q4_0 + fit 768  ≈ 59.4 tok/s
Q3_K_M + Q5_1 + fit 512  ≈ 59.5 tok/s
```

So lowering fit reserve from 768 → 512 MiB freed enough VRAM for better tensor placement without requiring lower-precision Q4 KV.

---

# Prompt Processing vs Generation

Do not mix these metrics.

Example:

```text
prompt processing:
  14986 tokens
  25.88 s
  ~579 tok/s
```

This is prefill.

Generation (`tg`) is autoregressive output:

```text
~59 tok/s single stream
```

Prefill is much faster because it is batch-parallel.

---

# Prompt Cache / Prefix Reuse

llama.cpp prompt reuse is working and can be extremely effective.

Example:

```text
f_sim_best = 0.976
f_keep = 1.000
```

A ~25K-token context only required:

```text
624 prompt tokens actually evaluated
```

So prefix/KV reuse should be preserved and measured.

Useful log fields:

```text
selected slot by LCP similarity
f_sim_best
f_keep
prompt eval time
prompt processing
```

`graphs reused` is NOT equivalent to prompt/KV cache reuse. It refers to computation graph reuse.

For unified-KV / multi-slot experiments, `CACHE_REUSE=0` has been used deliberately to avoid known/possible Qwen3.6 prompt-cache/tool-call pathologies while retaining ordinary prompt caching.

---

# Parallelism Findings

With two active decode streams:

```text
single stream:
  ~59.5 tok/s

two simultaneous streams:
  ~40–43 tok/s each
  ~80–85 tok/s aggregate
```

So continuous batching improves **aggregate throughput** by roughly 40%, but individual latency worsens.

GPU utilization/power may remain around:

```text
~67–75% GPU
~79–86 W
```

This is not paradoxical. The batching can make each busy GPU period more efficient without raising the sampled utilization percentage much.

## Critical latency problem: prefill starvation

A pathological case was observed with `PARALLEL=2` and a large prompt prefill in one slot.

Example:

```text
slot 1:
  prompt processing ~18.8K tokens
  ~580–640 tok/s

slot 0:
  existing decode tg_3s collapsed to ~0.28–0.31 tok/s
```

The long prefill arrived in chunks of about 2047 tokens, consistent with `BATCH=2048`.

This means aggregate server throughput can look good while end-to-end agent latency becomes much worse.

A benchmark that previously took about:

```text
~44 s
```

was observed at:

```text
2m10s
```

under a bad overlapping/prefill interaction.

Potential experiment not yet fully completed:

```text
BATCH=512
UBATCH=256
```

or:

```text
BATCH=256
UBATCH=256
```

to allow the scheduler to return to active decode streams more frequently.

Do NOT assume larger batch is always better for interactive latency.

---

# Important Client-Side Observation

Even when the upstream benchmark was configured with only **one top-level client job**, llama.cpp still showed two active model requests:

```text
slot 1: long-running task
slot 0: repeated short tasks
```

Therefore:

```text
1 top-level benchmark job != necessarily 1 llama.cpp request
```

The upstream harness/OpenCode can overlap provider calls internally.

This distinction must be visible in the benchmark reports.

Metrics should include:

```text
max concurrent llama.cpp requests
number of server task IDs
number of slots actually used
number of cancelled requests
```

---

# End-to-End Benchmark Observation

With `THREADS=6`, caller top-level concurrency=1, and the current Q3_K_M configuration, one real benchmark completed:

```text
PASS
took 54.3s
turns: 1
out_tokens: 0
```

Historic comparable result was around:

```text
~44s
```

The difference is currently considered within likely run-to-run/configuration variance. Repeated runs and medians are needed.

Do not optimize based on one run.

Use at least:

```text
7 repetitions per configuration
```

and compare medians.

---

# CPU Thread Findings

Thread counts currently under investigation:

```text
6
8
12
```

`THREADS=6` is interesting because the 5900X has 6 physical cores per CCD.

So far, 6 and 8 threads do not appear dramatically different.

Do not assume SMT-heavy values are beneficial.

The experiment should keep everything else fixed and compare:

```text
THREADS=6
THREADS=8
THREADS=12
```

Primary comparison: median end-to-end benchmark latency.

Secondary:
- generated-token throughput
- prompt throughput
- CPU memory traffic
- GPU utilization/power
- cancellations
- concurrent requests
- cache reuse

---

# DDR4 Profiling

AMD uProf works.

Command shape:

```bash
sudo AMDuProfPcm \
  -m memory \
  -a \
  -A system,package \
  -s \
  -o /path/to/memory.csv
```

Observed active inference traffic is roughly:

```text
~9–12.5 GB/s total DRAM bandwidth
~8.5–12.0 GB/s reads
usually <1 GB/s writes
peak read roughly ~12.8 GB/s
```

Channels are almost perfectly balanced.

Example:

```text
11.91 GB/s read total
  Ch A ≈ 5.95
  Ch B ≈ 5.96
```

Conclusion:

```text
DDR4 bandwidth is NOT saturated.
```

The remaining CPU-offload bottleneck is likely more about:
- CPU expert execution latency
- CPU/GPU dependency boundaries
- synchronization
- small transfers
- cache/CCD effects

rather than exhausting raw DDR bandwidth.

---

# PCIe Measurements

`amd-smi` does not expose live PCIe BW on this stack:

```text
BANDWIDTH: N/A
CURRENT_BANDWIDTH_SENT: N/A
CURRENT_BANDWIDTH_RECEIVED: N/A
```

However the kernel `amdgpu_benchmark` debugfs benchmark works.

GPU DRM debug path:

```text
/sys/kernel/debug/dri/1/amdgpu_benchmark
```

## GTT → VRAM

Large transfers:

```text
4 MiB   ~21.8 GB/s
8 MiB   ~24.5 GB/s
16 MiB  ~26.1 GB/s
32 MiB  ~26.9 GB/s
64 MiB  ~26.6 GB/s
```

## VRAM → GTT

Large transfers:

```text
4 MiB   ~18.5 GB/s
8 MiB   ~21.3 GB/s
16 MiB  ~23.1 GB/s
32 MiB  ~23.4 GB/s
64 MiB  ~24.1 GB/s
```

Conclusion:

```text
PCIe 4.0 x16 bulk transfer performance is healthy.
```

Do not spend engineering effort treating gross PCIe link throughput as the main bottleneck unless new evidence appears.

---

# AMD GPU Profiling

Use `amd-smi monitor` and capture machine-readable output where possible.

Desired fields:

```text
timestamp
GPU utilization / GFX activity
memory activity if available
power usage
VRAM usage
clocks if useful
```

Observed Q3_K_M fit512 behavior:

```text
GPU util ~70%
occasional spikes ~76%
power ~79–86 W
```

This is significantly below the GPU's high-power compute state and is consistent with dependency stalls / host-side MoE participation.

---

# llama.cpp Metrics

Enable:

```text
--metrics
```

Use:

```text
GET /metrics
```

Important metrics:

```text
llamacpp:tokens_predicted_total
llamacpp:prompt_tokens_total
llamacpp:requests_processing
llamacpp:requests_deferred
llamacpp:n_tokens_max
```

The instantaneous metric:

```text
llamacpp:predicted_tokens_seconds
```

has returned `0` in this environment despite active requests, so do not rely on it.

Compute aggregate generated tokens/sec from the delta of:

```text
llamacpp:tokens_predicted_total
```

over elapsed wall time.

Important distinction:

```text
llama per-task tg:
  speed while model task is actually decoding

wall generation tok/s:
  cumulative generated-token delta / full benchmark wall time
```

The latter includes tool gaps, prefills, orchestration, etc. Both are useful but mean different things.

---

# Reasoning / Qwen Chat Settings

Reasoning is enabled because Qwen3.6 behaved materially better at interpreting ambiguous coding intent with reasoning enabled.

Do not set a fixed positive host-level reasoning budget unless explicitly testing it.

The intent is to permit per-request reasoning budgets eventually.

Use Qwen-native historical-thinking behavior:

```json
{"preserve_thinking": false}
```

Historical completed-turn scratchpads should not grow indefinitely.

Do not casually replace this with llama.cpp's generic reasoning-preservation flags without testing tool-loop behavior.

---

# Existing Artifacts

Previous generated artifacts from the interactive work:

```text
Makefile.llamacpp-tuned-v2
llama_bench_harness.go
llama-bench-harness
```

The new implementation can replace the earlier Go harness if necessary. Treat the previous source as a prototype/specification, not sacred code.

The final implementation should be clean enough to keep in the llama.cpp source/workstation tooling repository.

---

# Required Go Harness Behavior

Implement a single executable, no shell orchestration required beyond the external commands it invokes.

Recommended binary name:

```text
llama-bench-harness
```

## Required lifecycle

For each experiment configuration and repetition:

1. Create a unique run directory.
2. Record environment + exact configuration.
3. Optionally run `llama-fit-params`.
4. Parse/store fitted placement.
5. Start `llama-server`.
6. Capture stdout/stderr and llama's own log file.
7. Poll `/health` until ready.
8. Start `AMDuProfPcm`.
9. Start `amd-smi monitor`.
10. Start a periodic `/metrics` sampler.
11. Start wall-clock timer.
12. Run the configured client benchmark.
13. Stop wall-clock timer.
14. Stop metrics sampler.
15. Gracefully stop uProf.
16. Stop AMD SMI.
17. Gracefully stop llama-server.
18. Parse all raw data.
19. Write per-run JSON.
20. Update cumulative CSV/Markdown summaries.
21. Continue to the next repetition/configuration.

Everything must also clean up correctly on:
- Ctrl+C
- SIGTERM
- client timeout
- client failure
- server crash
- profiler failure

Use process groups so child processes cannot leak.

---

# Configuration / CLI Requirements

The program should expose at least:

```text
--workdir
--out
--client-cmd
--runs
--threads
--server
--fit-binary
--model
--alias
--host
--port
--ctx
--parallel
--kv-unified
--kv-per-slot
--kv-k
--kv-v
--batch
--ubatch
--fit-target
--n-cpu-moe
--flash-attn
--cache-reuse
--reasoning
--preserve-thinking
--temp
--top-p
--top-k
--amd-smi
--uprof
--profile-gpu
--profile-memory
--client-timeout
--health-timeout
```

Support repeated:

```text
--server-arg
--server-env
--client-env
```

A config file (TOML/YAML/JSON) is optional but desirable if it makes experiment matrices cleaner.

---

# Experiment Matrix Support

The implementation should evolve beyond only `THREADS`.

At minimum support matrix axes for:

```text
threads
parallel
batch
ubatch
fit_target
kv_k
kv_v
model
preserve_thinking (prompt/session type)
```

Potential later axes:

```text
context
cache_reuse
flash_attention
```

The matrix runner should evaluate explicit axis values as a Cartesian product so interactions can be compared across settings. For each configuration and thread count, run warm-up first, then its counted repetitions on the same server session; restart the server only when a server setting or thread count changes.

The Go harness exposes Cartesian axes for thread count, context, parallel slots,
logical batch, microbatch, fit target, K/V cache types, model, preserve-thinking,
cache reuse, and FlashAttention. The `*-values` options add comma-separated
alternatives to the corresponding base setting; e.g. `--ctx-values 32768,65536`
keeps the base context and adds those values, deduplicating repeats. Preserve-
thinking remains a prompt/session-type axis. When expanding a study, include
only the dimensions and values that the study is intended to compare, and keep
the complete invocation in the run notes.

Prefer round-robin/interleaved execution order across configurations to reduce thermal/time bias, e.g.:

```text
6-thread run 1
8-thread run 1
12-thread run 1
6-thread run 2
8-thread run 2
12-thread run 2
...
```

rather than seven 6-thread runs followed by seven 8-thread runs.

---

# Required Raw Files Per Run

Use a structure similar to:

```text
benchmark-runs/
  environment.txt
  runs.csv
  summary.csv
  summary.md

  threads-06/
    fit.txt
    run-01/
      meta.json
      result.json
      client.log
      llama.log
      llama-console.log
      memory.csv
      gpu.csv
      metrics.csv
```

Never discard raw data merely because parsing failed.

---

# llama.cpp Log Parsing Requirements

Extract at least:

```text
number of unique task IDs
number of slots used
number of cancelled tasks

prompt tokens actually evaluated
prompt eval seconds
weighted prompt tok/s

generated tokens
eval seconds
weighted generated tok/s

tg_3s samples
minimum tg_3s
maximum tg_3s
mean tg_3s

max prompt-processing token count
large-prefill events

LCP/cache events:
  f_sim_best
  f_keep

server task overlap / concurrent slot activity where derivable
```

Detect pathological cases such as:

```text
an active decode stream tg_3s collapsing below a threshold
while another slot is doing a long prefill
```

This is a critical latency regression signal.

Suggested output field:

```text
decode_starvation_events
```

with threshold configurable.

---

# Client Log Parsing

The current benchmark emits lines such as:

```text
==> [Target=opencode/amdbeast/qwen36-coder Task=generate_template_main Variant=small:crypto_rand Arm=baseline-diff] PASS (took 54.3s, turns: 1, out_tokens: 0)
```

Parse:

```text
target
task
variant
arm
PASS/FAIL
reported duration
turn count
out_tokens
```

Do not rely only on this reported duration. Also measure process wall-clock time independently.

---

# uProf Parsing Requirements

Parse:

```text
Total Mem Bw (GB/s)
Total Mem RdBw (GB/s)
Total Mem WrBw (GB/s)

Mem Ch-A RdBw
Mem Ch-A WrBw
Mem Ch-B RdBw
Mem Ch-B WrBw
```

Report at least:

```text
sample count
mean
median
p95
max
```

for:
- total
- read
- write

Also report channel imbalance where useful:

```text
abs(chA_read - chB_read)
```

---

# AMD SMI Parsing Requirements

AMD SMI output format may vary by version. Make the parser tolerant of header-name variations.

Desired metrics:

```text
GFX/GPU utilization
memory activity
power W
VRAM used
clock(s), if stable to parse
```

Report:

```text
mean
median
p95
max
```

Do not fail the run if AMD SMI lacks a specific counter.

PCIe bandwidth is expected to be unavailable on this setup.

---

# Metrics Endpoint Parsing

Sample `/metrics` once per second during the client run.

At minimum record:

```text
timestamp
tokens_predicted_total
prompt_tokens_total
prompt_seconds_total
tokens_predicted_seconds_total
requests_processing
requests_deferred
n_tokens_max
```

Derived metrics:

```text
wall_generation_tps
wall_prompt_token_rate
max_requests_processing
max_requests_deferred
max_context_observed
```

Do not use `predicted_tokens_seconds` as the authoritative generation rate.

---

# Summary Statistics

Per configuration, report:

```text
runs
successful runs
failed runs

wall time:
  min
  median
  mean
  p95
  max

client-reported duration:
  median

llama decode:
  median weighted tg

wall generated-token rate:
  median

prompt processing rate:
  median

cancelled task count:
  median / total

DDR read GB/s:
  median
  p95
  max

GPU utilization:
  median
  p95
  max

GPU power:
  median
  p95
  max

fit:
  number of CPU-spilled blocks
  exact fitter output
```

Primary sorting/decision metric:

```text
median end-to-end wall time
```

not maximum aggregate tokens/sec.

---

# Fairness / Reproducibility Requirements

For fair comparisons:

- Restart llama-server between repetitions by default.
- Start client timing only after `/health` is ready.
- Preserve raw logs.
- Record current llama.cpp git commit.
- Record kernel/ROCm/AMD-SMI/uProf versions.
- Record exact server command.
- Record exact client command.
- Record exact environment variables.
- Record model quant.
- Do not silently reuse server KV state across benchmark repetitions unless explicitly running a cache-reuse experiment.
- Permit an optional warm-up run that is excluded from statistics.
- Use the same benchmark task for all configurations.
- Consider randomized or round-robin config order.

---

# Acceptance Criteria

The remote implementation is complete when:

1. The Go program builds with standard Go tooling.
2. Normal operation does not require a wrapper shell script.
3. It can run:
   ```text
   THREADS = 6, 8, 12
   ```
   for at least 3 repetitions each.
4. It starts/stops llama-server reliably.
5. It captures the real client benchmark.
6. It captures uProf DDR bandwidth.
7. It captures AMD SMI GPU telemetry.
8. It captures `/metrics`.
9. It parses llama.cpp task/prompt/decode logs.
10. It detects cancelled tasks.
11. It records LCP/cache statistics.
12. It preserves raw files.
13. It produces:
    ```text
    runs.csv
    summary.csv
    summary.md
    per-run result.json
    ```
14. Ctrl+C leaves no llama-server/uProf/amd-smi child processes behind.
15. Failed profiling does not invalidate the client benchmark; the run should finish with profiler fields marked unavailable.
16. The summary clearly distinguishes:
    - per-task generation throughput
    - aggregate/wall generated-token throughput
    - end-to-end task latency
17. The primary comparison is median end-to-end task latency.
18. The implementation is clean enough to extend to model/KV/batch/parallel experiment matrices later.

---

# Immediate First Experiment After Implementation

Keep the model/server configuration fixed:

```text
Qwen3.6 UD-Q3_K_M
CTX=65536
KV=q5_1/q5_1
BATCH=2048
UBATCH=256
FIT_TARGET=512
PARALLEL=2
CACHE_REUSE=0
unified KV
per-slot max=65536
reasoning on
preserve_thinking=false
```

Run:

```text
THREADS=6
THREADS=8
THREADS=12
```

at least 7 times each.

Compare medians.

Then test `PARALLEL=1` vs `PARALLEL=2` because the upstream workload can issue overlapping provider calls even when it has only one top-level job.

Then test prefill latency protection:

```text
BATCH=2048
BATCH=512
BATCH=256
```

with `UBATCH=256`.

The key question is whether smaller logical batches prevent a long prompt prefill from starving an existing decode stream without making total task latency worse.

---

# Engineering Principle

Do not optimize for the metric that looks best in isolation.

The observed system can have:

```text
higher aggregate tok/s
```

while producing:

```text
much worse end-to-end coding-agent latency
```

because large prefills can starve existing decodes and the upstream harness can overlap multiple provider calls internally.

The benchmark harness exists to make those interactions visible and reproducible.

## Offline Benchmark Analysis

The harness can analyze existing `runs.csv` files without starting llama-server:

```sh
./bin/llama-bench-harness analyze --min-runs 7 --out benchmark-runs/analysis/report.md \
  benchmark-runs/complete-matrix benchmark-runs/full-matrix
```

Each input may be a `runs.csv` file or its containing output directory. Repeated
paths are deduplicated by per-run directory. The report ranks cells by median
end-to-end wall time, shows successful sample counts and throughput/token
medians, compares matched preserve-thinking on/off cells, and summarizes
thread/configuration coverage. `--min-runs` sets the sample count required for
a cell to qualify as the reported latency leader.
