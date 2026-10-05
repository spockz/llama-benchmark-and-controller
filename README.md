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

## Build llama.cpp

The harness does not build llama.cpp itself. The repository includes a helper
Makefile at [`Makefile-llama`](Makefile-llama) with separate Vulkan and ROCm
builds. Copy it from this repository into the llama.cpp checkout under its own
name so the upstream `Makefile` stays untouched:

```sh
cp /path/to/tune-llama-cpp/Makefile-llama /path/to/llama.cpp/Makefile-llama
cd /path/to/llama.cpp
make -f Makefile-llama build-vulkan
make -f Makefile-llama build-rocm GPU_TARGET=gfx1201
```

Server targets ask which host/IP to bind to when run interactively. For
non-interactive use, pass `HOST="<your-ip>"` explicitly.

Use `make -f Makefile-llama build-all` to build both backends. The Makefile
defaults to `JOBS=12`; override it with `JOBS=N` to match the machine. Vulkan
builds into `build-vulkan/`, while ROCm builds into `build-rocm/`.

The ROCm Makefile setting `GPU_TARGET` defaults to `gfx1201`, the AMD GPU
architecture used for this machine. Check the architecture reported by
`rocminfo` and replace that value, either in the copied Makefile or on the make
command line, for the GPU in your system. The correct target is the GPU's
`gfx` architecture identifier, not its marketing name. Vulkan builds do not
use `GPU_TARGET`.

The harness defaults to the Vulkan binaries at
`build-vulkan/bin/llama-server` and `build-vulkan/bin/llama-fit-params`. To use
the ROCm build, add `--server ./build-rocm/bin/llama-server` and
`--fit-binary ./build-rocm/bin/llama-fit-params` to the harness command. These
paths are resolved relative to `--workdir`.

## Host and switch between the two models

The router service keeps one `llama-server` process on port 1234 and loads at
most one model at a time. The example presets configure Qwen3.6-35B-A3B
`UD-Q3_K_M` and `UD-IQ4_XS` at 128K context. The Q3 preset uses its existing
candidate settings (6 threads, batch 2048, ubatch 256, fit target 512 MiB,
Q5_1 K/V); the IQ4_XS preset uses 8 threads, batch/ubatch 512 and fit target
256 MiB. `n-gpu-layers = auto` and `fit = on` let the installed build place as
much as it can on the GPU. The Q3_K_M fit was only measured at 64K and spilled
some expert tensors to CPU RAM even there. At 128K it may spill more, so this
is a requested 128K configuration, not a verified all-GPU fit or a measured
performance result. Check the server log after startup; do not run fitting
while another model is using the GPU.

Copy [`models.ini.example`](models.ini.example) to
`~/.config/llama-modelctl/models.ini`. Replace `USER` in both
`slot-save-path` values with the account running llama-server, then create the
directories. Slot files live on the server host and must remain accessible to
llama-server:

```sh
install -Dm600 models.ini.example "$HOME/.config/llama-modelctl/models.ini"
mkdir -p "$HOME/.local/share/llama-modelctl/slots/q3-k-m" \
  "$HOME/.local/share/llama-modelctl/slots/iq4-xs"
```

Edit the two named sections to use local `model = /path/to/model.gguf` values
or a supported `hf = repository:quantization` source. Model IDs are the INI
section names (`qwen36-q3-k-m` and `qwen36-iq4-xs`). Tune each section
independently. Keep a trailing slash on each `slot-save-path`.

Install the router unit from [`systemd/llama-model-router.service`](systemd/llama-model-router.service),
adjust its llama.cpp path and GPU identifier for the serving machine, then set
the bind address in a server environment file and enable it:

```sh
install -Dm644 systemd/llama-model-router.service \
  "$HOME/.config/systemd/user/llama-model-router.service"
install -Dm600 systemd/server.env.example \
  "$HOME/.config/llama-modelctl/server.env"
systemctl --user daemon-reload
systemctl --user enable --now llama-model-router.service
systemctl --user status llama-model-router.service
```

Edit `~/.config/llama-modelctl/server.env` and set `LLAMA_SERVER_HOST` to
`<your-ip>` before starting the service.

The unit assumes the Vulkan build at `~/sources/llama.cpp` and GPU device 0.
For boot without an interactive login, enable user lingering with
`loginctl enable-linger "$USER"`. Do not run the older single-model unit on
port 1234 at the same time. Protect the management API with a trusted
network/firewall or SSH tunnel. If llama-server is configured with an API key,
pass that key to the controller as `--token` or `LLAMA_SERVER_TOKEN`.

Build the companion controller with `make build-modelctl` (or `make build`);
it produces `bin/llama-modelctl`. It only uses Go's standard library and can be
built for another machine with that machine's Go toolchain. To install the
latest version directly from GitHub, install Go and run:

```sh
go install github.com/spockz/llama-benchmark-and-controller/cmd/llama-modelctl@latest
```

This places `llama-modelctl` in `$(go env GOPATH)/bin` (often `$HOME/go/bin`);
add that directory to `PATH` if needed. Then use the installed command as in
these examples:

```sh
llama-modelctl --host "<your-ip>" --port 1234 --preset ./models.ini status
llama-modelctl --server "http://<your-ip>:1234" --preset ./models.ini load qwen36-q3-k-m
llama-modelctl --host "<your-ip>" --preset ./models.ini switch qwen36-iq4-xs
llama-modelctl --host "<your-ip>" --preset ./models.ini config qwen36-iq4-xs
llama-modelctl --host "<your-ip>" --preset ./models.ini save
llama-modelctl --host "<your-ip>" --preset ./models.ini unload
```

`unload` is the blanket GPU-release command: it saves each active model's cache,
waits for slot 0 to become idle, unloads every loaded or sleeping router model,
and verifies that none remain. For example, run
`llama-modelctl --host "<your-ip>" unload` before using the GPU for another
task. `switch` and `unload` recheck slot 0 before unloading and wait for each
transition. Pause new inference submissions while switching; the server API has no
atomic lock shared with ordinary generation requests. A failed save stops the
operation; `--discard-cache` is an explicit override. `switch`
restores a cache only when the manifest matches model identity/hash, adapter,
context/RoPE, K/V types, full preset settings and llama.cpp build. For models
whose files are only on the server, give the controller a trusted digest with
`--model-sha256 model-id=HEX` or use an HF snapshot revision it can identify;
uncertain caches are saved but never restored. Use `--session` to keep
independent conversations' slot files separate.

The controller keeps manifests and its exclusive lock in `--state-dir`. For
multiple remote clients to coordinate, point them all at the same shared state
directory with working file locks. Otherwise run the controller on the server
host (for example over SSH), where its default lock is shared by invocations.
Keep the cache manifest directory available to whichever client performs a
later restore. The slot cache only speeds up matching-prefix evaluation:
inference clients must continue sending full prompt/history, set `id_slot: 0`,
and enable prompt reuse. `--cache-ram` is an in-memory cache and does not
survive model process unloads.

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

## Machine and fastest measured setup

The benchmark workstation has an AMD Ryzen 9 5900X (12 cores / 24 threads),
64 GiB dual-channel DDR4-4000, and an AMD Radeon RX 9070 XT with 16 GiB VRAM
(RDNA4, `gfx1201`).

The fastest completed end-to-end setup recorded so far used parallelism 2,
batch 512, 8 CPU threads, and preserve-thinking off. Its median task wall time
was 80.6 s, with 47.1 tok/s whole-server generation rate, 46.9 tok/s weighted
task decode rate, and 96,602 median client prompt tokens. This is an earlier
measured result, not the winner of the IQ4_NL vs IQ4_XS full-grid matrix above;
use that matrix's completed analysis to identify its winner.

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
