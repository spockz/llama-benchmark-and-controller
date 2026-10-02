#!/usr/bin/env bash
set -euo pipefail

repo=/home/alessandro/sources/tune-llama-cpp
out="$repo/benchmark-runs/iq4-nl-vs-xs-full-grid-20260929"
cd "$repo"
export PATH=/home/alessandro/go/bin:/home/alessandro/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/usr/bin/site_perl:/usr/bin/vendor_perl:/usr/lib/jvm/default/bin
mkdir -p "$out"
exec >>"$out/screen.log" 2>&1

args=( \
  --workdir /home/alessandro/sources/llama.cpp \
  --out "$out" \
  --model 'unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL' \
  --model-values 'unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL,unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS' \
  --ctx 65536 --ctx-values 65536,131072,204800 \
  --threads 6,8,12 \
  --parallel 1 --parallel-values 1,2 \
  --batch 256 --batch-values 256,512,2048 \
  --ubatch 512 \
  --fit-target 256 --fit-target-values 256 \
  --kv-k q5_1 --kv-k-values q5_1 \
  --kv-v q5_1 --kv-v-values q5_1 \
  --kv-per-slot 204800 --kv-unified \
  --cache-reuse 0 --cache-reuse-values 0 \
  --flash-attn on --flash-attn-values on \
  --preserve-thinking=false --preserve-thinking-values false \
  --runs 7 --warmup-runs 1 --client-timeout 60m --resume --fail-fast \
  --port 1234 --amd-smi /opt/rocm/bin/amd-smi --profile-memory=false \
)

status=1
for attempt in {1..20}; do
  if "$repo/bin/llama-bench-harness" "${args[@]}"; then
    if "$repo/bin/llama-bench-harness" analyze \
      --min-runs 7 \
      --matrix-manifest "$out/matrix.json" \
      --out "$out/analysis.md" \
      "$out/runs.csv"; then
      exit 0
    else
      status=$?
      printf '\nExact matrix audit failed (attempt %d/20); resuming in 10 seconds.\n' "$attempt" >&2
    fi
  else
    status=$?
    printf '\nHarness exited with status %d (attempt %d/20); resuming in 10 seconds.\n' "$status" "$attempt" >&2
  fi
  if (( attempt < 20 )); then
    sleep 10
  fi
done

printf '\nHarness still exits with status %d after 20 attempts; stopping for manual review.\n' "$status" >&2
exit "$status"
