#!/usr/bin/env bash
# Run one clean task-11 baseline trial with OpenCode while keeping host configuration outside the trial.
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fixture_dir="$bundle_dir/fixture"
runs_dir="$bundle_dir/runs"
mkdir -p "$runs_dir"
work_dir="$(mktemp -d "$runs_dir/task-11.XXXXXX")"
cp -R "$fixture_dir/." "$work_dir"

open_code_state="$work_dir/.scratch/opencode"
go_state="$work_dir/.scratch/go"
mkdir -p \
  "$open_code_state/home" \
  "$open_code_state/config" \
  "$open_code_state/data" \
  "$open_code_state/cache" \
  "$go_state/build" \
  "$go_state/mod" \
  "$go_state/tmp" \
  "$go_state/bin" \
  "$go_state/path"

export HOME="$open_code_state/home"
export XDG_CONFIG_HOME="$open_code_state/config"
export XDG_DATA_HOME="$open_code_state/data"
export XDG_CACHE_HOME="$open_code_state/cache"
export OPENCODE_CONFIG="${OPENCODE_CONFIG:-$bundle_dir/opencode.json}"
export OPENCODE_DISABLE_PROJECT_CONFIG=1
export GOENV="$go_state/env"
export GOCACHE="$go_state/build"
export GOMODCACHE="$go_state/mod"
export GOTMPDIR="$go_state/tmp"
export GOBIN="$go_state/bin"
export GOPATH="$go_state/path"
export GOWORK=off
export GOFLAGS=

prompt='Do not edit tests. You are forbidden to modify protected files: "go.mod", "go.sum".

Audit event kinds may contain inconsistent whitespace and casing. Normalize them at the delivery boundary before persistence, and treat probe events as control traffic. Do not change the metrics or legacy export protocols. Preserve the module metadata and tests, and leave the workspace verified. Do not use semantic editing MCP tools; use standard file editing. When done, output DONE.'

printf 'Workspace: %s\n' "$work_dir" >&2
exec "${OPENCODE_BIN:-opencode}" run \
  --format json \
  --auto \
  --dir "$work_dir" \
  --model amdbeast/qwen36-coder \
  "$prompt"
