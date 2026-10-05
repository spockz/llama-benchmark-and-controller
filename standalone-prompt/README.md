# Task 11 OpenCode baseline

This bundle provides a repeatable, single-turn OpenCode baseline for host-performance tuning. It contains only the small visible fixture; it does not expose the benchmark oracle and does not send interactive follow-ups.

Run it from this directory:

```sh
./run.sh
```

The script creates a fresh copy at `runs/task-11.*`, runs the exact initial baseline prompt with `amdbeast/qwen36-coder`, and prints that workspace path to standard error. It emits OpenCode's JSON event stream on standard output. Prefix the command with your usual timing tool when measuring wall time.

The launcher supplies `--auto`, fixture-local OpenCode and Go state, and `OPENCODE_DISABLE_PROJECT_CONFIG=1`. Its `opencode.json` deliberately has no `mcp` entry, so this is the normal-editing baseline rather than the semantic-editor arm. When run directly, it asks for the llama-server URL; set `LLAMA_BASE_URL` for non-interactive use, or `OPENCODE_CONFIG` to provide a complete custom config. The benchmark harness generates a per-run config pointed at the llama-server instance being measured and passes it through `OPENCODE_CONFIG`.
