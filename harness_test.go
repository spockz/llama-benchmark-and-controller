// This file checks experiment expansion and parsers that define benchmark results.
package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFetchPromMetricsIncludesLabeledSamples(t *testing.T) {
	metrics, err := parsePromMetrics(strings.NewReader("llamacpp:requests_processing{slot=\"0\"} 1\nllamacpp:requests_processing{slot=\"1\"} 2\nllamacpp:n_tokens_max{slot=\"0\"} 4096\nllamacpp:n_tokens_max{slot=\"1\"} 2048\n"))
	if err != nil {
		t.Fatal(err)
	}
	if metrics["llamacpp:requests_processing"] != 3 || metrics["llamacpp:n_tokens_max"] != 4096 {
		t.Fatalf("unexpected labeled metric aggregation: %#v", metrics)
	}
}

func TestParseOpenCodeJSONStreamCapturesOutcomeAndUsage(t *testing.T) {
	stream := "" +
		`{"type":"text","part":{"type":"text","text":"Changes verified. DONE"}}` + "\n" +
		`{"type":"step_finish","part":{"type":"step-finish","reason":"tool-calls","tokens":{"input":100,"output":20,"reasoning":5,"cache":{"read":300,"write":0}}}}` + "\n" +
		`{"type":"step_finish","part":{"type":"step-finish","reason":"stop","tokens":{"input":80,"output":10,"reasoning":2,"cache":{"read":20,"write":0}}}}` + "\n"
	got := parseOpenCodeEvents(stream)
	if !got.Parsed || !got.Passed || got.Turns != 2 || got.OutTokens != 30 || got.PromptTokens != 500 || got.ReasoningTokens != 7 {
		t.Fatalf("OpenCode event stream not parsed: %#v", got)
	}
}

func TestParseOpenCodeJSONStreamAcceptsNormalStopWithoutMarker(t *testing.T) {
	stream := "" +
		`{"type":"text","part":{"type":"text","text":"All tests pass. Changes are complete."}}` + "\n" +
		`{"type":"step_finish","part":{"type":"step-finish","reason":"stop","tokens":{"input":80,"output":10}}}` + "\n"
	got := parseOpenCodeEvents(stream)
	if !got.Parsed || !got.Passed {
		t.Fatalf("normal OpenCode completion without a literal marker should pass: %#v", got)
	}
}

func TestWriteRunOpenCodeConfigOverridesOnlyProviderEndpoint(t *testing.T) {
	template := filepath.Join("standalone-prompt", "opencode.json")
	output := filepath.Join(t.TempDir(), "opencode.json")
	if err := writeRunOpenCodeConfig(template, output, "http://127.0.0.1:8080/v1", 131072); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}
	provider := parsed["provider"].(map[string]any)["amdbeast"].(map[string]any)
	options := provider["options"].(map[string]any)
	if options["baseURL"] != "http://127.0.0.1:8080/v1" {
		t.Fatalf("generated config has wrong endpoint: %#v", options)
	}
	model, ok := provider["models"].(map[string]any)["qwen36-coder"].(map[string]any)
	if !ok {
		t.Fatal("generated config dropped the benchmark model alias")
	}
	limit, ok := model["limit"].(map[string]any)
	if !ok || limit["context"] != float64(131072) || limit["output"] != float64(16384) {
		t.Fatalf("generated config dropped model token limits: %#v", model["limit"])
	}
	compaction, ok := parsed["compaction"].(map[string]any)
	if !ok || compaction["auto"] != true || compaction["reserved"] != float64(10000) {
		t.Fatalf("generated config dropped automatic compaction settings: %#v", parsed["compaction"])
	}
}

func TestExpandExperimentsUsesCartesianProduct(t *testing.T) {
	base := config{
		Model: "model-a", Context: 65536, Parallel: 2, KVK: "q5_1", KVV: "q5_1",
		Batch: 2048, UBatch: 256, FitTarget: 512, KVPerSlot: 65536,
		ParallelValues: "1,2,4", KVKValues: "q4_0",
	}
	experiments, err := expandExperiments(base)
	if err != nil {
		t.Fatal(err)
	}
	if len(experiments) != 6 {
		t.Fatalf("got %d experiments, want 3 parallel values x 2 K-cache values", len(experiments))
	}
	if experiments[0].Name != "base" || experiments[0].Config.Parallel != 2 {
		t.Fatalf("base configuration changed: %#v", experiments[0])
	}
	pairs := make(map[string]bool)
	for _, experiment := range experiments {
		key := fmt.Sprintf("%d/%s", experiment.Config.Parallel, experiment.Config.KVK)
		pairs[key] = true
	}
	for _, parallel := range []int{1, 2, 4} {
		for _, kvk := range []string{"q5_1", "q4_0"} {
			if !pairs[fmt.Sprintf("%d/%s", parallel, kvk)] {
				t.Errorf("missing parallel=%d kv-k=%s combination", parallel, kvk)
			}
		}
	}
}

func TestExpandCoveragePlanGroupsThreadCellsByConfiguration(t *testing.T) {
	plan := []coverageCell{
		{Settings: analysisSettings{Model: "model-a", Context: 65536, Parallel: 2, Batch: 512, UBatch: 256, FitTarget: 512, CacheReuse: 0, FlashAttention: "on", UnifiedKV: true, KVPerSlot: 65536, KVK: "q5_1", KVV: "q5_1"}, Threads: 6},
		{Settings: analysisSettings{Model: "model-a", Context: 65536, Parallel: 2, Batch: 512, UBatch: 256, FitTarget: 512, CacheReuse: 0, FlashAttention: "on", UnifiedKV: true, KVPerSlot: 65536, KVK: "q5_1", KVV: "q5_1"}, Threads: 8},
		{Settings: analysisSettings{Model: "model-b", Context: 131072, Parallel: 1, Batch: 256, UBatch: 128, FitTarget: 256, CacheReuse: 0, FlashAttention: "on", UnifiedKV: true, KVPerSlot: 204800, KVK: "q4_0", KVV: "q5_1"}, Threads: 12},
	}
	data, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "plan.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	experiments, err := expandExperiments(config{PlanFile: path, Model: "model-a", Context: 65536, Parallel: 2, Batch: 512, UBatch: 256, FitTarget: 512, KVPerSlot: 65536, KVK: "q5_1", KVV: "q5_1", FlashAttention: "on", UnifiedKV: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(experiments) != 2 {
		t.Fatalf("got %d configurations, want 2", len(experiments))
	}
	if len(experiments[0].Threads) != 2 || experiments[0].Threads[0] != 6 || experiments[0].Threads[1] != 8 {
		t.Fatalf("thread rows were not grouped: %#v", experiments[0].Threads)
	}
	if experiments[1].Config.Context != 131072 || experiments[1].Config.Parallel != 1 || experiments[1].Config.Model != "model-b" {
		t.Fatalf("planned settings were not applied: %#v", experiments[1].Config)
	}
}

func TestResumeLoadsOnlySuccessfulMatchingRuns(t *testing.T) {
	root := t.TempDir()
	exp := experiment{Name: "base", Key: "settings-a"}
	write := func(rep, exitCode int, key string) {
		t.Helper()
		result := runResult{
			ConfigKey: key, ThreadCount: 8, Repetition: rep,
			Warmup: rep < 0, ExitCode: exitCode,
			Client: clientStats{Parsed: true, Passed: true},
		}
		path := resultPath(root, exp, 8, rep)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		data, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(1, 0, exp.Key)
	write(2, -1, exp.Key)
	write(3, 0, "different-settings")
	write(-1, 0, exp.Key)

	results, err := loadResumableResults(root, []experiment{exp}, []int{8}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Repetition != 1 {
		t.Fatalf("resumable results = %#v; want only successful matching run 1", results)
	}
	if cellComplete(root, exp, 8, 3, 1) {
		t.Fatal("cellComplete() = true with failed and mismatched counted runs")
	}
}

func TestExpandExperimentsIncludesContextCacheAndFlashAxes(t *testing.T) {
	base := config{
		Model: "model-a", Context: 65536, Parallel: 2, KVK: "q5_1", KVV: "q5_1",
		Batch: 2048, UBatch: 256, FitTarget: 512, KVPerSlot: 65536,
		CacheReuse: 0, FlashAttention: "on",
		ContextValues: "32768", CacheReuseValues: "256", FlashAttentionValues: "off",
	}
	experiments, err := expandExperiments(base)
	if err != nil {
		t.Fatal(err)
	}
	if len(experiments) != 8 {
		t.Fatalf("got %d experiments, want 2^3 combinations", len(experiments))
	}
	pairs := make(map[string]bool)
	for _, experiment := range experiments {
		cfg := experiment.Config
		pairs[fmt.Sprintf("%d/%d/%s", cfg.Context, cfg.CacheReuse, cfg.FlashAttention)] = true
	}
	for _, ctx := range []int{65536, 32768} {
		for _, cacheReuse := range []int{0, 256} {
			for _, flash := range []string{"on", "off"} {
				key := fmt.Sprintf("%d/%d/%s", ctx, cacheReuse, flash)
				if !pairs[key] {
					t.Errorf("missing context/cache-reuse/flash-attn combination %s", key)
				}
			}
		}
	}
	keys := make(map[string]bool)
	for _, experiment := range experiments {
		if keys[experiment.Key] {
			t.Errorf("duplicate configuration key %q", experiment.Key)
		}
		keys[experiment.Key] = true
	}
}

func TestCacheReuseAxisRejectsNegativeValues(t *testing.T) {
	base := config{
		Model: "model-a", Context: 65536, Parallel: 2, KVK: "q5_1", KVV: "q5_1",
		Batch: 2048, UBatch: 256, FitTarget: 512, KVPerSlot: 65536,
		CacheReuseValues: "-1",
	}
	if _, err := expandExperiments(base); err == nil {
		t.Fatal("expected negative cache-reuse value to fail")
	}
}

func TestPreserveThinkingIsPromptSessionAxis(t *testing.T) {
	base := config{
		Model: "model-a", Context: 65536, Parallel: 2, KVK: "q5_1", KVV: "q5_1",
		Batch: 2048, UBatch: 256, FitTarget: 512, KVPerSlot: 65536,
		PreserveThinkValues: "true",
	}
	experiments, err := expandExperiments(base)
	if err != nil {
		t.Fatal(err)
	}
	if len(experiments) != 2 || experiments[0].Config.PreserveThink || !experiments[1].Config.PreserveThink {
		t.Fatalf("preserve-thinking axis = %#v, want default and enabled sessions", experiments)
	}
	if experiments[1].Name != "preserve-thinking-on" || experiments[0].Key == experiments[1].Key {
		t.Fatalf("preserve-thinking session identity not represented: %#v", experiments)
	}

	env := clientEnvironment(base, 8, 1, "/tmp/run", "/tmp/run/opencode.json")
	if !containsString(env, "BENCH_PRESERVE_THINKING=false") || !containsString(env, "BENCH_PROMPT_SESSION_TYPE=default") {
		t.Fatalf("default prompt/session type not passed to client: %v", env)
	}
	base.PreserveThink = true
	env = clientEnvironment(base, 8, 1, "/tmp/run", "/tmp/run/opencode.json")
	if !containsString(env, "BENCH_PRESERVE_THINKING=true") || !containsString(env, "BENCH_PROMPT_SESSION_TYPE=preserve-thinking") {
		t.Fatalf("preserve-thinking prompt/session type not passed to client: %v", env)
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

func TestCopyLogRangeCapturesOnlyOneRun(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "server.log")
	destination := filepath.Join(dir, "run", "llama.log")
	if err := os.MkdirAll(filepath.Dir(destination), 0o700); err != nil {
		t.Fatal(err)
	}
	log := "startup\nrun-one\nrun-two\nshutdown\n"
	if err := os.WriteFile(source, []byte(log), 0o600); err != nil {
		t.Fatal(err)
	}
	start := int64(len("startup\n"))
	end := start + int64(len("run-one\n"))
	if err := copyLogRange(source, destination, start, end); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(destination)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "run-one\n" {
		t.Fatalf("copied log slice = %q, want only run-one", got)
	}
}

func TestParseClientLogCapturesPassAndIdentity(t *testing.T) {
	path := filepath.Join(t.TempDir(), "client.log")
	line := "==> [Target=opencode/amdbeast/qwen36-coder Task=generate_template_main Variant=small:crypto_rand Arm=baseline-diff] PASS (took 54.3s, turns: 1, out_tokens: 12)\n"
	if err := os.WriteFile(path, []byte(line), 0o600); err != nil {
		t.Fatal(err)
	}
	got := parseClientLog(path)
	if !got.Parsed || !got.Passed || got.Target != "opencode/amdbeast/qwen36-coder" || got.Task != "generate_template_main" {
		t.Fatalf("client result identity/status not parsed: %#v", got)
	}
	if got.Variant != "small:crypto_rand" || got.Arm != "baseline-diff" || got.ReportedSeconds != 54.3 || got.Turns != 1 || got.OutTokens != 12 {
		t.Fatalf("client result fields not parsed: %#v", got)
	}
}

func TestParseLlamaLogSeparatesPromptAndDecode(t *testing.T) {
	path := filepath.Join(t.TempDir(), "llama.log")
	log := "" +
		"slot launch_slot_: id 0 | task 12\n" +
		"slot launch_slot_: id 1 | task 13\n" +
		"task 12 | n_gen = 10, tg = 0.5 t/s, tg_3s = 0.5 t/s\n" +
		"cancel task, id_task = 13\n" +
		"0.01.28.248 W srv          stop: cancel task, id_task = 14\n" +
		"task 12 | prompt eval time = 2000.00 ms / 100 tokens (50.00 tokens per second)\n" +
		"task 12 | eval time = 1000.00 ms / 20 tokens (20.00 tokens per second)\n" +
		"task 12 | n_gen = 20, tg = 20.0 t/s, tg_3s = 18.0 t/s\n" +
		"task 12 | prompt processing, n_tokens = 100\n" +
		"f_sim_best = 0.976, f_keep = 1.000\n" +
		""
	if err := os.WriteFile(path, []byte(log), 0o600); err != nil {
		t.Fatal(err)
	}
	got := parseLlamaLog(path, 1, 90)
	if got.TaskCount != 2 || got.SlotsUsed != 2 || got.CancelledTasks != 1 {
		t.Fatalf("task summary incorrect: %#v", got)
	}
	if got.PromptTokens != 100 || got.PromptSeconds != 2 || got.GeneratedTokens != 20 || got.EvalSecondsSummed != 1 {
		t.Fatalf("prompt and decode totals mixed or incorrect: %#v", got)
	}
	if got.PromptTPSWeighted != 50 || got.TaskTGWeighted != 20 || got.LCPCount != 1 || got.MaxPromptProcessingTokens != 100 {
		t.Fatalf("derived llama metrics incorrect: %#v", got)
	}
	if got.MaxConcurrentTasks != 2 || got.LargePrefillEvents != 1 || got.DecodeStarvationEvents != 1 {
		t.Fatalf("overlap and latency regression signals incorrect: %#v", got)
	}
}

func TestStatsOfUsesNearestRankP95(t *testing.T) {
	got := statsOf([]float64{5, 1, 3, 2, 4})
	if got.Count != 5 || got.Mean != 3 || got.Median != 3 || got.P95 != 5 || got.Min != 1 || got.Max != 5 {
		t.Fatalf("unexpected statistics: %#v", got)
	}
}

func TestCountFitSpillBlocksCountsOnlyCPUPlacement(t *testing.T) {
	fit := "blk.35 ffn_down = CPU\nblk.36 expert = CPU\nblk.37 ffn = GPU\nblk.35 another = CPU\n"
	if got := countFitSpillBlocks(fit); got != 2 {
		t.Fatalf("got %d CPU-spilled blocks, want 2", got)
	}
}

func TestParseGPULogAcceptsShortAMDHeadersAndUnits(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.csv")
	csv := "GPU,GFX,MEM,POWER (W),VRAM USED (MB)\n0,70%,35%,82 W,8000 / 16384 MB\n"
	if err := os.WriteFile(path, []byte(csv), 0o600); err != nil {
		t.Fatal(err)
	}
	got := parseGPULog(path)
	if got.Samples != 1 || got.MeanUtil != 70 || got.MeanMemUtil != 35 || got.MeanPower != 82 || got.MeanVRAMUsed != 8000 {
		t.Fatalf("GPU counters not parsed: %#v", got)
	}
}

func TestParseMemoryLogReportsDistributionStatistics(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.csv")
	data := "Total Mem Bw (GB/s),Total Mem RdBw (GB/s),Total Mem WrBw (GB/s)\n10,8,2\n12,10,2\n"
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	got := parseMemoryLog(path)
	if got.Samples != 2 || got.MeanTotal != 11 || got.MedianRead != 9 || got.P95Read != 10 || got.MaxRead != 10 {
		t.Fatalf("DDR metrics not parsed or summarized: %#v", got)
	}
}

func TestSummaryCSVRowsMatchTheirHeaders(t *testing.T) {
	dir := t.TempDir()
	result := runResult{
		ConfigName: "base", ConfigKey: "model|ctx", ThreadCount: 6, Repetition: 1,
		ExitCode: 0, WallSeconds: 12.5,
		Llama:  llamaStats{TaskTGWeighted: 30, CancelledTasks: 1, DecodeStarvationEvents: 2},
		Memory: memoryStats{MeanRead: 8, MaxRead: 10, P95Read: 9.5, ReadValues: []float64{8, 10}},
		GPU:    gpuStats{MeanUtil: 65, MaxUtil: 70, P95Util: 69, UtilValues: []float64{60, 70}},
	}
	if err := writeAllSummaries(dir, []runResult{result}); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"runs.csv", "summary.csv"} {
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		reader := csv.NewReader(f)
		header, err := reader.Read()
		if err != nil {
			_ = f.Close()
			t.Fatal(err)
		}
		row, err := reader.Read()
		if err != nil {
			_ = f.Close()
			t.Fatal(err)
		}
		if len(row) != len(header) {
			_ = f.Close()
			t.Fatalf("%s has %d values for %d columns", name, len(row), len(header))
		}
		if _, err := reader.Read(); err != io.EOF {
			_ = f.Close()
			t.Fatalf("%s contains unexpected extra rows: %v", name, err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestAggregateResultsRehydratesGPUCountersForResumedRuns(t *testing.T) {
	runDir := t.TempDir()
	telemetry := "timestamp,gpu,gfx,mem,power_usage,vram_used\n" +
		"1,0,40,2,100,12000\n" +
		"2,0,60,4,200,14000\n"
	if err := os.WriteFile(filepath.Join(runDir, "gpu.csv"), []byte(telemetry), 0o644); err != nil {
		t.Fatal(err)
	}

	// This is the shape loaded from result.json: telemetry slices are omitted,
	// while per-run summary counters remain present.
	got := aggregateResults([]runResult{{
		ConfigName: "resumed", ConfigKey: "resumed-key", ThreadCount: 8,
		Repetition: 1, ExitCode: 0, RunDir: runDir,
		GPU: gpuStats{Samples: 2, MeanUtil: 50, MeanPower: 150},
	}})
	if len(got) != 1 {
		t.Fatalf("got %d aggregate rows, want 1", len(got))
	}
	if got[0].MedianGPUUtil != 50 || got[0].P95GPUUtil != 60 || got[0].MaxGPUUtil != 60 {
		t.Fatalf("resumed GPU utilization metrics were lost: %#v", got[0])
	}
	if got[0].MedianPower != 150 || got[0].P95Power != 200 || got[0].MaxPower != 200 {
		t.Fatalf("resumed GPU power metrics were lost: %#v", got[0])
	}
}
