package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const analysisCSVHeader = "configuration,threads,run,exit_code,wall_s,client_reported_s,client_prompt_tokens,client_reasoning_tokens,turns,out_tokens,tasks,slots_used,cancelled_tasks,max_concurrent_tasks,large_prefill_events,decode_starvation_events,prompt_tokens,prompt_s,prompt_tps,generated_tokens,summed_eval_s,task_tg_weighted,min_tg3s,metrics_wall_gen_tps,max_requests_processing,max_requests_deferred,max_context,mem_mean_read_gbs,mem_max_read_gbs,mem_mean_write_gbs,gpu_mean_util_pct,gpu_max_util_pct,gpu_mean_power_w,gpu_max_power_w,fit_spill_blocks,run_dir,error\n"

func TestAnalysisReportRanksCellsAndPairsPreserveThinking(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "runs.csv")
	rows := analysisCSVHeader + strings.Join([]string{
		"base,8,1,0,10,10,1000,0,5,100,3,2,0,2,1,0,100,1,100,50,2,25,5,5,2,0,1000,0,0,0,60,70,100,200,6,/tmp/base-1,",
		"base,8,2,0,12,12,1200,0,6,120,3,2,0,2,1,0,100,1,100,60,2,25,5,5,2,0,1000,0,0,0,60,70,100,200,6,/tmp/base-2,",
		"preserve-thinking-on,8,1,0,11,11,900,0,4,90,3,2,0,2,1,0,100,1,100,55,2,25,5,5,2,0,1000,0,0,0,60,70,100,200,6,/tmp/think-1,",
		"preserve-thinking-on,8,2,0,9,9,1100,0,4,95,3,2,0,2,1,0,100,1,100,55,2,25,5,5,2,0,1000,0,0,0,60,70,100,200,6,/tmp/think-2,",
	}, "\n") + "\n"
	if err := os.WriteFile(path, []byte(rows), 0o600); err != nil {
		t.Fatal(err)
	}
	results, err := readAnalysisInputs([]string{path, path})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 4 {
		t.Fatalf("deduplicated result count = %d, want 4", len(results))
	}
	report := renderAnalysisReport(results, 2)
	for _, expected := range []string{
		"Best observed cell by median end-to-end wall time: **preserve-thinking-on, 8 threads**",
		"| base | 8 | 2/2, 2/2 | 1100 → 1000 | -9.1% | 11.00 → 10.00 | -9.1% | 110 → 92 |",
		"| preserve-thinking-on | 8 | 2/2 | 10.00s |",
	} {
		if !strings.Contains(report, expected) {
			t.Errorf("report does not contain %q\n%s", expected, report)
		}
	}
}

func TestRunAnalysisCommandRequiresInputs(t *testing.T) {
	if err := runAnalysisCommand(nil, &strings.Builder{}); err == nil || !strings.Contains(err.Error(), "usage:") {
		t.Fatalf("runAnalysisCommand() error = %v, want usage error", err)
	}
}

func TestValidateAnalysisCoverageRequiresExpectedCellsAndMinimumSuccesses(t *testing.T) {
	results := []runResult{
		{ConfigName: "cell-a", ThreadCount: 6, Repetition: 1, ExitCode: 0},
		{ConfigName: "cell-a", ThreadCount: 6, Repetition: 2, ExitCode: 0},
		{ConfigName: "cell-b", ThreadCount: 8, Repetition: 1, ExitCode: 0},
		{ConfigName: "cell-b", ThreadCount: 8, Repetition: 2, ExitCode: 1},
	}
	if err := validateAnalysisCoverage(results, 2, 1); err != nil {
		t.Fatalf("complete at min-runs=1: %v", err)
	}
	if err := validateAnalysisCoverage(results, 3, 1); err == nil || !strings.Contains(err.Error(), "found 2") {
		t.Fatalf("wrong expected cell count error = %v", err)
	}
	if err := validateAnalysisCoverage(results, 2, 2); err == nil || !strings.Contains(err.Error(), "cell-b threads=8 has 1/2") {
		t.Fatalf("incomplete cell error = %v", err)
	}
}

func TestRunAnalysisCommandReportsCompleteExpectedCoverage(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "runs.csv")
	rows := "configuration,threads,run,exit_code,wall_s,run_dir,error\n" + strings.Join([]string{
		"cell-a,6,1,0,10,/tmp/cell-a-1,",
		"cell-a,6,2,0,12,/tmp/cell-a-2,",
		"cell-b,8,1,0,11,/tmp/cell-b-1,",
		"cell-b,8,2,0,9,/tmp/cell-b-2,",
	}, "\n") + "\n"
	if err := os.WriteFile(path, []byte(rows), 0o600); err != nil {
		t.Fatal(err)
	}
	var output strings.Builder
	if err := runAnalysisCommand([]string{"--min-runs", "2", "--expected-cells", "2", path}, &output); err != nil {
		t.Fatalf("runAnalysisCommand() error = %v", err)
	}
	if !strings.Contains(output.String(), "PASS: all 2 expected configuration/thread cells have at least 2 successful counted runs") {
		t.Fatalf("report missing completeness audit:\n%s", output.String())
	}
}

func TestValidateAnalysisMatrixChecksExactCartesianCells(t *testing.T) {
	root := t.TempDir()
	settings := analysisSettings{
		Model: "model-a", Context: 65536, Parallel: 2, Batch: 2048, UBatch: 256,
		FitTarget: 512, CacheReuse: 0, FlashAttention: "on", UnifiedKV: true,
		KVPerSlot: 204800, KVK: "q5_1", KVV: "q5_1", PreserveThinking: false,
	}
	manifest := analysisMatrixManifest{
		Models: []string{"model-a"}, Contexts: []int{65536}, Threads: []int{6},
		Parallel: []int{2}, Batches: []int{2048}, UBatches: []int{256}, FitTargets: []int{512},
		CacheReuse: []int{0}, FlashAttention: []string{"on"}, UnifiedKV: []bool{true},
		KVPerSlot: []int{204800}, KVK: []string{"q5_1"}, KVV: []string{"q5_1"},
		PreserveThinking: []bool{false},
	}
	makeResult := func(name string, config analysisSettings, repetition, exitCode int) runResult {
		t.Helper()
		dir := filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		data, err := json.Marshal(analysisRunMeta{Config: config})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "meta.json"), data, 0o600); err != nil {
			t.Fatal(err)
		}
		return runResult{ConfigName: name, ConfigKey: config.key(true), ThreadCount: 6, Repetition: repetition, ExitCode: exitCode, RunDir: dir}
	}

	complete := []runResult{
		makeResult("valid-1", settings, 1, 0),
		makeResult("valid-2", settings, 2, 0),
	}
	if err := validateAnalysisMatrix(complete, manifest, 2); err != nil {
		t.Fatalf("complete matrix: %v", err)
	}
	if err := validateAnalysisMatrix(complete[:1], manifest, 2); err == nil || !strings.Contains(err.Error(), "under-sampled") {
		t.Fatalf("under-sampled matrix error = %v", err)
	}
	extra := settings
	extra.UBatch = 512
	unexpected := append(append([]runResult(nil), complete...), makeResult("unexpected", extra, 1, 0))
	if err := validateAnalysisMatrix(unexpected, manifest, 1); err == nil || !strings.Contains(err.Error(), "unexpected cells") {
		t.Fatalf("unexpected-cell error = %v", err)
	}
}

func TestGeneratePairwisePlanCoversEveryCrossAxisValuePair(t *testing.T) {
	manifest := analysisMatrixManifest{
		Models: []string{"model-a", "model-b"}, Contexts: []int{65536, 131072},
		Threads: []int{6, 8}, Parallel: []int{1, 2}, Batches: []int{256, 512},
		UBatches: []int{128, 256}, FitTargets: []int{256, 512},
		CacheReuse: []int{0}, FlashAttention: []string{"on"}, UnifiedKV: []bool{true},
		KVPerSlot: []int{204800}, KVK: []string{"q4_0", "q5_1"},
		KVV: []string{"q4_0", "q5_1"}, PreserveThinking: []bool{false, true},
	}
	plan, err := generatePairwisePlan(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan) == 0 || len(plan) >= 2*2*2*2*2*2*2*2*2*2 {
		t.Fatalf("unexpected plan size: %d", len(plan))
	}
	factors, err := coverageFactors(manifest)
	if err != nil {
		t.Fatal(err)
	}
	covered := make(map[string]bool)
	for _, cell := range plan {
		values := []string{
			cell.Settings.Model, strconv.Itoa(cell.Settings.Context), strconv.Itoa(cell.Settings.Parallel),
			strconv.Itoa(cell.Settings.Batch), strconv.Itoa(cell.Settings.UBatch), strconv.Itoa(cell.Settings.FitTarget),
			strconv.Itoa(cell.Settings.CacheReuse), cell.Settings.FlashAttention,
			strconv.FormatBool(cell.Settings.UnifiedKV), strconv.Itoa(cell.Settings.KVPerSlot),
			cell.Settings.KVK, cell.Settings.KVV, strconv.FormatBool(cell.Settings.PreserveThinking), strconv.Itoa(cell.Threads),
		}
		for i := 0; i < len(values); i++ {
			for j := i + 1; j < len(values); j++ {
				covered[pairKey(i, values[i], j, values[j])] = true
			}
		}
	}
	candidates := [][]string{{}}
	for _, factor := range factors {
		var next [][]string
		for _, partial := range candidates {
			for _, value := range factor.values {
				n := append([]string(nil), partial...)
				n = append(n, value)
				next = append(next, n)
			}
		}
		candidates = next
	}
	for _, values := range candidates {
		for i := 0; i < len(values); i++ {
			for j := i + 1; j < len(values); j++ {
				if !covered[pairKey(i, values[i], j, values[j])] {
					t.Errorf("pair not covered: axis %s=%s, axis %s=%s", factors[i].name, values[i], factors[j].name, values[j])
				}
			}
		}
	}
}

func TestValidateCoveragePlanRequiresSuccessfulPlannedCells(t *testing.T) {
	root := t.TempDir()
	settings := analysisSettings{
		Model: "model-a", Context: 65536, Parallel: 2, Batch: 512, UBatch: 256,
		FitTarget: 512, CacheReuse: 0, FlashAttention: "on", UnifiedKV: true,
		KVPerSlot: 65536, KVK: "q5_1", KVV: "q5_1", PreserveThinking: false,
	}
	cell := coverageCell{Settings: settings, Threads: 8}
	dir := filepath.Join(root, "run-01")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	meta, err := json.Marshal(analysisRunMeta{Config: settings})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "meta.json"), meta, 0o600); err != nil {
		t.Fatal(err)
	}
	// CSV analysis restores the exit code and metrics but not client flags.
	// The persisted zero exit code must still satisfy the coverage audit.
	result := runResult{RunDir: dir, ThreadCount: 8, Repetition: 1, ExitCode: 0}
	if err := validateCoveragePlan([]runResult{result}, []coverageCell{cell}, 1); err != nil {
		t.Fatalf("complete planned cell failed audit: %v", err)
	}
	if err := validateCoveragePlan(nil, []coverageCell{cell}, 1); err == nil || !strings.Contains(err.Error(), "under-sampled") {
		t.Fatalf("missing planned cell error = %v", err)
	}
}

func TestAnalysisSettingsSeparateServerAxes(t *testing.T) {
	base := analysisSettings{
		Model: "model-a", Context: 65536, Parallel: 2, Batch: 2048,
		UBatch: 256, FitTarget: 512, KVK: "q5_1", KVV: "q5_1",
		FlashAttention: "on", UnifiedKV: true, KVPerSlot: 65536,
	}
	variants := []analysisSettings{base}
	changed := base
	changed.Context = 32768
	variants = append(variants, changed)
	changed = base
	changed.CacheReuse = 256
	variants = append(variants, changed)
	changed = base
	changed.FlashAttention = "off"
	variants = append(variants, changed)
	changed = base
	changed.UnifiedKV = false
	variants = append(variants, changed)
	changed = base
	changed.KVPerSlot = 32768
	variants = append(variants, changed)

	keys := make(map[string]bool)
	for _, settings := range variants {
		key := settings.key(false)
		if keys[key] {
			t.Errorf("different server settings collapsed to key %q", key)
		}
		keys[key] = true
	}
}
