package main

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type analysisGroup struct {
	name        string
	identity    string
	preserveKey string
	preserveOn  bool
	thread      int
	runs        []runResult
}

type analysisSettings struct {
	Model            string `json:"model"`
	Context          int    `json:"ctx"`
	Parallel         int    `json:"parallel"`
	Batch            int    `json:"batch"`
	UBatch           int    `json:"ubatch"`
	FitTarget        int    `json:"fit_target"`
	CacheReuse       int    `json:"cache_reuse"`
	FlashAttention   string `json:"flash_attention"`
	UnifiedKV        bool   `json:"unified_kv"`
	KVPerSlot        int    `json:"kv_per_slot"`
	KVK              string `json:"kv_k"`
	KVV              string `json:"kv_v"`
	PreserveThinking bool   `json:"preserve_thinking"`
}

type analysisRunMeta struct {
	Config        analysisSettings `json:"config"`
	ServerCommand []string         `json:"server_command"`
}

type analysisMatrixManifest struct {
	Models           []string `json:"models"`
	Contexts         []int    `json:"contexts"`
	Threads          []int    `json:"threads"`
	Parallel         []int    `json:"parallel"`
	Batches          []int    `json:"batches"`
	UBatches         []int    `json:"ubatches"`
	FitTargets       []int    `json:"fit_targets"`
	CacheReuse       []int    `json:"cache_reuse"`
	FlashAttention   []string `json:"flash_attention"`
	UnifiedKV        []bool   `json:"unified_kv"`
	KVPerSlot        []int    `json:"kv_per_slot"`
	KVK              []string `json:"kv_k"`
	KVV              []string `json:"kv_v"`
	PreserveThinking []bool   `json:"preserve_thinking"`
}

type analysisMatrixAxis struct {
	name   string
	values []string
	apply  func(*analysisSettings, string)
}

func (s analysisSettings) key(includePreserve bool) string {
	key := fmt.Sprintf("%s|%d|%d|%d|%d|%d|%d|%s|%s|%s|%t|%d", s.Model, s.Context, s.Parallel, s.Batch, s.UBatch, s.FitTarget, s.CacheReuse, s.FlashAttention, s.KVK, s.KVV, s.UnifiedKV, s.KVPerSlot)
	if includePreserve {
		key += fmt.Sprintf("|%t", s.PreserveThinking)
	}
	return key
}

func (s analysisSettings) label() string {
	return fmt.Sprintf("model=%s,ctx=%d,p=%d,b=%d,ub=%d,fit=%d,kv=%s/%s,cache-reuse=%d,flash=%s,unified-kv=%t,kv-per-slot=%d,preserve=%t",
		s.Model, s.Context, s.Parallel, s.Batch, s.UBatch, s.FitTarget, s.KVK, s.KVV, s.CacheReuse, s.FlashAttention, s.UnifiedKV, s.KVPerSlot, s.PreserveThinking)
}

func readAnalysisSettings(runDir string) (analysisSettings, bool) {
	if runDir == "" {
		return analysisSettings{}, false
	}
	data, err := os.ReadFile(filepath.Join(runDir, "meta.json"))
	if err != nil {
		return analysisSettings{}, false
	}
	var meta analysisRunMeta
	if err := json.Unmarshal(data, &meta); err != nil || meta.Config.Model == "" {
		return analysisSettings{}, false
	}
	if meta.Config.FlashAttention == "" {
		for i := 0; i+1 < len(meta.ServerCommand); i++ {
			if meta.ServerCommand[i] == "-fa" {
				meta.Config.FlashAttention = meta.ServerCommand[i+1]
				break
			}
		}
	}
	return meta.Config, true
}

func groupIdentity(result runResult) (label, identity, preserveKey string, preserveOn bool) {
	if settings, ok := readAnalysisSettings(result.RunDir); ok {
		return settings.label(), settings.key(true), settings.key(false), settings.PreserveThinking
	}
	baseName, preserveOn := preserveBaseName(result.ConfigName)
	if preserveOn {
		return result.ConfigName, "name:" + result.ConfigName, "name:" + baseName, true
	}
	return result.ConfigName, "name:" + result.ConfigName, "name:" + result.ConfigName, false
}

// runAnalysisCommand summarizes one or more runs.csv files without rerunning
// the benchmark. Directories are accepted as shorthand for directories that
// contain runs.csv.
func runAnalysisCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("analyze", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	output := fs.String("out", "", "write the Markdown report to this file instead of stdout")
	minRuns := fs.Int("min-runs", 1, "minimum successful runs for a cell to qualify as the latency leader")
	expectedCells := fs.Int("expected-cells", 0, "require exactly this many configuration/thread cells and at least --min-runs successes in each")
	matrixManifest := fs.String("matrix-manifest", "", "require exact Cartesian coverage of the values in this JSON file")
	coveragePlan := fs.String("coverage-plan", "", "require exact setting/thread coverage of a generated pairwise plan JSON")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprintln(stdout, "Usage: llama-bench-harness analyze [--out report.md] [--min-runs N] [--expected-cells N] [--matrix-manifest matrix.json] <runs.csv-or-output-dir>...")
			fs.SetOutput(stdout)
			fs.PrintDefaults()
			return nil
		}
		return err
	}
	if *minRuns < 1 {
		return errors.New("--min-runs must be at least 1")
	}
	if *expectedCells < 0 {
		return errors.New("--expected-cells cannot be negative")
	}
	inputs := fs.Args()
	if len(inputs) == 0 {
		return errors.New("usage: llama-bench-harness analyze [--out report.md] [--min-runs N] [--expected-cells N] [--matrix-manifest matrix.json] <runs.csv-or-output-dir>...")
	}

	results, err := readAnalysisInputs(inputs)
	if err != nil {
		return err
	}
	report := renderAnalysisReport(results, *minRuns)
	if *expectedCells > 0 {
		if err := validateAnalysisCoverage(results, *expectedCells, *minRuns); err != nil {
			return err
		}
		report = strings.TrimRight(report, "\n") + fmt.Sprintf("\n\n## Completeness audit\n\nPASS: all %d expected configuration/thread cells have at least %d successful counted runs.\n", *expectedCells, *minRuns)
	}
	if *matrixManifest != "" {
		manifest, err := readAnalysisMatrixManifest(*matrixManifest)
		if err != nil {
			return err
		}
		if err := validateAnalysisMatrix(results, manifest, *minRuns); err != nil {
			return err
		}
		report = strings.TrimRight(report, "\n") + fmt.Sprintf("\n\n## Exact matrix audit\n\nPASS: all expected Cartesian cells across %d threads have at least %d successful counted runs.\n", len(manifest.Threads), *minRuns)
	}
	if *coveragePlan != "" {
		plan, err := readCoveragePlan(*coveragePlan)
		if err != nil {
			return err
		}
		if err := validateCoveragePlan(results, plan, *minRuns); err != nil {
			return err
		}
		report = strings.TrimRight(report, "\n") + fmt.Sprintf("\n\n## Pairwise coverage audit\n\nPASS: all %d planned setting/thread cells have at least %d successful counted runs.\n", len(plan), *minRuns)
	}
	if *output == "" || *output == "-" {
		_, err = io.WriteString(stdout, report)
		return err
	}
	if err := os.MkdirAll(filepath.Dir(*output), 0o755); err != nil {
		return fmt.Errorf("create report directory: %w", err)
	}
	if err := os.WriteFile(*output, []byte(report), 0o644); err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	_, err = fmt.Fprintf(stdout, "Analysis report: %s\n", *output)
	return err
}

func readCoveragePlan(path string) ([]coverageCell, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read coverage plan %q: %w", path, err)
	}
	var plan []coverageCell
	if err := json.Unmarshal(data, &plan); err != nil {
		return nil, fmt.Errorf("parse coverage plan %q: %w", path, err)
	}
	if len(plan) == 0 {
		return nil, errors.New("coverage plan contains no cells")
	}
	return plan, nil
}

func validateCoveragePlan(results []runResult, plan []coverageCell, minRuns int) error {
	expected := make(map[string]int, len(plan))
	for _, cell := range plan {
		key := fmt.Sprintf("%s|%d", cell.Settings.key(true), cell.Threads)
		if _, exists := expected[key]; exists {
			return fmt.Errorf("coverage plan repeats cell %q", key)
		}
		expected[key] = 0
	}
	for _, result := range results {
		if result.Warmup {
			continue
		}
		settings, ok := readAnalysisSettings(result.RunDir)
		if !ok {
			return fmt.Errorf("cannot verify planned cell metadata for %s", result.RunDir)
		}
		key := fmt.Sprintf("%s|%d", settings.key(true), result.ThreadCount)
		count, exists := expected[key]
		if !exists {
			return fmt.Errorf("unexpected setting/thread cell %q", key)
		}
		// The persisted runs.csv carries the harness exit code, but not the
		// client's Parsed/Passed flags. A zero exit code is the authoritative
		// success signal for both the CSV and in-memory run results.
		if result.ExitCode == 0 {
			expected[key] = count + 1
		}
	}
	var missing []string
	for key, count := range expected {
		if count < minRuns {
			missing = append(missing, fmt.Sprintf("%s has %d/%d successful runs", key, count, minRuns))
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("pairwise coverage audit failed: %d cells under-sampled; first: %s", len(missing), strings.Join(missing[:min(3, len(missing))], "; "))
	}
	return nil
}

func readAnalysisMatrixManifest(path string) (analysisMatrixManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return analysisMatrixManifest{}, fmt.Errorf("read matrix manifest %q: %w", path, err)
	}
	var manifest analysisMatrixManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return analysisMatrixManifest{}, fmt.Errorf("parse matrix manifest %q: %w", path, err)
	}
	return manifest, nil
}

func validateAnalysisMatrix(results []runResult, manifest analysisMatrixManifest, minRuns int) error {
	stringValues := func(values []string) []string { return values }
	intValues := func(values []int) []string {
		out := make([]string, len(values))
		for i, value := range values {
			out[i] = strconv.Itoa(value)
		}
		return out
	}
	boolValues := func(values []bool) []string {
		out := make([]string, len(values))
		for i, value := range values {
			out[i] = strconv.FormatBool(value)
		}
		return out
	}
	axes := []analysisMatrixAxis{
		{name: "models", values: stringValues(manifest.Models), apply: func(s *analysisSettings, v string) { s.Model = v }},
		{name: "contexts", values: intValues(manifest.Contexts), apply: func(s *analysisSettings, v string) { s.Context, _ = strconv.Atoi(v) }},
		{name: "parallel", values: intValues(manifest.Parallel), apply: func(s *analysisSettings, v string) { s.Parallel, _ = strconv.Atoi(v) }},
		{name: "batches", values: intValues(manifest.Batches), apply: func(s *analysisSettings, v string) { s.Batch, _ = strconv.Atoi(v) }},
		{name: "ubatches", values: intValues(manifest.UBatches), apply: func(s *analysisSettings, v string) { s.UBatch, _ = strconv.Atoi(v) }},
		{name: "fit_targets", values: intValues(manifest.FitTargets), apply: func(s *analysisSettings, v string) { s.FitTarget, _ = strconv.Atoi(v) }},
		{name: "cache_reuse", values: intValues(manifest.CacheReuse), apply: func(s *analysisSettings, v string) { s.CacheReuse, _ = strconv.Atoi(v) }},
		{name: "flash_attention", values: stringValues(manifest.FlashAttention), apply: func(s *analysisSettings, v string) { s.FlashAttention = v }},
		{name: "unified_kv", values: boolValues(manifest.UnifiedKV), apply: func(s *analysisSettings, v string) { s.UnifiedKV, _ = strconv.ParseBool(v) }},
		{name: "kv_per_slot", values: intValues(manifest.KVPerSlot), apply: func(s *analysisSettings, v string) { s.KVPerSlot, _ = strconv.Atoi(v) }},
		{name: "kv_k", values: stringValues(manifest.KVK), apply: func(s *analysisSettings, v string) { s.KVK = v }},
		{name: "kv_v", values: stringValues(manifest.KVV), apply: func(s *analysisSettings, v string) { s.KVV = v }},
		{name: "preserve_thinking", values: boolValues(manifest.PreserveThinking), apply: func(s *analysisSettings, v string) { s.PreserveThinking, _ = strconv.ParseBool(v) }},
	}
	configs := []analysisSettings{{}}
	for _, axis := range axes {
		if len(axis.values) == 0 {
			return fmt.Errorf("matrix manifest: axis %s has no values", axis.name)
		}
		seen := make(map[string]bool, len(axis.values))
		for _, value := range axis.values {
			if seen[value] {
				return fmt.Errorf("matrix manifest: axis %s repeats value %q", axis.name, value)
			}
			seen[value] = true
		}
		next := make([]analysisSettings, 0, len(configs)*len(axis.values))
		for _, config := range configs {
			for _, value := range axis.values {
				candidate := config
				axis.apply(&candidate, value)
				next = append(next, candidate)
			}
		}
		configs = next
	}
	if len(manifest.Threads) == 0 {
		return errors.New("matrix manifest: threads has no values")
	}
	threadSeen := make(map[int]bool, len(manifest.Threads))
	expected := make(map[string]bool, len(configs)*len(manifest.Threads))
	for _, thread := range manifest.Threads {
		if threadSeen[thread] {
			return fmt.Errorf("matrix manifest: threads repeats value %d", thread)
		}
		threadSeen[thread] = true
		for _, config := range configs {
			expected[fmt.Sprintf("%s|%d", config.key(true), thread)] = true
		}
	}
	observed := make(map[string]*analysisGroup)
	for _, result := range results {
		settings, ok := readAnalysisSettings(result.RunDir)
		if !ok {
			return fmt.Errorf("matrix audit: configuration metadata missing for %s", result.RunDir)
		}
		key := fmt.Sprintf("%s|%d", settings.key(true), result.ThreadCount)
		group := observed[key]
		if group == nil {
			group = &analysisGroup{name: settings.label(), identity: settings.key(true), thread: result.ThreadCount}
			observed[key] = group
		}
		group.runs = append(group.runs, result)
	}
	var missing, unexpected, underSampled []string
	for key := range expected {
		group := observed[key]
		if group == nil {
			missing = append(missing, key)
			continue
		}
		if summary := summarizeAnalysisGroup(group); summary.Success < minRuns {
			underSampled = append(underSampled, fmt.Sprintf("%s threads=%d has %d/%d successful runs", group.name, group.thread, summary.Success, summary.Runs))
		}
	}
	for key, group := range observed {
		if !expected[key] {
			unexpected = append(unexpected, fmt.Sprintf("%s threads=%d", group.name, group.thread))
		}
	}
	if len(missing) > 0 || len(unexpected) > 0 || len(underSampled) > 0 {
		sort.Strings(missing)
		sort.Strings(unexpected)
		sort.Strings(underSampled)
		return fmt.Errorf("matrix audit failed: %d missing cells, %d unexpected cells, %d under-sampled cells (need %d successful runs)", len(missing), len(unexpected), len(underSampled), minRuns)
	}
	return nil
}

func validateAnalysisCoverage(results []runResult, expectedCells, minRuns int) error {
	groups := make(map[string]*analysisGroup)
	for _, result := range results {
		name, identity, preserveKey, preserveOn := groupIdentity(result)
		key := fmt.Sprintf("%s|%d", identity, result.ThreadCount)
		group := groups[key]
		if group == nil {
			group = &analysisGroup{name: name, identity: identity, preserveKey: preserveKey, preserveOn: preserveOn, thread: result.ThreadCount}
			groups[key] = group
		}
		group.runs = append(group.runs, result)
	}
	if len(groups) != expectedCells {
		return fmt.Errorf("completeness audit: found %d configuration/thread cells, want %d", len(groups), expectedCells)
	}
	var incomplete []string
	for _, group := range groups {
		summary := summarizeAnalysisGroup(group)
		if summary.Success < minRuns {
			incomplete = append(incomplete, fmt.Sprintf("%s threads=%d has %d/%d successful runs", group.name, group.thread, summary.Success, summary.Runs))
		}
	}
	if len(incomplete) > 0 {
		sort.Strings(incomplete)
		incompleteCount := len(incomplete)
		const maxDetails = 10
		if len(incomplete) > maxDetails {
			incomplete = append(incomplete[:maxDetails], fmt.Sprintf("... and %d more incomplete cells", len(incomplete)-maxDetails))
		}
		return fmt.Errorf("completeness audit: %d cells have fewer than %d successful runs: %s", incompleteCount, minRuns, strings.Join(incomplete, "; "))
	}
	return nil
}

func readAnalysisInputs(inputs []string) ([]runResult, error) {
	var results []runResult
	seen := make(map[string]bool)
	for _, input := range inputs {
		path := input
		info, err := os.Stat(path)
		if err != nil {
			return nil, fmt.Errorf("stat %q: %w", path, err)
		}
		if info.IsDir() {
			path = filepath.Join(path, "runs.csv")
		}
		parsed, err := readRunsCSV(path)
		if err != nil {
			return nil, fmt.Errorf("read %q: %w", path, err)
		}
		for _, result := range parsed {
			key := result.RunDir
			if key == "" {
				key = fmt.Sprintf("%s|%s|%d|%d", path, result.ConfigName, result.ThreadCount, result.Repetition)
			}
			if seen[key] {
				continue
			}
			seen[key] = true
			results = append(results, result)
		}
	}
	if len(results) == 0 {
		return nil, errors.New("input files contain no run rows")
	}
	return results, nil
}

func readRunsCSV(path string) ([]runResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	header, err := r.Read()
	if err != nil {
		return nil, err
	}
	columns := make(map[string]int, len(header))
	for i, name := range header {
		columns[strings.TrimSpace(name)] = i
	}
	required := []string{"configuration", "threads", "run", "exit_code", "wall_s", "run_dir", "error"}
	for _, name := range required {
		if _, ok := columns[name]; !ok {
			return nil, fmt.Errorf("missing required column %q", name)
		}
	}
	var results []runResult
	for line := 2; ; line++ {
		row, readErr := r.Read()
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return nil, fmt.Errorf("line %d: %w", line, readErr)
		}
		if len(row) == 1 && strings.TrimSpace(row[0]) == "" {
			continue
		}
		value := func(name string) string {
			i, ok := columns[name]
			if !ok || i >= len(row) {
				return ""
			}
			return strings.TrimSpace(row[i])
		}
		parseInt := func(name string) (int, error) {
			s := value(name)
			if s == "" {
				return 0, nil
			}
			n, parseErr := strconv.Atoi(s)
			if parseErr != nil {
				return 0, fmt.Errorf("line %d column %s: %w", line, name, parseErr)
			}
			return n, nil
		}
		parseFloat := func(name string) (float64, error) {
			s := value(name)
			if s == "" {
				return 0, nil
			}
			n, parseErr := strconv.ParseFloat(s, 64)
			if parseErr != nil {
				return 0, fmt.Errorf("line %d column %s: %w", line, name, parseErr)
			}
			return n, nil
		}
		var result runResult
		result.ConfigName = value("configuration")
		result.ConfigKey = result.ConfigName
		result.RunDir = value("run_dir")
		result.Error = value("error")
		intFields := []struct {
			name string
			to   *int
		}{
			{"threads", &result.ThreadCount}, {"run", &result.Repetition}, {"exit_code", &result.ExitCode},
			{"client_prompt_tokens", &result.Client.PromptTokens}, {"client_reasoning_tokens", &result.Client.ReasoningTokens},
			{"turns", &result.Client.Turns}, {"out_tokens", &result.Client.OutTokens},
			{"tasks", &result.Llama.TaskCount}, {"slots_used", &result.Llama.SlotsUsed},
			{"cancelled_tasks", &result.Llama.CancelledTasks}, {"max_concurrent_tasks", &result.Llama.MaxConcurrentTasks},
			{"large_prefill_events", &result.Llama.LargePrefillEvents}, {"decode_starvation_events", &result.Llama.DecodeStarvationEvents},
			{"fit_spill_blocks", &result.FitSpillBlocks},
		}
		for _, field := range intFields {
			*field.to, err = parseInt(field.name)
			if err != nil {
				return nil, err
			}
		}
		floatFields := []struct {
			name string
			to   *float64
		}{
			{"wall_s", &result.WallSeconds}, {"client_reported_s", &result.Client.ReportedSeconds},
			{"prompt_s", &result.Llama.PromptSeconds}, {"prompt_tps", &result.Llama.PromptTPSWeighted},
			{"summed_eval_s", &result.Llama.EvalSecondsSummed}, {"task_tg_weighted", &result.Llama.TaskTGWeighted},
			{"min_tg3s", &result.Llama.MinTG3s}, {"metrics_wall_gen_tps", &result.Metrics.WallGenerationTPS},
			{"max_requests_processing", &result.Metrics.MaxRequestsProcessing},
			{"max_requests_deferred", &result.Metrics.MaxRequestsDeferred}, {"max_context", &result.Metrics.MaxContextObserved},
			{"mem_mean_read_gbs", &result.Memory.MeanRead}, {"mem_max_read_gbs", &result.Memory.MaxRead},
			{"mem_mean_write_gbs", &result.Memory.MeanWrite}, {"gpu_mean_util_pct", &result.GPU.MeanUtil},
			{"gpu_max_util_pct", &result.GPU.MaxUtil}, {"gpu_mean_power_w", &result.GPU.MeanPower},
			{"gpu_max_power_w", &result.GPU.MaxPower},
		}
		for _, field := range floatFields {
			*field.to, err = parseFloat(field.name)
			if err != nil {
				return nil, err
			}
			if math.IsNaN(*field.to) || math.IsInf(*field.to, 0) {
				return nil, fmt.Errorf("line %d column %s: non-finite value", line, field.name)
			}
		}
		if promptTokens, parseErr := strconv.ParseInt(value("prompt_tokens"), 10, 64); parseErr == nil {
			result.Llama.PromptTokens = promptTokens
		} else if value("prompt_tokens") != "" {
			return nil, fmt.Errorf("line %d column prompt_tokens: %w", line, parseErr)
		}
		if generatedTokens, parseErr := strconv.ParseInt(value("generated_tokens"), 10, 64); parseErr == nil {
			result.Llama.GeneratedTokens = generatedTokens
		} else if value("generated_tokens") != "" {
			return nil, fmt.Errorf("line %d column generated_tokens: %w", line, parseErr)
		}
		results = append(results, result)
	}
	return results, nil
}

func renderAnalysisReport(results []runResult, minRuns int) string {
	groups := make(map[string]*analysisGroup)
	for _, result := range results {
		name, identity, preserveKey, preserveOn := groupIdentity(result)
		key := fmt.Sprintf("%s|%d", identity, result.ThreadCount)
		group := groups[key]
		if group == nil {
			group = &analysisGroup{name: name, identity: identity, preserveKey: preserveKey, preserveOn: preserveOn, thread: result.ThreadCount}
			groups[key] = group
		}
		group.runs = append(group.runs, result)
	}
	ordered := make([]*analysisGroup, 0, len(groups))
	for _, group := range groups {
		ordered = append(ordered, group)
	}
	sort.Slice(ordered, func(i, j int) bool {
		ai, aj := summarizeAnalysisGroup(ordered[i]), summarizeAnalysisGroup(ordered[j])
		if ai.Success != aj.Success {
			return ai.Success > aj.Success
		}
		if ai.MedianWall != aj.MedianWall {
			return ai.MedianWall < aj.MedianWall
		}
		if ordered[i].name != ordered[j].name {
			return ordered[i].name < ordered[j].name
		}
		return ordered[i].thread < ordered[j].thread
	})

	var b strings.Builder
	fmt.Fprintf(&b, "# Benchmark analysis\n\nInput: %d counted runs across %d configuration/thread cells. Latency leader requires at least %d successful runs.\n\n", len(results), len(groups), minRuns)
	var leader *analysisGroup
	for _, group := range ordered {
		summary := summarizeAnalysisGroup(group)
		if leader == nil && summary.Success >= minRuns {
			leader = group
		}
	}
	if leader != nil {
		s := summarizeAnalysisGroup(leader)
		fmt.Fprintf(&b, "Best observed cell by median end-to-end wall time: **%s, %d threads** — %.2fs median (%d/%d successful runs).\n\n", leader.name, leader.thread, s.MedianWall, s.Success, s.Runs)
	}
	b.WriteString("## Cell results\n\n")
	b.WriteString("| configuration | threads | ok/runs | median wall | p95 wall | median task decode tok/s | median wall gen tok/s | median prompt tok/s | median client prompt tokens | median output tokens | GPU util | DDR read GB/s |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, group := range ordered {
		s := summarizeAnalysisGroup(group)
		fmt.Fprintf(&b, "| %s | %d | %d/%d | %.2fs | %.2fs | %.2f | %.2f | %.1f | %.0f | %.0f | %.1f%% | %.2f |\n",
			group.name, group.thread, s.Success, s.Runs, s.MedianWall, s.P95Wall, s.MedianTaskTG, s.MedianWallTG,
			s.MedianPromptTPS, s.MedianClientPrompt, s.MedianOutputTokens, s.MedianGPUUtil, s.MedianMemRead)
	}

	b.WriteString("\n## Preserve-thinking comparisons\n\n")
	b.WriteString("Matched cells compare the same configuration and thread count, with only preserve-thinking changed. Positive token deltas mean preserve-thinking used more tokens.\n\n")
	b.WriteString("| configuration | threads | off/on runs | client prompt tokens off → on | delta | wall seconds off → on | delta | output tokens off → on |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|---:|\n")
	pairs := preservePairs(ordered)
	if len(pairs) == 0 {
		b.WriteString("No matched preserve-thinking cells are present.\n")
	} else {
		for _, pair := range pairs {
			off, on := summarizeAnalysisGroup(pair.off), summarizeAnalysisGroup(pair.on)
			promptDelta := percentDelta(off.MedianClientPrompt, on.MedianClientPrompt)
			wallDelta := percentDelta(off.MedianWall, on.MedianWall)
			fmt.Fprintf(&b, "| %s | %d | %d/%d, %d/%d | %.0f → %.0f | %+.1f%% | %.2f → %.2f | %+.1f%% | %.0f → %.0f |\n",
				pair.name, pair.off.thread, off.Success, off.Runs, on.Success, on.Runs,
				off.MedianClientPrompt, on.MedianClientPrompt, promptDelta,
				off.MedianWall, on.MedianWall, wallDelta, off.MedianOutputTokens, on.MedianOutputTokens)
		}
	}
	b.WriteString("\n## Sample coverage\n\n")
	b.WriteString("| configuration | threads measured | cells complete at requested minimum |\n|---|---|---:|\n")
	coverage := make(map[string]map[int]bool)
	for _, group := range ordered {
		s := summarizeAnalysisGroup(group)
		if coverage[group.name] == nil {
			coverage[group.name] = make(map[int]bool)
		}
		coverage[group.name][group.thread] = s.Success >= minRuns
	}
	configNames := make([]string, 0, len(coverage))
	for name := range coverage {
		configNames = append(configNames, name)
	}
	sort.Strings(configNames)
	for _, name := range configNames {
		threads := make([]int, 0, len(coverage[name]))
		complete := 0
		for thread, done := range coverage[name] {
			threads = append(threads, thread)
			if done {
				complete++
			}
		}
		sort.Ints(threads)
		threadLabels := make([]string, 0, len(threads))
		for _, thread := range threads {
			threadLabels = append(threadLabels, strconv.Itoa(thread))
		}
		fmt.Fprintf(&b, "| %s | %s | %d/%d |\n", name, strings.Join(threadLabels, ", "), complete, len(threads))
	}
	b.WriteString("\n`task decode tok/s` is per-task decode throughput. `wall gen tok/s` includes gaps between client requests. DDR values are zero when memory profiling was unavailable or disabled.\n")
	return b.String()
}

type analysisSummary struct {
	Runs               int
	Success            int
	MedianWall         float64
	P95Wall            float64
	MedianTaskTG       float64
	MedianWallTG       float64
	MedianPromptTPS    float64
	MedianClientPrompt float64
	MedianOutputTokens float64
	MedianGPUUtil      float64
	MedianMemRead      float64
}

func summarizeAnalysisGroup(group *analysisGroup) analysisSummary {
	var successful []runResult
	var wall, wallTG, taskTG, promptTPS, clientPrompt, output, gpu, mem []float64
	for _, run := range group.runs {
		if run.ExitCode != 0 {
			continue
		}
		successful = append(successful, run)
		wall = append(wall, run.WallSeconds)
		clientPrompt = append(clientPrompt, float64(run.Client.PromptTokens))
		output = append(output, float64(run.Client.OutTokens))
		if run.Llama.TaskTGWeighted > 0 {
			taskTG = append(taskTG, run.Llama.TaskTGWeighted)
		}
		if run.Metrics.WallGenerationTPS > 0 {
			wallTG = append(wallTG, run.Metrics.WallGenerationTPS)
		}
		if run.Llama.PromptTPSWeighted > 0 {
			promptTPS = append(promptTPS, run.Llama.PromptTPSWeighted)
		}
		if run.GPU.MeanUtil > 0 {
			gpu = append(gpu, run.GPU.MeanUtil)
		}
		if run.Memory.MeanRead > 0 {
			mem = append(mem, run.Memory.MeanRead)
		}
	}
	wallStats, taskStats := statsOf(wall), statsOf(taskTG)
	return analysisSummary{
		Runs: len(group.runs), Success: len(successful), MedianWall: wallStats.Median, P95Wall: wallStats.P95,
		MedianTaskTG: taskStats.Median, MedianWallTG: statsOf(wallTG).Median, MedianPromptTPS: statsOf(promptTPS).Median,
		MedianClientPrompt: statsOf(clientPrompt).Median, MedianOutputTokens: statsOf(output).Median,
		MedianGPUUtil: statsOf(gpu).Median, MedianMemRead: statsOf(mem).Median,
	}
}

type preservePair struct {
	name string
	off  *analysisGroup
	on   *analysisGroup
}

func preservePairs(groups []*analysisGroup) []preservePair {
	by := make(map[string]*analysisGroup, len(groups))
	for _, group := range groups {
		by[fmt.Sprintf("%s|%d", group.identity, group.thread)] = group
	}
	var pairs []preservePair
	for _, group := range groups {
		if !group.preserveOn {
			continue
		}
		off := by[fmt.Sprintf("%s|%d", group.preserveKey, group.thread)]
		if off != nil {
			pairs = append(pairs, preservePair{name: off.name, off: off, on: group})
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].name != pairs[j].name {
			return pairs[i].name < pairs[j].name
		}
		return pairs[i].off.thread < pairs[j].off.thread
	})
	return pairs
}

func preserveBaseName(name string) (string, bool) {
	if name == "preserve-thinking-on" {
		return "base", true
	}
	base, ok := strings.CutSuffix(name, "_preserve-thinking-on")
	return base, ok
}

func percentDelta(before, after float64) float64 {
	if before == 0 {
		return 0
	}
	return (after/before - 1) * 100
}
