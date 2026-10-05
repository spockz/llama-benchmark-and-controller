// This file implements the single executable that owns benchmark run lifecycles and reports.
package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

type stringList []string

func (s *stringList) String() string { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error {
	*s = append(*s, v)
	return nil
}

type config struct {
	WorkDir              string
	OutputDir            string
	ClientCmd            string
	PromptScript         string
	OpenCodeConfig       string
	Shell                string
	Runs                 int
	WarmupRuns           int
	ThreadValues         string
	ContextValues        string
	ParallelValues       string
	BatchValues          string
	UBatchValues         string
	FitTargetValues      string
	KVKValues            string
	KVVValues            string
	ModelValues          string
	PlanFile             string
	PreserveThinkValues  string
	CacheReuseValues     string
	FlashAttentionValues string
	FailFast             bool
	Resume               bool
	ClientTimeout        time.Duration
	HealthTimeout        time.Duration
	ServerSettle         time.Duration

	ServerBinary        string
	FitBinary           string
	InspectFit          bool
	Model               string
	Alias               string
	Host                string
	Port                int
	Context             int
	Parallel            int
	UnifiedKV           bool
	KVPerSlot           int
	KVK                 string
	KVV                 string
	Batch               int
	UBatch              int
	FitTarget           int
	NGL                 string
	NCPUMoE             int
	FlashAttention      string
	CacheReuse          int
	LoadMode            string
	LazyMode            string
	Reasoning           bool
	PreserveThink       bool
	Temp                float64
	TopP                float64
	TopK                int
	LogVerbosity        int
	StarvationThreshold float64
	LargePrefillTokens  int64

	AmdSMI         string
	GPU            int
	ProfileGPU     bool
	UProf          string
	ProfileMemory  bool
	UProfSudo      bool
	ModprobeUncore bool

	ServerArgs stringList
	ServerEnv  stringList
	ClientEnv  stringList
}

type managedProc struct {
	cmd  *exec.Cmd
	done chan struct{}
	mu   sync.Mutex
	err  error
}

type experiment struct {
	Name    string
	Key     string
	Config  config
	Threads []int
}

type experimentAxis struct {
	name   string
	values []string
	apply  func(*config, string) error
}

func startManaged(cmd *exec.Cmd) (*managedProc, error) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	p := &managedProc{cmd: cmd, done: make(chan struct{})}
	go func() {
		err := cmd.Wait()
		p.mu.Lock()
		p.err = err
		p.mu.Unlock()
		close(p.done)
	}()
	return p, nil
}

func (p *managedProc) exited() (bool, error) {
	select {
	case <-p.done:
		p.mu.Lock()
		defer p.mu.Unlock()
		return true, p.err
	default:
		return false, nil
	}
}

func (p *managedProc) stop(sig syscall.Signal, timeout time.Duration, viaSudo bool) {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return
	}
	if exited, _ := p.exited(); exited {
		return
	}
	pid := p.cmd.Process.Pid
	if viaSudo {
		sigName := signalName(sig)
		_ = exec.Command("sudo", "-n", "kill", "-"+sigName, "--", fmt.Sprintf("-%d", pid)).Run()
	} else {
		_ = syscall.Kill(-pid, sig)
	}
	select {
	case <-p.done:
		return
	case <-time.After(timeout):
	}
	if viaSudo {
		_ = exec.Command("sudo", "-n", "kill", "-KILL", "--", fmt.Sprintf("-%d", pid)).Run()
	} else {
		_ = syscall.Kill(-pid, syscall.SIGKILL)
	}
	select {
	case <-p.done:
	case <-time.After(2 * time.Second):
	}
}

func signalName(sig syscall.Signal) string {
	switch sig {
	case syscall.SIGINT:
		return "INT"
	case syscall.SIGTERM:
		return "TERM"
	default:
		return strconv.Itoa(int(sig))
	}
}

type floatStats struct {
	Values []float64 `json:"-"`
	Count  int       `json:"count"`
	Mean   float64   `json:"mean"`
	Min    float64   `json:"min"`
	Max    float64   `json:"max"`
	Median float64   `json:"median"`
	P95    float64   `json:"p95"`
}

func statsOf(v []float64) floatStats {
	if len(v) == 0 {
		return floatStats{}
	}
	x := append([]float64(nil), v...)
	sort.Float64s(x)
	var sum float64
	for _, n := range x {
		sum += n
	}
	med := x[len(x)/2]
	if len(x)%2 == 0 {
		med = (x[len(x)/2-1] + x[len(x)/2]) / 2
	}
	p95 := x[int(math.Ceil(float64(len(x))*0.95))-1]
	return floatStats{
		Values: v,
		Count:  len(v),
		Mean:   sum / float64(len(v)),
		Min:    x[0],
		Max:    x[len(x)-1],
		Median: med,
		P95:    p95,
	}
}

type llamaStats struct {
	TaskCount                 int     `json:"task_count"`
	SlotsUsed                 int     `json:"slots_used"`
	CancelledTasks            int     `json:"cancelled_tasks"`
	PromptTokens              int64   `json:"prompt_tokens"`
	PromptSeconds             float64 `json:"prompt_seconds"`
	PromptTPSWeighted         float64 `json:"prompt_tps_weighted"`
	GeneratedTokens           int64   `json:"generated_tokens"`
	EvalSecondsSummed         float64 `json:"eval_seconds_summed"`
	TaskTGWeighted            float64 `json:"task_tg_weighted"`
	MinTG3s                   float64 `json:"min_tg_3s"`
	MaxTG3s                   float64 `json:"max_tg_3s"`
	MeanTG3s                  float64 `json:"mean_tg_3s"`
	MaxPromptProcessingTokens int64   `json:"max_prompt_processing_tokens"`
	LCPCount                  int     `json:"lcp_count"`
	MeanLCPSimilarity         float64 `json:"mean_lcp_similarity"`
	MeanLCPKeep               float64 `json:"mean_lcp_keep"`
	MaxConcurrentTasks        int     `json:"max_concurrent_tasks"`
	LargePrefillEvents        int     `json:"large_prefill_events"`
	DecodeStarvationEvents    int     `json:"decode_starvation_events"`
}

type memoryStats struct {
	ReadValues  []float64 `json:"-"`
	Samples     int       `json:"samples"`
	MeanTotal   float64   `json:"mean_total_gbs"`
	MaxTotal    float64   `json:"max_total_gbs"`
	MeanRead    float64   `json:"mean_read_gbs"`
	MaxRead     float64   `json:"max_read_gbs"`
	MeanWrite   float64   `json:"mean_write_gbs"`
	MaxWrite    float64   `json:"max_write_gbs"`
	MedianTotal float64   `json:"median_total_gbs"`
	MedianRead  float64   `json:"median_read_gbs"`
	MedianWrite float64   `json:"median_write_gbs"`
	P95Total    float64   `json:"p95_total_gbs"`
	P95Read     float64   `json:"p95_read_gbs"`
	P95Write    float64   `json:"p95_write_gbs"`
}

type gpuStats struct {
	UtilValues    []float64 `json:"-"`
	PowerValues   []float64 `json:"-"`
	Samples       int       `json:"samples"`
	MeanUtil      float64   `json:"mean_gfx_util_pct"`
	MaxUtil       float64   `json:"max_gfx_util_pct"`
	MeanMemUtil   float64   `json:"mean_mem_util_pct"`
	MaxMemUtil    float64   `json:"max_mem_util_pct"`
	MedianMemUtil float64   `json:"median_mem_util_pct"`
	P95MemUtil    float64   `json:"p95_mem_util_pct"`
	MeanPower     float64   `json:"mean_power_w"`
	MaxPower      float64   `json:"max_power_w"`
	MedianPower   float64   `json:"median_power_w"`
	P95Power      float64   `json:"p95_power_w"`
	MeanVRAMUsed  float64   `json:"mean_vram_used_mb"`
	MaxVRAMUsed   float64   `json:"max_vram_used_mb"`
	MedianVRAM    float64   `json:"median_vram_used_mb"`
	P95VRAM       float64   `json:"p95_vram_used_mb"`
	MedianUtil    float64   `json:"median_gfx_util_pct"`
	P95Util       float64   `json:"p95_gfx_util_pct"`
}

type metricSample struct {
	Time               time.Time
	PromptTokensTotal  float64
	PredictedTokens    float64
	PromptSecondsTotal float64
	PredSecondsTotal   float64
	RequestsProcessing float64
	RequestsDeferred   float64
	NTokensMax         float64
}

type metricStats struct {
	Samples               int     `json:"samples"`
	WallGenerationTPS     float64 `json:"wall_generation_tps"`
	WallPromptTPS         float64 `json:"wall_prompt_tps"`
	MaxRequestsProcessing float64 `json:"max_requests_processing"`
	MaxRequestsDeferred   float64 `json:"max_requests_deferred"`
	MaxContextObserved    float64 `json:"max_context_observed"`
	PredictedTokensDelta  float64 `json:"predicted_tokens_delta"`
	PromptTokensDelta     float64 `json:"prompt_tokens_delta"`
}

type clientStats struct {
	Parsed          bool    `json:"parsed"`
	Passed          bool    `json:"passed"`
	Target          string  `json:"target,omitempty"`
	Task            string  `json:"task,omitempty"`
	Variant         string  `json:"variant,omitempty"`
	Arm             string  `json:"arm,omitempty"`
	ReportedSeconds float64 `json:"reported_seconds"`
	Turns           int     `json:"turns"`
	OutTokens       int     `json:"out_tokens"`
	PromptTokens    int     `json:"prompt_tokens"`
	ReasoningTokens int     `json:"reasoning_tokens"`
}

type runResult struct {
	ConfigName     string      `json:"config_name"`
	ConfigKey      string      `json:"config_key"`
	Warmup         bool        `json:"warmup"`
	GPUProfiler    string      `json:"gpu_profiler_status"`
	MemoryProfiler string      `json:"memory_profiler_status"`
	ThreadCount    int         `json:"threads"`
	Repetition     int         `json:"repetition"`
	RunDir         string      `json:"run_dir"`
	ExitCode       int         `json:"exit_code"`
	WallSeconds    float64     `json:"wall_seconds"`
	Client         clientStats `json:"client"`
	Llama          llamaStats  `json:"llama"`
	Memory         memoryStats `json:"memory"`
	GPU            gpuStats    `json:"gpu"`
	Metrics        metricStats `json:"metrics"`
	FitSpillBlocks int         `json:"fit_spill_blocks"`
	FitOutput      string      `json:"fit_output,omitempty"`
	Error          string      `json:"error,omitempty"`
}

type runMeta struct {
	StartedAt     string         `json:"started_at"`
	ConfigName    string         `json:"config_name"`
	ConfigKey     string         `json:"config_key"`
	Warmup        bool           `json:"warmup"`
	Threads       int            `json:"threads"`
	Repetition    int            `json:"repetition"`
	ClientCommand string         `json:"client_command"`
	Shell         string         `json:"shell"`
	ServerCommand []string       `json:"server_command"`
	ServerEnv     []string       `json:"server_env,omitempty"`
	ClientEnv     []string       `json:"client_env,omitempty"`
	GitCommit     string         `json:"git_commit,omitempty"`
	Config        map[string]any `json:"config"`
}

var (
	rePromptEval = regexp.MustCompile(`task\s+(\d+).+prompt eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens .*?([0-9.]+) tokens per second`)
	reEval       = regexp.MustCompile(`task\s+(\d+)\s+\|\s+eval time =\s*([0-9.]+) ms /\s*([0-9]+) tokens .*?([0-9.]+) tokens per second`)
	reProgress   = regexp.MustCompile(`task\s+(\d+).+n_gen =\s*([0-9]+), tg =\s*([0-9.]+) t/s, tg_3s =\s*([0-9.]+) t/s`)
	reLaunch     = regexp.MustCompile(`slot\s+launch_slot_: id\s+([0-9]+)\s+\|\s+task\s+([0-9]+)`)
	reCancel     = regexp.MustCompile(`cancel task, id_task =\s*([0-9]+)`)
	reStopCancel = regexp.MustCompile(`\bsrv\s+stop:\s+cancel task\b`)
	reLCP        = regexp.MustCompile(`f_sim_best =\s*([0-9.]+).*f_keep =\s*([0-9.]+)`)
	rePrefill    = regexp.MustCompile(`task\s+(\d+).+prompt processing, n_tokens =\s*([0-9]+)`)
	reClient     = regexp.MustCompile(`\[Target=([^ ]+) Task=([^ ]+) Variant=([^ ]+) Arm=([^]]+)\] (PASS|FAIL) \(took ([^,]+), turns:\s*([0-9]+), out_tokens:\s*([0-9]+)\)`)
	reFitBlock   = regexp.MustCompile(`blk[^0-9]*([0-9]+)`)
)

func main() {
	if runAuxiliaryCommand(os.Args[1:]) {
		return
	}
	cfg, threadValues := prepareBenchmarkConfig(parseFlags())
	writeEnvironmentSnapshot(cfg)
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	experiments, results := loadBenchmarkState(cfg, threadValues)
	fitByConfig, fitSpillByConfig := inspectExperimentFits(ctx, cfg, experiments)
	interrupted := runExperimentMatrix(ctx, cfg, threadValues, experiments, fitByConfig, fitSpillByConfig, &results)
	if interrupted {
		return
	}
	fmt.Println()
	printAggregateSummary(results)
	fmt.Printf("\nResults: %s\n", cfg.OutputDir)
}

func runAuxiliaryCommand(args []string) bool {
	if len(args) == 0 {
		return false
	}
	var err error
	switch args[0] {
	case "analyze":
		err = runAnalysisCommand(args[1:], os.Stdout)
	case "plan":
		err = runPlanCommand(args[1:], os.Stdout)
	default:
		return false
	}
	if err != nil {
		fatalf("%s: %v", args[0], err)
	}
	return true
}

func prepareBenchmarkConfig(cfg config) (config, []int) {
	invocationDir, err := os.Getwd()
	if err != nil {
		fatalf("current directory: %v", err)
	}
	threads, err := parseThreadValues(cfg.ThreadValues)
	if err != nil {
		fatalf("invalid --threads: %v", err)
	}
	if cfg.Runs < 1 {
		fatalf("--runs must be >= 1")
	}
	if cfg.WarmupRuns < 0 {
		fatalf("--warmup-runs must be >= 0")
	}
	cfg = resolveBenchmarkPaths(cfg, invocationDir)
	cfg = resolveProfilerTools(cfg)
	cfg = setupMemoryProfiler(cfg)
	return cfg, threads
}

func resolveBenchmarkPaths(cfg config, invocationDir string) config {
	var err error
	if cfg.ClientCmd == "" {
		promptScript, resolveErr := resolveFile(cfg.PromptScript, invocationDir)
		if resolveErr != nil {
			fatalf("prompt script: %v", resolveErr)
		}
		cfg.ClientCmd = "bash " + shellQuote(promptScript)
	}
	cfg.WorkDir, err = filepath.Abs(cfg.WorkDir)
	if err != nil {
		fatalf("workdir: %v", err)
	}
	cfg.OpenCodeConfig, err = resolveFile(cfg.OpenCodeConfig, invocationDir)
	if err != nil {
		fatalf("OpenCode config: %v", err)
	}
	cfg.OutputDir, err = filepath.Abs(cfg.OutputDir)
	if err != nil {
		fatalf("output dir: %v", err)
	}
	if err := os.MkdirAll(cfg.OutputDir, 0o755); err != nil {
		fatalf("create output dir: %v", err)
	}
	cfg.ServerBinary = resolveRequiredExecutable(cfg.ServerBinary, cfg.WorkDir, "server binary")
	cfg.Shell = resolveRequiredExecutable(cfg.Shell, cfg.WorkDir, "shell")
	return cfg
}

func resolveRequiredExecutable(path, workDir, label string) string {
	resolved, err := resolveExecutable(path, workDir)
	if err != nil {
		fatalf("%s: %v", label, err)
	}
	return resolved
}

func resolveProfilerTools(cfg config) config {
	var err error
	if cfg.InspectFit {
		cfg.FitBinary, err = resolveExecutable(cfg.FitBinary, cfg.WorkDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: fit binary unavailable: %v; disabling fit inspection\n", err)
			cfg.InspectFit = false
		}
	}
	if cfg.ProfileGPU {
		cfg.AmdSMI, err = resolveExecutable(cfg.AmdSMI, cfg.WorkDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: amd-smi unavailable: %v; disabling GPU profiling\n", err)
			cfg.ProfileGPU = false
		}
	}
	if cfg.ProfileMemory {
		cfg.UProf, err = resolveExecutable(cfg.UProf, cfg.WorkDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: AMDuProfPcm unavailable: %v; disabling memory profiling\n", err)
			cfg.ProfileMemory = false
		}
	}
	return cfg
}

func setupMemoryProfiler(cfg config) config {
	if !cfg.ProfileMemory || !cfg.UProfSudo {
		return cfg
	}
	fmt.Println("Acquiring sudo credentials for AMD uProf...")
	cmd := exec.Command("sudo", "-v")
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: sudo credentials unavailable: %v; disabling uProf\n", err)
		cfg.ProfileMemory = false
		return cfg
	}
	if cfg.ModprobeUncore {
		if err := exec.Command("sudo", "-n", "modprobe", "amd_uncore").Run(); err != nil {
			fmt.Fprintf(os.Stderr, "warning: sudo modprobe amd_uncore failed: %v\n", err)
		}
	}
	return cfg
}

func loadBenchmarkState(cfg config, threads []int) ([]experiment, []runResult) {
	experiments, err := expandExperiments(cfg)
	if err != nil {
		fatalf("experiment matrix: %v", err)
	}
	var results []runResult
	if cfg.Resume {
		results, err = loadResumableResults(cfg.OutputDir, experiments, threads, cfg.Runs)
		if err != nil {
			fatalf("load resumable results: %v", err)
		}
		fmt.Printf("Resuming %d successful counted runs from %s; failed or missing runs will be retried.\n", len(results), cfg.OutputDir)
	}
	return experiments, results
}

func inspectExperimentFits(ctx context.Context, cfg config, experiments []experiment) (map[string]string, map[string]int) {
	fitByConfig := make(map[string]string)
	fitSpillByConfig := make(map[string]int)
	if !cfg.InspectFit {
		return fitByConfig, fitSpillByConfig
	}
	for _, experiment := range experiments {
		if ctx.Err() != nil {
			fatalf("interrupted")
		}
		inspectExperimentFit(ctx, cfg, experiment, fitByConfig, fitSpillByConfig)
	}
	return fitByConfig, fitSpillByConfig
}

func inspectExperimentFit(ctx context.Context, cfg config, experiment experiment, fitByConfig map[string]string, fitSpillByConfig map[string]int) {
	dir := filepath.Join(cfg.OutputDir, "configs", experiment.Name)
	fitPath := filepath.Join(dir, "fit.txt")
	if cfg.Resume {
		if output, err := os.ReadFile(fitPath); err == nil && len(output) > 0 {
			fitByConfig[experiment.Key] = string(output)
			fitSpillByConfig[experiment.Key] = countFitSpillBlocks(string(output))
			return
		}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "warning: create fit directory: %v\n", err)
		return
	}
	output, fitErr := inspectFit(ctx, experiment.Config)
	if fitErr != nil {
		fmt.Fprintf(os.Stderr, "warning: fit inspection %s failed: %v\n", experiment.Name, fitErr)
	}
	fitByConfig[experiment.Key] = output
	fitSpillByConfig[experiment.Key] = countFitSpillBlocks(output)
	if err := os.WriteFile(fitPath, []byte(output), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "warning: write fit output: %v\n", err)
	}
}

func runExperimentMatrix(ctx context.Context, cfg config, threadValues []int, experiments []experiment, fitByConfig map[string]string, fitSpillByConfig map[string]int, results *[]runResult) bool {
	for _, experiment := range experiments {
		for _, threads := range experimentThreadValues(experiment, threadValues) {
			if shouldSkipCell(cfg, experiment, threads) {
				continue
			}
			interrupted, err := runExperimentCell(ctx, cfg, experiment, threads, fitByConfig, fitSpillByConfig, results)
			if err != nil {
				fatalf("benchmark cell failed: %v", err)
			}
			if interrupted {
				return true
			}
		}
	}
	return false
}

func shouldSkipCell(cfg config, experiment experiment, threads int) bool {
	if cfg.Resume && cellComplete(cfg.OutputDir, experiment, threads, cfg.Runs, cfg.WarmupRuns) {
		fmt.Printf("\n=== already complete %s threads=%d; skipping ===\n", experiment.Name, threads)
		return true
	}
	return false
}

func runExperimentCell(ctx context.Context, cfg config, experiment experiment, threads int, fitByConfig map[string]string, fitSpillByConfig map[string]int, results *[]runResult) (bool, error) {
	if ctx.Err() != nil {
		writePartialSummaries(cfg.OutputDir, *results)
		return true, nil
	}
	fmt.Printf("\n=== starting %s threads=%d ===\n", experiment.Name, threads)
	session, err := startServerSession(ctx, experiment.Config, experiment.Name, threads)
	if err != nil {
		return false, fmt.Errorf("start server for %s threads=%d: %w", experiment.Name, threads, err)
	}
	if err := runWarmups(ctx, cfg, experiment, threads, fitByConfig, fitSpillByConfig, session); err != nil {
		session.stop()
		return false, err
	}
	if err := runCountedRepetitions(ctx, cfg, experiment, threads, fitByConfig, fitSpillByConfig, session, results); err != nil {
		session.stop()
		return false, err
	}
	session.stop()
	if ctx.Err() != nil {
		writePartialSummaries(cfg.OutputDir, *results)
		return true, nil
	}
	return false, nil
}

func runWarmups(ctx context.Context, cfg config, experiment experiment, threads int, fitByConfig map[string]string, fitSpillByConfig map[string]int, session *serverSession) error {
	for warmup := 1; warmup <= cfg.WarmupRuns && ctx.Err() == nil; warmup++ {
		fmt.Printf("\n=== warm-up %d/%d %s threads=%d ===\n", warmup, cfg.WarmupRuns, experiment.Name, threads)
		result := executeRun(ctx, experiment.Config, experiment.Name, experiment.Key, threads, -warmup,
			fitByConfig[experiment.Key], fitSpillByConfig[experiment.Key], session)
		if result.ExitCode != 0 {
			return fmt.Errorf("warm-up failed in %s: %s", result.RunDir, result.Error)
		}
	}
	return nil
}

func runCountedRepetitions(ctx context.Context, cfg config, experiment experiment, threads int, fitByConfig map[string]string, fitSpillByConfig map[string]int, session *serverSession, results *[]runResult) error {
	for rep := 1; rep <= cfg.Runs && ctx.Err() == nil; rep++ {
		if cfg.Resume && resumableResult(cfg.OutputDir, experiment, threads, rep) {
			fmt.Printf("\n=== %s threads=%d run=%d/%d already complete; skipping ===\n", experiment.Name, threads, rep, cfg.Runs)
			continue
		}
		fmt.Printf("\n=== %s threads=%d run=%d/%d ===\n", experiment.Name, threads, rep, cfg.Runs)
		result := executeRun(ctx, experiment.Config, experiment.Name, experiment.Key, threads, rep,
			fitByConfig[experiment.Key], fitSpillByConfig[experiment.Key], session)
		*results = append(*results, result)
		printRunSummary(result)
		if err := writeAllSummaries(cfg.OutputDir, *results); err != nil {
			return fmt.Errorf("write summaries: %w", err)
		}
		if result.ExitCode != 0 && cfg.FailFast {
			return fmt.Errorf("client failed in %s", result.RunDir)
		}
	}
	return nil
}

func writePartialSummaries(outputDir string, results []runResult) {
	if err := writeAllSummaries(outputDir, results); err != nil {
		fmt.Fprintf(os.Stderr, "write partial summaries: %v\n", err)
	}
	fmt.Println("Interrupted; partial results written.")
}

func parseFlags() config {
	var cfg config
	registerRunFlags(&cfg)
	registerMatrixFlags(&cfg)
	registerModelFlags(&cfg)
	registerProfilerFlags(&cfg)
	flag.Var(&cfg.ServerArgs, "server-arg", "extra llama-server argument; repeat for multiple arguments")
	flag.Var(&cfg.ServerEnv, "server-env", "extra server environment KEY=VALUE; repeatable")
	flag.Var(&cfg.ClientEnv, "client-env", "extra client environment KEY=VALUE; repeatable")
	flag.Parse()
	return cfg
}

func registerRunFlags(cfg *config) {
	flag.StringVar(&cfg.WorkDir, "workdir", "../llama.cpp", "llama.cpp checkout root")
	flag.StringVar(&cfg.OutputDir, "out", "./benchmark-runs", "output directory")
	flag.StringVar(&cfg.ClientCmd, "client-cmd", "", "client benchmark command, executed through --shell (default: standalone prompt run.sh)")
	flag.StringVar(&cfg.PromptScript, "prompt-script", "standalone-prompt/run.sh", "standalone prompt script used when --client-cmd is empty")
	flag.StringVar(&cfg.OpenCodeConfig, "opencode-config", "standalone-prompt/opencode.json", "OpenCode config template; endpoint is set to this run's llama-server")
	flag.StringVar(&cfg.Shell, "shell", "/bin/bash", "shell used for --client-cmd")
	flag.IntVar(&cfg.Runs, "runs", 7, "runs per THREADS value")
	flag.IntVar(&cfg.WarmupRuns, "warmup-runs", 0, "warm-up runs per configuration, excluded from summaries")
	flag.StringVar(&cfg.ThreadValues, "threads", "6,8,12", "comma-separated llama.cpp CPU thread counts")
	flag.BoolVar(&cfg.FailFast, "fail-fast", false, "stop on first client failure")
	flag.BoolVar(&cfg.Resume, "resume", false, "resume an existing output directory, reusing only successful matching runs")
	flag.DurationVar(&cfg.ClientTimeout, "client-timeout", 30*time.Minute, "maximum duration of one client benchmark")
	flag.DurationVar(&cfg.HealthTimeout, "health-timeout", 10*time.Minute, "server startup/health timeout")
	flag.DurationVar(&cfg.ServerSettle, "server-settle", 500*time.Millisecond, "delay after server becomes healthy")
}

func registerMatrixFlags(cfg *config) {
	flag.StringVar(&cfg.ContextValues, "ctx-values", "", "total context/KV pool values; combines with other axes as a Cartesian product")
	flag.StringVar(&cfg.ParallelValues, "parallel-values", "", "parallel slot values; combines with other axes as a Cartesian product")
	flag.StringVar(&cfg.BatchValues, "batch-values", "", "logical batch values; combines with other axes as a Cartesian product")
	flag.StringVar(&cfg.UBatchValues, "ubatch-values", "", "microbatch values; combines with other axes as a Cartesian product")
	flag.StringVar(&cfg.FitTargetValues, "fit-target-values", "", "fit target values; combines with other axes as a Cartesian product")
	flag.StringVar(&cfg.KVKValues, "kv-k-values", "", "K cache values; combines with other axes as a Cartesian product")
	flag.StringVar(&cfg.KVVValues, "kv-v-values", "", "V cache values; combines with other axes as a Cartesian product")
	flag.StringVar(&cfg.ModelValues, "model-values", "", "model values; combines with other axes as a Cartesian product")
	flag.StringVar(&cfg.PlanFile, "plan", "", "JSON pairwise coverage plan produced by the plan subcommand")
	flag.StringVar(&cfg.PreserveThinkValues, "preserve-thinking-values", "", "true/false prompt-session types; combines with other axes as a Cartesian product")
	flag.StringVar(&cfg.CacheReuseValues, "cache-reuse-values", "", "cache-reuse chunk values; combines with other axes as a Cartesian product")
	flag.StringVar(&cfg.FlashAttentionValues, "flash-attn-values", "", "FlashAttention values (on/off/auto); combines with other axes as a Cartesian product")
}

func registerModelFlags(cfg *config) {
	flag.StringVar(&cfg.ServerBinary, "server", "./build-vulkan/bin/llama-server", "llama-server binary")
	flag.StringVar(&cfg.FitBinary, "fit-binary", "./build-vulkan/bin/llama-fit-params", "llama-fit-params binary")
	flag.BoolVar(&cfg.InspectFit, "inspect-fit", true, "run llama-fit-params once for each server configuration")
	flag.StringVar(&cfg.Model, "model", "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q3_K_M", "Hugging Face repo[:quant]")
	flag.StringVar(&cfg.Alias, "alias", "qwen36-coder", "server model alias")
	flag.StringVar(&cfg.Host, "host", "127.0.0.1", "llama-server bind host")
	flag.IntVar(&cfg.Port, "port", 1234, "llama-server port")
	flag.IntVar(&cfg.Context, "ctx", 65536, "total context/KV pool tokens")
	flag.IntVar(&cfg.Parallel, "parallel", 2, "llama-server parallel slots")
	flag.BoolVar(&cfg.UnifiedKV, "kv-unified", true, "enable unified KV")
	flag.IntVar(&cfg.KVPerSlot, "kv-per-slot", 65536, "maximum context per slot when unified KV is enabled")
	flag.StringVar(&cfg.KVK, "kv-k", "q5_1", "K cache quantization")
	flag.StringVar(&cfg.KVV, "kv-v", "q5_1", "V cache quantization")
	flag.IntVar(&cfg.Batch, "batch", 2048, "logical batch size")
	flag.IntVar(&cfg.UBatch, "ubatch", 256, "physical microbatch size")
	flag.IntVar(&cfg.FitTarget, "fit-target", 512, "VRAM fit target/headroom MiB")
	flag.StringVar(&cfg.NGL, "ngl", "auto", "GPU layer placement for llama-server")
	flag.IntVar(&cfg.NCPUMoE, "n-cpu-moe", 0, "explicit CPU MoE layers")
	flag.StringVar(&cfg.FlashAttention, "flash-attn", "on", "FlashAttention: on/off/auto")
	flag.IntVar(&cfg.CacheReuse, "cache-reuse", 0, "cache-reuse chunk size")
	flag.StringVar(&cfg.LoadMode, "load-mode", "auto", "model load mode")
	flag.StringVar(&cfg.LazyMode, "lazy-mode", "auto", "lazy tensor mode")
	flag.BoolVar(&cfg.Reasoning, "reasoning", true, "enable reasoning")
	flag.BoolVar(&cfg.PreserveThink, "preserve-thinking", false, "preserve thinking blocks in chat templates and prompt/session metadata")
	flag.Float64Var(&cfg.Temp, "temp", 0.6, "sampling temperature")
	flag.Float64Var(&cfg.TopP, "top-p", 0.95, "top-p")
	flag.IntVar(&cfg.TopK, "top-k", 20, "top-k")
	flag.Float64Var(&cfg.StarvationThreshold, "starvation-threshold", 1.0, "tg_3s tokens/sec threshold for a decode-starvation event")
	flag.Int64Var(&cfg.LargePrefillTokens, "large-prefill-tokens", 2048, "minimum prompt tokens to count as a large prefill")
	flag.IntVar(&cfg.LogVerbosity, "log-verbosity", 3, "llama.cpp log verbosity")
}

func registerProfilerFlags(cfg *config) {
	flag.StringVar(&cfg.AmdSMI, "amd-smi", "amd-smi", "amd-smi executable")
	flag.IntVar(&cfg.GPU, "gpu", 0, "AMD GPU index")
	flag.BoolVar(&cfg.ProfileGPU, "profile-gpu", true, "capture amd-smi GPU metrics")
	flag.StringVar(&cfg.UProf, "uprof", "AMDuProfPcm", "AMDuProfPcm executable")
	flag.BoolVar(&cfg.ProfileMemory, "profile-memory", true, "capture AMD uProf memory bandwidth")
	flag.BoolVar(&cfg.UProfSudo, "uprof-sudo", true, "run AMDuProfPcm under sudo")
	flag.BoolVar(&cfg.ModprobeUncore, "modprobe-amd-uncore", true, "sudo modprobe amd_uncore before profiling")
}

func parseThreadValues(s string) ([]int, error) {
	var out []int
	seen := map[int]bool{}
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil || n < 1 {
			return nil, fmt.Errorf("%q is not a positive integer", p)
		}
		if !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("no thread counts supplied")
	}
	return out, nil
}

func loadResumableResults(root string, experiments []experiment, threads []int, runs int) ([]runResult, error) {
	results := make([]runResult, 0)
	for _, experiment := range experiments {
		for _, thread := range experimentThreadValues(experiment, threads) {
			for rep := 1; rep <= runs; rep++ {
				path := resultPath(root, experiment, thread, rep)
				data, err := os.ReadFile(path)
				if errors.Is(err, os.ErrNotExist) {
					continue
				}
				if err != nil {
					return nil, fmt.Errorf("read %s: %w", path, err)
				}
				var result runResult
				if err := json.Unmarshal(data, &result); err != nil {
					continue
				}
				if validResumableResult(result, experiment, thread, rep) {
					results = append(results, result)
				}
			}
		}
	}
	return results, nil
}

func experimentThreadValues(experiment experiment, defaults []int) []int {
	if len(experiment.Threads) > 0 {
		return experiment.Threads
	}
	return defaults
}

func cellComplete(root string, experiment experiment, threads, runs, warmups int) bool {
	for warmup := 1; warmup <= warmups; warmup++ {
		if !resumableResult(root, experiment, threads, -warmup) {
			return false
		}
	}
	for rep := 1; rep <= runs; rep++ {
		if !resumableResult(root, experiment, threads, rep) {
			return false
		}
	}
	return true
}

func resumableResult(root string, experiment experiment, threads, rep int) bool {
	data, err := os.ReadFile(resultPath(root, experiment, threads, rep))
	if err != nil {
		return false
	}
	var result runResult
	if err := json.Unmarshal(data, &result); err != nil {
		return false
	}
	return validResumableResult(result, experiment, threads, rep)
}

func validResumableResult(result runResult, experiment experiment, threads, rep int) bool {
	return result.ConfigKey == experiment.Key &&
		result.ThreadCount == threads && result.Repetition == rep &&
		result.Warmup == (rep < 0) && result.ExitCode == 0 &&
		result.Client.Parsed && result.Client.Passed
}

func resultPath(root string, experiment experiment, threads, rep int) string {
	return filepath.Join(experimentRoot(root, experiment.Name), fmt.Sprintf("threads-%02d", threads), repetitionDir(rep), "result.json")
}

func expandExperiments(base config) ([]experiment, error) {
	if strings.TrimSpace(base.PlanFile) != "" {
		return expandCoveragePlan(base)
	}
	configs, err := expandConfigAxes(base, experimentAxes(base))
	if err != nil {
		return nil, err
	}
	return experimentsFromConfigs(base, configs), nil
}

func axisValuesWithBase(baseValue, values string) []string {
	out := []string{baseValue}
	seen := map[string]bool{baseValue: true}
	for _, value := range splitAxis(values) {
		value = strings.TrimSpace(value)
		if value != "" && !seen[value] {
			out = append(out, value)
			seen[value] = true
		}
	}
	return out
}

func intExperimentAxis(name, values string, baseValue int, set func(*config, int)) (experimentAxis, bool) {
	if strings.TrimSpace(values) == "" {
		return experimentAxis{}, false
	}
	return experimentAxis{
		name: name, values: axisValuesWithBase(strconv.Itoa(baseValue), values),
		apply: func(c *config, value string) error {
			n, err := strconv.Atoi(value)
			if err != nil || n < 1 {
				return errors.New("must be a positive integer")
			}
			set(c, n)
			return nil
		},
	}, true
}

func stringExperimentAxis(name, values, baseValue string, set func(*config, string)) (experimentAxis, bool) {
	if strings.TrimSpace(values) == "" {
		return experimentAxis{}, false
	}
	return experimentAxis{
		name: name, values: axisValuesWithBase(baseValue, values),
		apply: func(c *config, value string) error { set(c, value); return nil },
	}, true
}

func experimentAxes(base config) []experimentAxis {
	var axes []experimentAxis
	add := func(axis experimentAxis, ok bool) {
		if ok {
			axes = append(axes, axis)
		}
	}
	add(intExperimentAxis("parallel", base.ParallelValues, base.Parallel, func(c *config, v int) { c.Parallel = v }))
	add(intExperimentAxis("context", base.ContextValues, base.Context, func(c *config, v int) { c.Context = v }))
	add(intExperimentAxis("batch", base.BatchValues, base.Batch, func(c *config, v int) { c.Batch = v }))
	add(intExperimentAxis("ubatch", base.UBatchValues, base.UBatch, func(c *config, v int) { c.UBatch = v }))
	add(intExperimentAxis("fit-target", base.FitTargetValues, base.FitTarget, func(c *config, v int) { c.FitTarget = v }))
	add(stringExperimentAxis("kv-k", base.KVKValues, base.KVK, func(c *config, v string) { c.KVK = v }))
	add(stringExperimentAxis("kv-v", base.KVVValues, base.KVV, func(c *config, v string) { c.KVV = v }))
	add(stringExperimentAxis("model", base.ModelValues, base.Model, func(c *config, v string) { c.Model = v }))
	return appendOptionalExperimentAxes(base, axes)
}

func appendOptionalExperimentAxes(base config, axes []experimentAxis) []experimentAxis {
	if strings.TrimSpace(base.CacheReuseValues) != "" {
		axes = append(axes, experimentAxis{name: "cache-reuse", values: axisValuesWithBase(strconv.Itoa(base.CacheReuse), base.CacheReuseValues), apply: setCacheReuse})
	}
	if strings.TrimSpace(base.FlashAttentionValues) != "" {
		axes = append(axes, experimentAxis{name: "flash-attention", values: axisValuesWithBase(base.FlashAttention, base.FlashAttentionValues), apply: setFlashAttention})
	}
	if strings.TrimSpace(base.PreserveThinkValues) != "" {
		axes = append(axes, experimentAxis{name: "preserve-thinking", values: axisValuesWithBase(strconv.FormatBool(base.PreserveThink), base.PreserveThinkValues), apply: setPreserveThinking})
	}
	return axes
}

func setCacheReuse(c *config, value string) error {
	n, err := strconv.Atoi(value)
	if err != nil || n < 0 {
		return errors.New("must be a non-negative integer")
	}
	c.CacheReuse = n
	return nil
}

func setFlashAttention(c *config, value string) error {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "on", "off", "auto":
		c.FlashAttention = value
		return nil
	default:
		return errors.New("must be on, off, or auto")
	}
}

func setPreserveThinking(c *config, value string) error {
	preserve, err := strconv.ParseBool(value)
	if err != nil {
		return errors.New("must be true or false")
	}
	c.PreserveThink = preserve
	return nil
}

func expandConfigAxes(base config, axes []experimentAxis) ([]config, error) {
	configs := []config{base}
	for _, axis := range axes {
		next := make([]config, 0, len(configs)*len(axis.values))
		for _, partial := range configs {
			for _, value := range axis.values {
				candidate := partial
				if err := axis.apply(&candidate, value); err != nil {
					return nil, fmt.Errorf("%s value %q: %w", axis.name, value, err)
				}
				next = append(next, candidate)
			}
		}
		configs = next
	}
	return configs, nil
}

func experimentsFromConfigs(base config, configs []config) []experiment {
	out := make([]experiment, 0, len(configs))
	seen := make(map[string]bool)
	for _, candidate := range configs {
		key := configKey(candidate)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, experiment{Name: experimentName(base, candidate), Key: key, Config: candidate})
	}
	return out
}

func expandCoveragePlan(base config) ([]experiment, error) {
	data, err := os.ReadFile(base.PlanFile)
	if err != nil {
		return nil, fmt.Errorf("read pairwise plan %q: %w", base.PlanFile, err)
	}
	var cells []coverageCell
	if err := json.Unmarshal(data, &cells); err != nil {
		return nil, fmt.Errorf("parse pairwise plan %q: %w", base.PlanFile, err)
	}
	if len(cells) == 0 {
		return nil, errors.New("pairwise plan contains no cells")
	}
	byKey := make(map[string]int, len(cells))
	out := make([]experiment, 0, len(cells))
	seenCells := make(map[string]bool, len(cells))
	for i, cell := range cells {
		s := cell.Settings
		if strings.TrimSpace(s.Model) == "" || s.Context < 1 || s.Parallel < 1 || s.Batch < 1 ||
			s.UBatch < 1 || s.FitTarget < 1 || s.KVPerSlot < 1 || cell.Threads < 1 {
			return nil, fmt.Errorf("pairwise plan cell %d has an empty model or non-positive numeric setting", i+1)
		}
		if s.CacheReuse < 0 || (s.FlashAttention != "on" && s.FlashAttention != "off" && s.FlashAttention != "auto") {
			return nil, fmt.Errorf("pairwise plan cell %d has invalid cache-reuse or flash-attention setting", i+1)
		}
		candidate := base
		candidate.Model, candidate.Context, candidate.Parallel = s.Model, s.Context, s.Parallel
		candidate.Batch, candidate.UBatch, candidate.FitTarget = s.Batch, s.UBatch, s.FitTarget
		candidate.CacheReuse, candidate.FlashAttention = s.CacheReuse, s.FlashAttention
		candidate.UnifiedKV, candidate.KVPerSlot = s.UnifiedKV, s.KVPerSlot
		candidate.KVK, candidate.KVV = s.KVK, s.KVV
		candidate.PreserveThink = s.PreserveThinking
		key := configKey(candidate)
		cellKey := fmt.Sprintf("%s|%d", key, cell.Threads)
		if seenCells[cellKey] {
			return nil, fmt.Errorf("pairwise plan repeats setting/thread cell %q", cellKey)
		}
		seenCells[cellKey] = true
		index, ok := byKey[key]
		if !ok {
			index = len(out)
			byKey[key] = index
			out = append(out, experiment{Name: experimentName(base, candidate), Key: key, Config: candidate})
		}
		out[index].Threads = append(out[index].Threads, cell.Threads)
	}
	return out, nil
}

func experimentName(base, candidate config) string {
	var parts []string
	add := func(name, value string) {
		parts = append(parts, name+"-"+safeName(value))
	}
	if candidate.Parallel != base.Parallel {
		add("parallel", strconv.Itoa(candidate.Parallel))
	}
	if candidate.Context != base.Context {
		add("ctx", strconv.Itoa(candidate.Context))
	}
	if candidate.Batch != base.Batch {
		add("batch", strconv.Itoa(candidate.Batch))
	}
	if candidate.UBatch != base.UBatch {
		add("ubatch", strconv.Itoa(candidate.UBatch))
	}
	if candidate.FitTarget != base.FitTarget {
		add("fit-target", strconv.Itoa(candidate.FitTarget))
	}
	if candidate.KVK != base.KVK {
		add("kv-k", candidate.KVK)
	}
	if candidate.KVV != base.KVV {
		add("kv-v", candidate.KVV)
	}
	if candidate.Model != base.Model {
		add("model", candidate.Model)
	}
	if candidate.CacheReuse != base.CacheReuse {
		add("cache-reuse", strconv.Itoa(candidate.CacheReuse))
	}
	if candidate.FlashAttention != base.FlashAttention {
		add("flash-attn", candidate.FlashAttention)
	}
	if candidate.PreserveThink != base.PreserveThink {
		if candidate.PreserveThink {
			add("preserve-thinking", "on")
		} else {
			add("preserve-thinking", "off")
		}
	}
	if len(parts) == 0 {
		return "base"
	}
	return strings.Join(parts, "_")
}

func splitAxis(values string) []string {
	if strings.TrimSpace(values) == "" {
		return nil
	}
	return strings.Split(values, ",")
}

func safeName(s string) string {
	var out strings.Builder
	for _, r := range strings.ToLower(s) {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			out.WriteRune(r)
		} else {
			out.WriteByte('-')
		}
	}
	return strings.Trim(out.String(), "-")
}

func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "'\\''") + "'"
}

func configKey(cfg config) string {
	return strings.Join([]string{
		cfg.Model, strconv.Itoa(cfg.Context), strconv.Itoa(cfg.Parallel), cfg.KVK, cfg.KVV,
		strconv.Itoa(cfg.Batch), strconv.Itoa(cfg.UBatch), strconv.Itoa(cfg.FitTarget),
		strconv.Itoa(cfg.CacheReuse), cfg.FlashAttention, strconv.FormatBool(cfg.UnifiedKV), strconv.Itoa(cfg.KVPerSlot),
		strconv.FormatBool(cfg.PreserveThink),
	}, "|")
}

func resolveExecutable(name, workdir string) (string, error) {
	if strings.ContainsRune(name, os.PathSeparator) {
		if !filepath.IsAbs(name) {
			name = filepath.Join(workdir, name)
		}
		abs, err := filepath.Abs(name)
		if err != nil {
			return "", err
		}
		if st, err := os.Stat(abs); err != nil {
			return "", err
		} else if st.IsDir() {
			return "", fmt.Errorf("%s is a directory", abs)
		}
		return abs, nil
	}
	return exec.LookPath(name)
}

func resolveFile(name, workdir string) (string, error) {
	if !filepath.IsAbs(name) {
		name = filepath.Join(workdir, name)
	}
	path, err := filepath.Abs(name)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("%s is a directory", path)
	}
	return path, nil
}

func serverArgs(cfg config, threads int, logFile string) []string {
	args := []string{
		"-hf", cfg.Model,
		"--alias", cfg.Alias,
		"--host", cfg.Host,
		"--port", strconv.Itoa(cfg.Port),
		"--jinja",
		"-c", strconv.Itoa(cfg.Context),
		"-np", strconv.Itoa(cfg.Parallel),
		"-ngl", cfg.NGL,
		"-ncmoe", strconv.Itoa(cfg.NCPUMoE),
		"-fit", "on",
		"-fitt", strconv.Itoa(cfg.FitTarget),
		"-fa", cfg.FlashAttention,
		"-ctk", cfg.KVK,
		"-ctv", cfg.KVV,
		"-b", strconv.Itoa(cfg.Batch),
		"-ub", strconv.Itoa(cfg.UBatch),
		"-t", strconv.Itoa(threads),
		"-tb", strconv.Itoa(threads),
		"-lm", cfg.LoadMode,
		"-lzm", cfg.LazyMode,
		"--cache-reuse", strconv.Itoa(cfg.CacheReuse),
		"--no-context-shift",
		"--chat-template-kwargs", fmt.Sprintf(`{"preserve_thinking":%t}`, cfg.PreserveThink),
		"--metrics",
		"--log-file", logFile,
		"--log-verbosity", strconv.Itoa(cfg.LogVerbosity),
		"--temp", fmt.Sprintf("%g", cfg.Temp),
		"--top-p", fmt.Sprintf("%g", cfg.TopP),
		"--top-k", strconv.Itoa(cfg.TopK),
	}
	if cfg.Reasoning {
		args = append(args, "--reasoning", "on")
	} else {
		args = append(args, "--reasoning", "off")
	}
	if cfg.UnifiedKV {
		args = append(args, "--kv-unified")
		if cfg.KVPerSlot > 0 {
			args = append(args, "--kv-unified-per-slot", strconv.Itoa(cfg.KVPerSlot))
		}
	}
	args = append(args, cfg.ServerArgs...)
	return args
}

type serverSession struct {
	proc    *managedProc
	baseURL string
	logPath string
}

func (s *serverSession) stop() {
	if s != nil {
		s.proc.stop(syscall.SIGTERM, 5*time.Second, false)
	}
}

func experimentRoot(outputDir, configName string) string {
	if configName == "base" {
		return outputDir
	}
	return filepath.Join(outputDir, "configs", configName)
}

func startServerSession(parent context.Context, cfg config, configName string, threads int) (*serverSession, error) {
	groupDir := filepath.Join(experimentRoot(cfg.OutputDir, configName), fmt.Sprintf("threads-%02d", threads))
	if err := os.MkdirAll(groupDir, 0o755); err != nil {
		return nil, err
	}
	logPath := filepath.Join(groupDir, "server.log")
	consolePath := filepath.Join(groupDir, "server-console.log")
	if err := os.Remove(logPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	consoleFile, err := os.Create(consolePath)
	if err != nil {
		return nil, err
	}
	args := serverArgs(cfg, threads, logPath)
	cmd := exec.Command(cfg.ServerBinary, args...)
	cmd.Dir = cfg.WorkDir
	cmd.Env = envWith(os.Environ(), cfg.ServerEnv)
	cmd.Stdout = consoleFile
	cmd.Stderr = consoleFile
	proc, startErr := startManaged(cmd)
	_ = consoleFile.Close()
	if startErr != nil {
		return nil, startErr
	}
	session := &serverSession{
		proc:    proc,
		baseURL: "http://" + net.JoinHostPort(connectHost(cfg.Host), strconv.Itoa(cfg.Port)),
		logPath: logPath,
	}
	if err := waitHealthy(parent, session.baseURL+"/health", cfg.HealthTimeout, proc); err != nil {
		session.stop()
		return nil, err
	}
	if cfg.ServerSettle > 0 {
		timer := time.NewTimer(cfg.ServerSettle)
		defer timer.Stop()
		select {
		case <-parent.Done():
			session.stop()
			return nil, parent.Err()
		case <-timer.C:
		}
	}
	return session, nil
}

func logFileSize(path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

func copyLogRange(source, destination string, start, end int64) error {
	if end < start {
		return fmt.Errorf("invalid log range %d:%d", start, end)
	}
	src, err := os.Open(source)
	if err != nil {
		return err
	}
	defer src.Close()
	dst, err := os.Create(destination)
	if err != nil {
		return err
	}
	defer dst.Close()
	_, err = io.Copy(dst, io.NewSectionReader(src, start, end-start))
	return err
}

func fitArgs(cfg config) []string {
	args := []string{
		"-hf", cfg.Model,
		"-c", strconv.Itoa(cfg.Context),
		"-np", strconv.Itoa(cfg.Parallel),
		"-ctk", cfg.KVK,
		"-ctv", cfg.KVV,
		"-b", strconv.Itoa(cfg.Batch),
		"-ub", strconv.Itoa(cfg.UBatch),
		"-fa", cfg.FlashAttention,
		"-fitt", strconv.Itoa(cfg.FitTarget),
	}
	return args
}

type runExecution struct {
	runDir   string
	llamaLog string
	logStart int64
}

func executeRun(parent context.Context, cfg config, configName, configKey string, threads, rep int, fitOut string, fitSpill int, session *serverSession) runResult {
	execution, failed := prepareRun(cfg, configName, configKey, threads, rep, fitOut, fitSpill, session)
	if failed != nil {
		return *failed
	}
	profilers := startRunProfilers(cfg, threads, rep, execution.runDir)
	return collectRun(parent, cfg, configName, configKey, threads, rep, fitOut, fitSpill, session, *execution, profilers)
}

func prepareRun(cfg config, configName, configKey string, threads, rep int, fitOut string, fitSpill int, session *serverSession) (*runExecution, *runResult) {
	root := experimentRoot(cfg.OutputDir, configName)
	runDir := filepath.Join(root, fmt.Sprintf("threads-%02d", threads), repetitionDir(rep))
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		failed := runResult{ConfigName: configName, ConfigKey: configKey, ThreadCount: threads, Repetition: rep, RunDir: runDir, ExitCode: -1, Error: err.Error()}
		return nil, &failed
	}
	opencodeConfigPath := filepath.Join(runDir, "opencode.json")
	if err := writeRunOpenCodeConfig(cfg.OpenCodeConfig, opencodeConfigPath, session.baseURL+"/v1", cfg.Context); err != nil {
		failed := failedRun(configName, configKey, threads, rep, runDir, fitOut, fitSpill, fmt.Errorf("prepare OpenCode config: %w", err))
		return nil, &failed
	}

	llamaLog := filepath.Join(runDir, "llama.log")
	serverExited, serverErr := session.proc.exited()
	if serverExited {
		if serverErr == nil {
			serverErr = errors.New("server exited")
		}
		failed := failedRun(configName, configKey, threads, rep, runDir, fitOut, fitSpill, fmt.Errorf("server unavailable: %w", serverErr))
		return nil, &failed
	}
	if err := writeRunMetadata(cfg, configName, configKey, threads, rep, runDir, opencodeConfigPath, session); err != nil {
		failed := failedRun(configName, configKey, threads, rep, runDir, fitOut, fitSpill, fmt.Errorf("write metadata: %w", err))
		return nil, &failed
	}

	logStart, err := logFileSize(session.logPath)
	if err != nil {
		failed := failedRun(configName, configKey, threads, rep, runDir, fitOut, fitSpill, fmt.Errorf("server log: %w", err))
		return nil, &failed
	}
	return &runExecution{runDir: runDir, llamaLog: llamaLog, logStart: logStart}, nil
}

func writeRunMetadata(cfg config, configName, configKey string, threads, rep int, runDir, opencodeConfigPath string, session *serverSession) error {
	meta := runMeta{
		StartedAt: time.Now().Format(time.RFC3339Nano), ConfigName: configName,
		ConfigKey: configKey, Warmup: rep < 0, Threads: threads, Repetition: rep,
		ClientCommand: cfg.ClientCmd, Shell: cfg.Shell,
		ServerCommand: append([]string{cfg.ServerBinary}, serverArgs(cfg, threads, session.logPath)...),
		ServerEnv:     append([]string(nil), cfg.ServerEnv...),
		ClientEnv:     clientEnvironment(cfg, threads, rep, runDir, opencodeConfigPath),
		GitCommit:     gitCommit(cfg.WorkDir),
		Config: map[string]any{
			"model": cfg.Model, "ctx": cfg.Context, "parallel": cfg.Parallel,
			"reasoning": cfg.Reasoning, "preserve_thinking": cfg.PreserveThink,
			"prompt_session_type": promptSessionType(cfg.PreserveThink),
			"kv_k":                cfg.KVK, "kv_v": cfg.KVV, "batch": cfg.Batch, "ubatch": cfg.UBatch,
			"fit_target": cfg.FitTarget, "cache_reuse": cfg.CacheReuse, "unified_kv": cfg.UnifiedKV,
			"flash_attention": cfg.FlashAttention, "kv_per_slot": cfg.KVPerSlot,
			"starvation_threshold_tps": cfg.StarvationThreshold,
			"large_prefill_tokens":     cfg.LargePrefillTokens,
		},
	}
	return writeJSON(filepath.Join(runDir, "meta.json"), meta)
}

type runProfilers struct {
	gpuProc      *managedProc
	uprofProc    *managedProc
	gpuStatus    string
	memoryStatus string
}

func startRunProfilers(cfg config, threads, rep int, runDir string) runProfilers {
	profilers := runProfilers{gpuStatus: "disabled", memoryStatus: "disabled"}
	if cfg.ProfileGPU {
		profilers.gpuProc, profilers.gpuStatus = startGPUProfiler(cfg, runDir)
	}
	if cfg.ProfileMemory {
		profilers.uprofProc, profilers.memoryStatus = startMemoryProfiler(cfg, runDir)
	}
	return profilers
}

func startGPUProfiler(cfg config, runDir string) (*managedProc, string) {
	status := "unavailable"
	gpuFile, err := os.Create(filepath.Join(runDir, "gpu.csv"))
	if err != nil {
		return nil, status
	}
	gpuErr, err := os.Create(filepath.Join(runDir, "gpu.err.log"))
	if err != nil {
		_ = gpuFile.Close()
		return nil, status
	}
	cmd := exec.Command(cfg.AmdSMI, "monitor", "--gpu", strconv.Itoa(cfg.GPU), "--power-usage", "--gfx", "--mem", "--vram-usage", "--watch", "1", "--csv")
	cmd.Dir = cfg.WorkDir
	cmd.Stdout, cmd.Stderr = gpuFile, gpuErr
	proc, startErr := startManaged(cmd)
	_ = gpuFile.Close()
	_ = gpuErr.Close()
	if startErr != nil {
		status = "failed to start: " + startErr.Error()
		fmt.Fprintf(os.Stderr, "warning: amd-smi start failed: %v\n", startErr)
		return nil, status
	}
	return proc, "running"
}

func startMemoryProfiler(cfg config, runDir string) (*managedProc, string) {
	status := "unavailable"
	console, err := os.Create(filepath.Join(runDir, "uprof-console.log"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: uProf console log: %v\n", err)
		return nil, status
	}
	argv := []string{cfg.UProf}
	if cfg.UProfSudo {
		argv = []string{"sudo", "-n", cfg.UProf}
	}
	argv = append(argv, "-m", "memory", "-a", "-A", "system,package", "-s", "-o", filepath.Join(runDir, "memory.csv"))
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = cfg.WorkDir
	cmd.Stdout, cmd.Stderr = console, console
	proc, startErr := startManaged(cmd)
	_ = console.Close()
	if startErr != nil {
		status = "failed to start: " + startErr.Error()
		fmt.Fprintf(os.Stderr, "warning: AMDuProfPcm start failed: %v\n", startErr)
		return nil, status
	}
	return proc, "running"
}

func collectRun(parent context.Context, cfg config, configName, configKey string, threads, rep int, fitOut string, fitSpill int, session *serverSession, execution runExecution, profilers runProfilers) runResult {
	metricsCancel, metricsDone := startMetricSampler(parent, session.baseURL+"/metrics", filepath.Join(execution.runDir, "metrics.csv"))
	start := time.Now()
	clientExit, clientErr := runClient(parent, cfg, threads, rep, execution.runDir)
	end := time.Now()
	metricsCancel()
	metrics := <-metricsDone
	metrics = captureRunLogs(session.logPath, execution, metrics, start, end)
	stopRunProfilers(cfg, profilers)
	if clientErr != nil {
		fmt.Fprintf(os.Stderr, "client run error: %v\n", clientErr)
	}
	return buildRunResult(cfg, configName, configKey, threads, rep, fitOut, fitSpill, execution, clientExit, clientErr, end.Sub(start), metrics, profilers)
}

func captureRunLogs(serverLog string, execution runExecution, metrics metricStats, start, end time.Time) metricStats {
	logEnd, err := logFileSize(serverLog)
	if err == nil {
		err = copyLogRange(serverLog, execution.llamaLog, execution.logStart, logEnd)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: capture server log for %s: %v\n", execution.runDir, err)
	}
	if elapsed := end.Sub(start).Seconds(); elapsed > 0 {
		metrics.WallGenerationTPS = metrics.PredictedTokensDelta / elapsed
		metrics.WallPromptTPS = metrics.PromptTokensDelta / elapsed
	}
	return metrics
}

func stopRunProfilers(cfg config, profilers runProfilers) {
	if profilers.uprofProc != nil {
		profilers.uprofProc.stop(syscall.SIGINT, 5*time.Second, cfg.UProfSudo)
	}
	if profilers.gpuProc != nil {
		profilers.gpuProc.stop(syscall.SIGTERM, 3*time.Second, false)
	}
}

func buildRunResult(cfg config, configName, configKey string, threads, rep int, fitOut string, fitSpill int, execution runExecution, clientExit int, clientErr error, wallTime time.Duration, metrics metricStats, profilers runProfilers) runResult {
	runDir, clientLog := execution.runDir, filepath.Join(execution.runDir, "client.log")
	res := runResult{
		ConfigName: configName, ConfigKey: configKey, Warmup: rep < 0,
		ThreadCount: threads, Repetition: rep, RunDir: runDir, ExitCode: clientExit,
		WallSeconds: wallTime.Seconds(), Client: parseClientLog(clientLog),
		Llama:  parseLlamaLog(execution.llamaLog, cfg.StarvationThreshold, cfg.LargePrefillTokens),
		Memory: parseMemoryLog(filepath.Join(runDir, "memory.csv")),
		GPU:    parseGPULog(filepath.Join(runDir, "gpu.csv")), Metrics: metrics,
		FitSpillBlocks: fitSpill, FitOutput: strings.TrimSpace(fitOut),
	}
	if res.Client.Parsed && res.Client.ReportedSeconds == 0 {
		res.Client.ReportedSeconds = res.WallSeconds
	}
	res.GPUProfiler = profilerResult(cfg.ProfileGPU, profilers.gpuStatus, res.GPU.Samples)
	res.MemoryProfiler = profilerResult(cfg.ProfileMemory, profilers.memoryStatus, res.Memory.Samples)
	if res.ExitCode == 0 && !res.Client.Parsed {
		res.ExitCode, res.Error = 1, "client benchmark result line was not parsed"
	} else if res.ExitCode == 0 && !res.Client.Passed {
		res.ExitCode, res.Error = 1, "client benchmark reported FAIL"
	}
	if clientErr != nil {
		res.Error = clientErr.Error()
	}
	if err := writeJSON(filepath.Join(runDir, "result.json"), res); err != nil {
		fmt.Fprintf(os.Stderr, "write result for %s: %v\n", runDir, err)
	}
	return res
}

func profilerResult(enabled bool, status string, samples int) string {
	if enabled && status == "running" {
		status = "unavailable: no parseable samples"
		if samples > 0 {
			return "captured"
		}
	}
	return status
}

func failedRun(configName, configKey string, threads, rep int, dir, fitOut string, fitSpill int, err error) runResult {
	res := runResult{
		ConfigName: configName, ConfigKey: configKey,
		Warmup:      rep < 0,
		ThreadCount: threads, Repetition: rep, RunDir: dir,
		ExitCode: -1, Error: err.Error(), FitOutput: strings.TrimSpace(fitOut), FitSpillBlocks: fitSpill,
	}
	if writeErr := writeJSON(filepath.Join(dir, "result.json"), res); writeErr != nil {
		fmt.Fprintf(os.Stderr, "write failed-run result for %s: %v\n", dir, writeErr)
	}
	return res
}

func runClient(parent context.Context, cfg config, threads, rep int, runDir string) (int, error) {
	logPath := filepath.Join(runDir, "client.log")
	f, err := os.Create(logPath)
	if err != nil {
		return -1, err
	}
	defer f.Close()

	ctx, cancel := context.WithTimeout(parent, cfg.ClientTimeout)
	defer cancel()

	cmd := exec.Command(cfg.Shell, "-lc", cfg.ClientCmd)
	cmd.Dir = cfg.WorkDir
	extra := clientEnvironment(cfg, threads, rep, runDir, filepath.Join(runDir, "opencode.json"))
	cmd.Env = envWith(os.Environ(), extra)
	cmd.Stdout = io.MultiWriter(os.Stdout, f)
	cmd.Stderr = io.MultiWriter(os.Stderr, f)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		return -1, err
	}
	return awaitClient(ctx, cmd, cfg.ClientTimeout)
}

func awaitClient(ctx context.Context, cmd *exec.Cmd, timeout time.Duration) (int, error) {
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	select {
	case err := <-done:
		if err == nil {
			return 0, nil
		}
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return ee.ExitCode(), err
		}
		return -1, err
	case <-ctx.Done():
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
			<-done
		}
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return -1, fmt.Errorf("client timeout after %s", timeout)
		}
		return -1, ctx.Err()
	}
}

func clientEnvironment(cfg config, threads, rep int, runDir, opencodeConfigPath string) []string {
	extra := append([]string(nil), cfg.ClientEnv...)
	return append(extra,
		fmt.Sprintf("LLAMA_BASE_URL=http://%s/v1", net.JoinHostPort(connectHost(cfg.Host), strconv.Itoa(cfg.Port))),
		fmt.Sprintf("LLAMA_HOST=%s", connectHost(cfg.Host)),
		fmt.Sprintf("LLAMA_PORT=%d", cfg.Port),
		fmt.Sprintf("BENCH_THREADS=%d", threads),
		fmt.Sprintf("BENCH_RUN=%d", absInt(rep)),
		fmt.Sprintf("BENCH_WARMUP=%t", rep < 0),
		fmt.Sprintf("BENCH_PRESERVE_THINKING=%t", cfg.PreserveThink),
		fmt.Sprintf("BENCH_PROMPT_SESSION_TYPE=%s", promptSessionType(cfg.PreserveThink)),
		fmt.Sprintf("BENCH_RUN_DIR=%s", runDir),
		fmt.Sprintf("OPENCODE_CONFIG=%s", opencodeConfigPath),
	)
}

func promptSessionType(preserveThinking bool) string {
	if preserveThinking {
		return "preserve-thinking"
	}
	return "default"
}

func writeRunOpenCodeConfig(templatePath, outputPath, baseURL string, contextLimit int) error {
	contents, err := os.ReadFile(templatePath)
	if err != nil {
		return err
	}
	var configData map[string]any
	if err := json.Unmarshal(contents, &configData); err != nil {
		return fmt.Errorf("parse %s: %w", templatePath, err)
	}
	providers, ok := configData["provider"].(map[string]any)
	if !ok {
		return fmt.Errorf("%s has no provider map", templatePath)
	}
	provider, ok := providers["amdbeast"].(map[string]any)
	if !ok {
		return fmt.Errorf("%s has no amdbeast provider", templatePath)
	}
	options, ok := provider["options"].(map[string]any)
	if !ok {
		return fmt.Errorf("%s has no amdbeast options map", templatePath)
	}
	options["baseURL"] = baseURL
	models, ok := provider["models"].(map[string]any)
	if !ok {
		return fmt.Errorf("%s has no amdbeast models map", templatePath)
	}
	model, ok := models["qwen36-coder"].(map[string]any)
	if !ok {
		return fmt.Errorf("%s has no qwen36-coder model", templatePath)
	}
	limit, ok := model["limit"].(map[string]any)
	if !ok {
		return fmt.Errorf("%s has no qwen36-coder limit map", templatePath)
	}
	limit["context"] = contextLimit
	encoded, err := json.MarshalIndent(configData, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(outputPath, append(encoded, '\n'), 0o600)
}

func absInt(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

func repetitionDir(rep int) string {
	if rep < 0 {
		return fmt.Sprintf("warmup-%02d", absInt(rep))
	}
	return fmt.Sprintf("run-%02d", rep)
}

func inspectFit(ctx context.Context, cfg config) (string, error) {
	args := fitArgs(cfg)
	cmd := exec.Command(cfg.FitBinary, args...)
	cmd.Dir = cfg.WorkDir
	cmd.Env = envWith(os.Environ(), cfg.ServerEnv)
	var output strings.Builder
	cmd.Stdout = &output
	cmd.Stderr = &output
	proc, err := startManaged(cmd)
	if err != nil {
		return output.String(), err
	}
	select {
	case <-ctx.Done():
		proc.stop(syscall.SIGTERM, 2*time.Second, false)
		return output.String(), ctx.Err()
	case <-proc.done:
		_, err := proc.exited()
		return output.String(), err
	}
}

func countFitSpillBlocks(s string) int {
	seen := map[string]bool{}
	for _, line := range strings.Split(s, "\n") {
		if !strings.Contains(strings.ToUpper(line), "CPU") {
			continue
		}
		for _, match := range reFitBlock.FindAllStringSubmatch(line, -1) {
			if len(match) > 1 {
				seen[match[1]] = true
			}
		}
	}
	return len(seen)
}

func waitHealthy(ctx context.Context, url string, timeout time.Duration, proc *managedProc) error {
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	client := &http.Client{Timeout: 1 * time.Second}

	for {
		if exited, err := proc.exited(); exited {
			return fmt.Errorf("llama-server exited before healthy: %v", err)
		}
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if resp, err := client.Do(req); err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-deadline.C:
			return fmt.Errorf("server health timeout after %s", timeout)
		case <-ticker.C:
		}
	}
}

func connectHost(bind string) string {
	switch bind {
	case "0.0.0.0", "::", "[::]":
		return "127.0.0.1"
	default:
		return bind
	}
}

func startMetricSampler(parent context.Context, url, path string) (context.CancelFunc, <-chan metricStats) {
	ctx, cancel := context.WithCancel(parent)
	done := make(chan metricStats, 1)
	go func() {
		defer close(done)
		f, err := os.Create(path)
		if err != nil {
			done <- metricStats{}
			return
		}
		defer f.Close()
		w := csv.NewWriter(f)
		_ = w.Write([]string{
			"timestamp", "prompt_tokens_total", "tokens_predicted_total",
			"prompt_seconds_total", "tokens_predicted_seconds_total",
			"requests_processing", "requests_deferred", "n_tokens_max",
		})
		defer w.Flush()

		var samples []metricSample
		client := &http.Client{Timeout: 800 * time.Millisecond}
		sample := func() {
			m, err := fetchPromMetrics(ctx, client, url)
			if err != nil {
				return
			}
			s := metricSample{
				Time:               time.Now(),
				PromptTokensTotal:  m["llamacpp:prompt_tokens_total"],
				PredictedTokens:    m["llamacpp:tokens_predicted_total"],
				PromptSecondsTotal: m["llamacpp:prompt_seconds_total"],
				PredSecondsTotal:   m["llamacpp:tokens_predicted_seconds_total"],
				RequestsProcessing: m["llamacpp:requests_processing"],
				RequestsDeferred:   m["llamacpp:requests_deferred"],
				NTokensMax:         m["llamacpp:n_tokens_max"],
			}
			samples = append(samples, s)
			_ = w.Write([]string{
				s.Time.Format(time.RFC3339Nano),
				f64(s.PromptTokensTotal), f64(s.PredictedTokens),
				f64(s.PromptSecondsTotal), f64(s.PredSecondsTotal),
				f64(s.RequestsProcessing), f64(s.RequestsDeferred), f64(s.NTokensMax),
			})
			w.Flush()
		}

		sample()
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				sample()
				done <- analyzeMetricSamples(samples)
				return
			case <-ticker.C:
				sample()
			}
		}
	}()
	return cancel, done
}

func fetchPromMetrics(ctx context.Context, client *http.Client, url string) (map[string]float64, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metrics HTTP %d", resp.StatusCode)
	}
	return parsePromMetrics(resp.Body)
}

func parsePromMetrics(reader io.Reader) (map[string]float64, error) {
	out := map[string]float64{}
	sc := bufio.NewScanner(reader)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		name := strings.SplitN(fields[0], "{", 2)[0]
		v, err := strconv.ParseFloat(fields[1], 64)
		if err == nil {
			if name == "llamacpp:n_tokens_max" {
				out[name] = math.Max(out[name], v)
			} else {
				out[name] += v
			}
		}
	}
	return out, sc.Err()
}

func analyzeMetricSamples(s []metricSample) metricStats {
	var out metricStats
	out.Samples = len(s)
	if len(s) == 0 {
		return out
	}
	for _, x := range s {
		out.MaxRequestsProcessing = math.Max(out.MaxRequestsProcessing, x.RequestsProcessing)
		out.MaxRequestsDeferred = math.Max(out.MaxRequestsDeferred, x.RequestsDeferred)
		out.MaxContextObserved = math.Max(out.MaxContextObserved, x.NTokensMax)
	}
	if len(s) >= 2 {
		first, last := s[0], s[len(s)-1]
		sec := last.Time.Sub(first.Time).Seconds()
		if sec > 0 {
			out.PredictedTokensDelta = last.PredictedTokens - first.PredictedTokens
			out.PromptTokensDelta = last.PromptTokensTotal - first.PromptTokensTotal
			out.WallGenerationTPS = out.PredictedTokensDelta / sec
			out.WallPromptTPS = out.PromptTokensDelta / sec
		}
	}
	return out
}

type llamaTiming struct {
	tokens int64
	sec    float64
}

type llamaLogAccumulator struct {
	prompts, evals       map[string]llamaTiming
	tasks, active, low   map[string]bool
	slots                map[string]bool
	tg3, lcpSim, lcpKeep []float64
	cancelled            int
	maxPrefill           int64
	maxConcurrent        int
	largePrefillEvents   int
	starvationEvents     int
	starvationThreshold  float64
	largePrefillTokens   int64
}

func parseLlamaLog(path string, starvationThreshold float64, largePrefillTokens int64) llamaStats {
	f, err := os.Open(path)
	if err != nil {
		return llamaStats{}
	}
	defer f.Close()
	acc := newLlamaLogAccumulator(starvationThreshold, largePrefillTokens)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		acc.consume(scanner.Text())
	}
	return acc.stats()
}

func newLlamaLogAccumulator(starvationThreshold float64, largePrefillTokens int64) *llamaLogAccumulator {
	return &llamaLogAccumulator{
		prompts: make(map[string]llamaTiming), evals: make(map[string]llamaTiming),
		tasks: make(map[string]bool), active: make(map[string]bool), low: make(map[string]bool), slots: make(map[string]bool),
		starvationThreshold: starvationThreshold, largePrefillTokens: largePrefillTokens,
	}
}

func (acc *llamaLogAccumulator) consume(line string) {
	acc.consumeLaunch(line)
	acc.consumePrompt(line)
	acc.consumeEval(line)
	acc.consumeProgress(line)
	acc.consumeCancellation(line)
	acc.consumePrefixReuse(line)
	acc.consumePrefill(line)
}

func (acc *llamaLogAccumulator) consumeLaunch(line string) {
	match := reLaunch.FindStringSubmatch(line)
	if match == nil {
		return
	}
	acc.slots[match[1]] = true
	acc.tasks[match[2]] = true
	acc.active[match[2]] = true
	acc.maxConcurrent = max(acc.maxConcurrent, len(acc.active))
}

func (acc *llamaLogAccumulator) consumePrompt(line string) {
	match := rePromptEval.FindStringSubmatch(line)
	if match == nil {
		return
	}
	ms, _ := strconv.ParseFloat(match[2], 64)
	tokens, _ := strconv.ParseInt(match[3], 10, 64)
	acc.prompts[match[1]] = llamaTiming{tokens: tokens, sec: ms / 1000}
}

func (acc *llamaLogAccumulator) consumeEval(line string) {
	match := reEval.FindStringSubmatch(line)
	if match == nil {
		return
	}
	ms, _ := strconv.ParseFloat(match[2], 64)
	tokens, _ := strconv.ParseInt(match[3], 10, 64)
	acc.evals[match[1]] = llamaTiming{tokens: tokens, sec: ms / 1000}
	delete(acc.active, match[1])
	delete(acc.low, match[1])
}

func (acc *llamaLogAccumulator) consumeProgress(line string) {
	match := reProgress.FindStringSubmatch(line)
	if match == nil {
		return
	}
	rate, _ := strconv.ParseFloat(match[4], 64)
	if rate < 0 {
		return
	}
	acc.tg3 = append(acc.tg3, rate)
	if rate < acc.starvationThreshold && len(acc.active) > 1 && !acc.low[match[1]] {
		acc.starvationEvents++
		acc.low[match[1]] = true
	} else if rate >= acc.starvationThreshold {
		delete(acc.low, match[1])
	}
}

func (acc *llamaLogAccumulator) consumeCancellation(line string) {
	match := reCancel.FindStringSubmatch(line)
	if match == nil || reStopCancel.MatchString(line) {
		return
	}
	acc.cancelled++
	delete(acc.active, match[1])
	delete(acc.low, match[1])
}

func (acc *llamaLogAccumulator) consumePrefixReuse(line string) {
	match := reLCP.FindStringSubmatch(line)
	if match == nil {
		return
	}
	similarity, _ := strconv.ParseFloat(match[1], 64)
	keep, _ := strconv.ParseFloat(match[2], 64)
	acc.lcpSim = append(acc.lcpSim, similarity)
	acc.lcpKeep = append(acc.lcpKeep, keep)
}

func (acc *llamaLogAccumulator) consumePrefill(line string) {
	match := rePrefill.FindStringSubmatch(line)
	if match == nil {
		return
	}
	tokens, _ := strconv.ParseInt(match[2], 10, 64)
	if tokens >= acc.largePrefillTokens {
		acc.largePrefillEvents++
	}
	if tokens > acc.maxPrefill {
		acc.maxPrefill = tokens
	}
}

func (acc *llamaLogAccumulator) stats() llamaStats {
	out := llamaStats{
		TaskCount: len(acc.tasks), SlotsUsed: len(acc.slots), CancelledTasks: acc.cancelled,
		MaxPromptProcessingTokens: acc.maxPrefill, MaxConcurrentTasks: acc.maxConcurrent,
		LargePrefillEvents: acc.largePrefillEvents, DecodeStarvationEvents: acc.starvationEvents,
	}
	for _, timing := range acc.prompts {
		out.PromptTokens += timing.tokens
		out.PromptSeconds += timing.sec
	}
	if out.PromptSeconds > 0 {
		out.PromptTPSWeighted = float64(out.PromptTokens) / out.PromptSeconds
	}
	for _, timing := range acc.evals {
		out.GeneratedTokens += timing.tokens
		out.EvalSecondsSummed += timing.sec
	}
	if out.EvalSecondsSummed > 0 {
		out.TaskTGWeighted = float64(out.GeneratedTokens) / out.EvalSecondsSummed
	}
	setLlamaDecodeStats(&out, acc.tg3)
	setLlamaPrefixStats(&out, acc.lcpSim, acc.lcpKeep)
	return out
}

func setLlamaDecodeStats(out *llamaStats, values []float64) {
	if len(values) == 0 {
		return
	}
	stats := statsOf(values)
	out.MinTG3s, out.MaxTG3s, out.MeanTG3s = stats.Min, stats.Max, stats.Mean
}

func setLlamaPrefixStats(out *llamaStats, similarities, keep []float64) {
	if len(similarities) == 0 {
		return
	}
	out.LCPCount = len(similarities)
	out.MeanLCPSimilarity = statsOf(similarities).Mean
	out.MeanLCPKeep = statsOf(keep).Mean
}

func parseMemoryLog(path string) memoryStats {
	f, err := os.Open(path)
	if err != nil {
		return memoryStats{}
	}
	defer f.Close()
	parser := memoryLogParser{totalIdx: -1, readIdx: -1, writeIdx: -1}
	scanLogLines(f, parser.consumeLine)
	return parser.stats()
}

type memoryLogParser struct {
	totalIdx, readIdx, writeIdx int
	headerFound                 bool
	total, read, write          []float64
}

func (p *memoryLogParser) consumeLine(line string) {
	record, err := csv.NewReader(strings.NewReader(strings.TrimSpace(line))).Read()
	if err != nil || len(record) == 0 {
		return
	}
	if !p.headerFound {
		p.readHeader(record)
		return
	}
	p.readSample(record)
}

func (p *memoryLogParser) readHeader(record []string) {
	for i, field := range record {
		switch normalizeHeader(field) {
		case "TOTAL_MEM_BW_GB_S":
			if p.totalIdx < 0 {
				p.totalIdx = i
			}
		case "TOTAL_MEM_RDBW_GB_S":
			if p.readIdx < 0 {
				p.readIdx = i
			}
		case "TOTAL_MEM_WRBW_GB_S":
			if p.writeIdx < 0 {
				p.writeIdx = i
			}
		}
	}
	p.headerFound = p.totalIdx >= 0 && p.readIdx >= 0 && p.writeIdx >= 0
}

func (p *memoryLogParser) readSample(record []string) {
	maxIdx := max(p.totalIdx, p.readIdx, p.writeIdx)
	if len(record) <= maxIdx {
		return
	}
	total, e1 := strconv.ParseFloat(strings.TrimSpace(record[p.totalIdx]), 64)
	read, e2 := strconv.ParseFloat(strings.TrimSpace(record[p.readIdx]), 64)
	write, e3 := strconv.ParseFloat(strings.TrimSpace(record[p.writeIdx]), 64)
	if e1 != nil || e2 != nil || e3 != nil {
		return
	}
	p.total, p.read, p.write = append(p.total, total), append(p.read, read), append(p.write, write)
}

func (p *memoryLogParser) stats() memoryStats {
	ts, rs, ws := statsOf(p.total), statsOf(p.read), statsOf(p.write)
	return memoryStats{
		ReadValues: append([]float64(nil), p.read...),
		Samples:    len(p.total), MeanTotal: ts.Mean, MaxTotal: ts.Max,
		MeanRead: rs.Mean, MaxRead: rs.Max, MeanWrite: ws.Mean, MaxWrite: ws.Max,
		MedianTotal: ts.Median, MedianRead: rs.Median, MedianWrite: ws.Median,
		P95Total: ts.P95, P95Read: rs.P95, P95Write: ws.P95,
	}
}

func parseGPULog(path string) gpuStats {
	f, err := os.Open(path)
	if err != nil {
		return gpuStats{}
	}
	defer f.Close()
	parser := gpuLogParser{index: map[string]int{}}
	scanLogLines(f, parser.consumeLine)
	return parser.stats()
}

func scanLogLines(r io.Reader, consume func(string)) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			consume(line)
		}
	}
}

type gpuLogParser struct {
	header                     []string
	index                      map[string]int
	util, memUtil, power, vram []float64
}

func (p *gpuLogParser) consumeLine(line string) {
	if strings.HasPrefix(line, "WARNING") {
		return
	}
	record, err := csv.NewReader(strings.NewReader(line)).Read()
	if err != nil || len(record) < 2 {
		return
	}
	if p.header == nil {
		p.readHeader(record)
		return
	}
	p.readSample(record)
}

func (p *gpuLogParser) readHeader(record []string) {
	for _, field := range record {
		if normalizeHeader(field) == "GPU" {
			p.header = record
			break
		}
	}
	for i, field := range p.header {
		p.index[normalizeHeader(field)] = i
	}
}

func (p *gpuLogParser) appendField(record []string, index int, values *[]float64) {
	if index < 0 || index >= len(record) {
		return
	}
	if value, ok := parseNumber(record[index]); ok {
		*values = append(*values, value)
	}
}

func (p *gpuLogParser) readSample(record []string) {
	p.appendField(record, findHeaderIndex(p.index, "GFX_UTIL", "GFX%", "GFXUTIL", "GFX", "GFX_ACTIVITY", "GPU_UTIL", "GPU_UTILIZATION", "GPU%"), &p.util)
	p.appendField(record, findHeaderIndex(p.index, "MEM_UTIL", "MEM%", "MEMUTIL", "MEM_ACTIVITY", "MEM"), &p.memUtil)
	p.appendField(record, findHeaderContains(p.index, "POWER"), &p.power)
	index := findHeaderIndex(p.index, "VRAM_USED", "VRAM_USED_MB", "VRAM_USAGE", "VRAM_USAGE_MB")
	if index >= 0 && index < len(record) {
		field := strings.SplitN(record[index], "/", 2)[0]
		if value, ok := parseNumber(field); ok {
			p.vram = append(p.vram, value)
		}
	}
}

func (p *gpuLogParser) stats() gpuStats {
	us, mus, ps, vs := statsOf(p.util), statsOf(p.memUtil), statsOf(p.power), statsOf(p.vram)
	return gpuStats{
		UtilValues: append([]float64(nil), p.util...), PowerValues: append([]float64(nil), p.power...),
		Samples: max(len(p.util), len(p.memUtil), len(p.power), len(p.vram)), MeanUtil: us.Mean, MaxUtil: us.Max,
		MeanMemUtil: mus.Mean, MaxMemUtil: mus.Max,
		MedianMemUtil: mus.Median, P95MemUtil: mus.P95,
		MeanPower: ps.Mean, MaxPower: ps.Max, MedianPower: ps.Median, P95Power: ps.P95,
		MeanVRAMUsed: vs.Mean, MaxVRAMUsed: vs.Max, MedianVRAM: vs.Median, P95VRAM: vs.P95,
		MedianUtil: us.Median, P95Util: us.P95,
	}
}

func normalizeHeader(s string) string {
	s = strings.ToUpper(strings.TrimSpace(s))
	r := strings.NewReplacer(" ", "_", "-", "_", "(", "", ")", "", "/", "_")
	return r.Replace(s)
}

func findHeaderIndex(m map[string]int, names ...string) int {
	for _, n := range names {
		if i, ok := m[normalizeHeader(n)]; ok {
			return i
		}
	}
	return -1
}

func findHeaderContains(m map[string]int, needle string) int {
	needle = normalizeHeader(needle)
	for k, i := range m {
		if strings.Contains(k, needle) {
			return i
		}
	}
	return -1
}

func parseNumber(s string) (float64, bool) {
	var b strings.Builder
	for _, r := range s {
		if (r >= '0' && r <= '9') || r == '.' || r == '-' || r == '+' || r == 'e' || r == 'E' {
			b.WriteRune(r)
		}
	}
	if b.Len() == 0 {
		return 0, false
	}
	v, err := strconv.ParseFloat(b.String(), 64)
	return v, err == nil
}

func parseClientLog(path string) clientStats {
	b, err := os.ReadFile(path)
	if err != nil {
		return clientStats{}
	}
	all := reClient.FindAllStringSubmatch(string(b), -1)
	if len(all) == 0 {
		return parseOpenCodeEvents(string(b))
	}
	m := all[len(all)-1]
	d, err := time.ParseDuration(strings.TrimSpace(m[6]))
	if err != nil {
		return clientStats{}
	}
	turns, _ := strconv.Atoi(m[7])
	out, _ := strconv.Atoi(m[8])
	return clientStats{
		Parsed: true, Passed: m[5] == "PASS", Target: m[1], Task: m[2],
		Variant: m[3], Arm: m[4], ReportedSeconds: d.Seconds(), Turns: turns, OutTokens: out,
	}
}

type openCodeEvent struct {
	Type string `json:"type"`
	Part struct {
		Text   string `json:"text"`
		Reason string `json:"reason"`
		Tokens struct {
			Input     int `json:"input"`
			Output    int `json:"output"`
			Reasoning int `json:"reasoning"`
			Cache     struct {
				Read  int `json:"read"`
				Write int `json:"write"`
			} `json:"cache"`
		} `json:"tokens"`
	} `json:"part"`
}

func parseOpenCodeEvents(log string) clientStats {
	var result clientStats
	result.Target = "opencode"
	result.Task = "task-11"
	result.Variant = "standalone-prompt"
	result.Arm = "baseline"
	var text strings.Builder
	hasError := false
	stoppedNormally := false
	scanner := bufio.NewScanner(strings.NewReader(log))
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		var event openCodeEvent
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			continue
		}
		switch event.Type {
		case "text":
			result.Parsed = true
			text.WriteString(event.Part.Text)
		case "step_finish":
			result.Parsed = true
			result.Turns++
			if event.Part.Reason == "stop" {
				stoppedNormally = true
			}
			result.OutTokens += event.Part.Tokens.Output
			result.PromptTokens += event.Part.Tokens.Input + event.Part.Tokens.Cache.Read + event.Part.Tokens.Cache.Write
			result.ReasoningTokens += event.Part.Tokens.Reasoning
		case "step_start", "tool_use":
			result.Parsed = true
		case "error":
			result.Parsed = true
			hasError = true
		}
	}
	result.Passed = result.Parsed && stoppedNormally && !hasError && strings.TrimSpace(text.String()) != ""
	return result
}

func writeAllSummaries(root string, results []runResult) error {
	if err := writeRunsCSV(filepath.Join(root, "runs.csv"), results); err != nil {
		return err
	}
	if err := writeSummaryCSV(filepath.Join(root, "summary.csv"), results); err != nil {
		return err
	}
	return writeSummaryMarkdown(filepath.Join(root, "summary.md"), results)
}

func writeRunsCSV(path string, results []runResult) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{
		"configuration", "threads", "run", "exit_code", "wall_s", "client_reported_s",
		"client_prompt_tokens", "client_reasoning_tokens", "turns", "out_tokens",
		"tasks", "slots_used", "cancelled_tasks", "max_concurrent_tasks",
		"large_prefill_events", "decode_starvation_events",
		"prompt_tokens", "prompt_s", "prompt_tps",
		"generated_tokens", "summed_eval_s", "task_tg_weighted", "min_tg3s",
		"metrics_wall_gen_tps", "max_requests_processing", "max_requests_deferred", "max_context",
		"mem_mean_read_gbs", "mem_max_read_gbs", "mem_mean_write_gbs",
		"gpu_mean_util_pct", "gpu_max_util_pct", "gpu_mean_power_w", "gpu_max_power_w",
		"fit_spill_blocks", "run_dir", "error",
	}
	if err := w.Write(header); err != nil {
		return err
	}
	for _, r := range results {
		row := []string{
			r.ConfigName, strconv.Itoa(r.ThreadCount), strconv.Itoa(r.Repetition), strconv.Itoa(r.ExitCode),
			ff(r.WallSeconds), ff(r.Client.ReportedSeconds), strconv.Itoa(r.Client.PromptTokens),
			strconv.Itoa(r.Client.ReasoningTokens), strconv.Itoa(r.Client.Turns), strconv.Itoa(r.Client.OutTokens),
			strconv.Itoa(r.Llama.TaskCount), strconv.Itoa(r.Llama.SlotsUsed), strconv.Itoa(r.Llama.CancelledTasks),
			strconv.Itoa(r.Llama.MaxConcurrentTasks), strconv.Itoa(r.Llama.LargePrefillEvents), strconv.Itoa(r.Llama.DecodeStarvationEvents),
			strconv.FormatInt(r.Llama.PromptTokens, 10), ff(r.Llama.PromptSeconds), ff(r.Llama.PromptTPSWeighted),
			strconv.FormatInt(r.Llama.GeneratedTokens, 10), ff(r.Llama.EvalSecondsSummed), ff(r.Llama.TaskTGWeighted), ff(r.Llama.MinTG3s),
			ff(r.Metrics.WallGenerationTPS), ff(r.Metrics.MaxRequestsProcessing), ff(r.Metrics.MaxRequestsDeferred), ff(r.Metrics.MaxContextObserved),
			ff(r.Memory.MeanRead), ff(r.Memory.MaxRead), ff(r.Memory.MeanWrite),
			ff(r.GPU.MeanUtil), ff(r.GPU.MaxUtil), ff(r.GPU.MeanPower), ff(r.GPU.MaxPower),
			strconv.Itoa(r.FitSpillBlocks), r.RunDir, r.Error,
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

type aggregate struct {
	ConfigName       string
	ConfigKey        string
	Threads          int
	Runs             int
	Success          int
	MeanWall         float64
	MedianWall       float64
	P95Wall          float64
	MinWall          float64
	MaxWall          float64
	MedianClient     float64
	MedianTaskTG     float64
	MedianMetricTG   float64
	MedianPromptTPS  float64
	MedianMemRead    float64
	P95MemRead       float64
	MaxMemRead       float64
	MedianGPUUtil    float64
	P95GPUUtil       float64
	MaxGPUUtil       float64
	MedianPower      float64
	P95Power         float64
	MaxPower         float64
	MedianCancelled  float64
	MedianStarvation float64
	TotalCancelled   int
	SpillBlocks      int
}

func aggregateResults(results []runResult) []aggregate {
	groups := groupResultsByConfigThread(results)
	keys := sortedKeys(groups)
	out := make([]aggregate, 0, len(keys))
	for _, key := range keys {
		out = append(out, aggregateRunGroup(groups[key]))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Success == 0 || out[j].Success == 0 {
			return out[i].Success > out[j].Success
		}
		if out[i].MedianWall != out[j].MedianWall {
			return out[i].MedianWall < out[j].MedianWall
		}
		if out[i].ConfigName != out[j].ConfigName {
			return out[i].ConfigName < out[j].ConfigName
		}
		return out[i].Threads < out[j].Threads
	})
	return out
}

func groupResultsByConfigThread(results []runResult) map[string][]runResult {
	groups := make(map[string][]runResult)
	for _, result := range results {
		key := fmt.Sprintf("%s|%d", result.ConfigKey, result.ThreadCount)
		groups[key] = append(groups[key], result)
	}
	return groups
}

func sortedKeys(groups map[string][]runResult) []string {
	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

type aggregateSamples struct {
	wall, client, taskTG, metricTG []float64
	prompt, memory, gpu, power     []float64
	memoryRaw, gpuRaw, powerRaw    []float64
	cancelled, starvation          []float64
	success, spill, totalCancelled int
}

func aggregateRunGroup(runs []runResult) aggregate {
	samples := aggregateSamples{}
	for _, run := range runs {
		samples.add(run)
	}
	return samples.result(runs)
}

func (samples *aggregateSamples) add(run runResult) {
	samples.totalCancelled += run.Llama.CancelledTasks
	samples.spill = max(samples.spill, run.FitSpillBlocks)
	if run.ExitCode != 0 {
		return
	}
	samples.success++
	samples.wall = append(samples.wall, run.WallSeconds)
	appendPositive(&samples.client, run.Client.ReportedSeconds)
	appendPositive(&samples.taskTG, run.Llama.TaskTGWeighted)
	appendPositive(&samples.metricTG, run.Metrics.WallGenerationTPS)
	appendPositive(&samples.prompt, run.Llama.PromptTPSWeighted)
	if run.Memory.MeanRead > 0 {
		samples.memory = append(samples.memory, run.Memory.MeanRead)
		samples.memoryRaw = append(samples.memoryRaw, run.Memory.ReadValues...)
	}
	samples.addGPU(run)
	samples.cancelled = append(samples.cancelled, float64(run.Llama.CancelledTasks))
	samples.starvation = append(samples.starvation, float64(run.Llama.DecodeStarvationEvents))
}

func appendPositive(values *[]float64, value float64) {
	if value > 0 {
		*values = append(*values, value)
	}
}

func (samples *aggregateSamples) addGPU(run runResult) {
	gpu := run.GPU
	// Raw samples are omitted from result.json and reloaded on resume.
	if gpu.Samples > 0 && len(gpu.UtilValues) == 0 && len(gpu.PowerValues) == 0 {
		gpu = parseGPULog(filepath.Join(run.RunDir, "gpu.csv"))
	}
	if gpu.Samples == 0 {
		return
	}
	appendPositive(&samples.gpu, gpu.MeanUtil)
	samples.gpuRaw = append(samples.gpuRaw, gpu.UtilValues...)
	if gpu.MeanPower > 0 {
		samples.power = append(samples.power, gpu.MeanPower)
		samples.powerRaw = append(samples.powerRaw, gpu.PowerValues...)
	}
}

func (samples aggregateSamples) result(runs []runResult) aggregate {
	wallStats := statsOf(samples.wall)
	memoryStats, gpuStats, powerStats := statsOf(samples.memoryRaw), statsOf(samples.gpuRaw), statsOf(samples.powerRaw)
	first := runs[0]
	return aggregate{
		ConfigName: first.ConfigName, ConfigKey: first.ConfigKey, Threads: first.ThreadCount,
		Runs: len(runs), Success: samples.success,
		MeanWall: wallStats.Mean, MedianWall: wallStats.Median, P95Wall: wallStats.P95,
		MinWall: wallStats.Min, MaxWall: wallStats.Max,
		MedianClient: statsOf(samples.client).Median, MedianTaskTG: statsOf(samples.taskTG).Median,
		MedianMetricTG: statsOf(samples.metricTG).Median, MedianPromptTPS: statsOf(samples.prompt).Median,
		MedianMemRead: statsOf(samples.memory).Median, P95MemRead: memoryStats.P95, MaxMemRead: memoryStats.Max,
		MedianGPUUtil: statsOf(samples.gpu).Median, P95GPUUtil: gpuStats.P95, MaxGPUUtil: gpuStats.Max,
		MedianPower: statsOf(samples.power).Median, P95Power: powerStats.P95, MaxPower: powerStats.Max,
		MedianCancelled: statsOf(samples.cancelled).Median, MedianStarvation: statsOf(samples.starvation).Median,
		TotalCancelled: samples.totalCancelled, SpillBlocks: samples.spill,
	}
}

func writeSummaryCSV(path string, results []runResult) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	if err := w.Write([]string{
		"configuration", "config_key", "threads", "runs", "success", "mean_wall_s", "median_wall_s", "p95_wall_s", "min_wall_s", "max_wall_s",
		"median_client_s", "median_task_tg", "median_wall_gen_tps", "median_prompt_tps",
		"median_mem_read_gbs", "p95_mem_read_gbs", "max_mem_read_gbs",
		"median_gpu_util_pct", "p95_gpu_util_pct", "max_gpu_util_pct",
		"median_power_w", "p95_power_w", "max_power_w",
		"median_cancelled_tasks", "total_cancelled_tasks", "median_decode_starvation_events", "fit_spill_blocks",
	}); err != nil {
		return err
	}
	for _, a := range aggregateResults(results) {
		if err := w.Write([]string{
			a.ConfigName, a.ConfigKey, strconv.Itoa(a.Threads), strconv.Itoa(a.Runs), strconv.Itoa(a.Success),
			ff(a.MeanWall), ff(a.MedianWall), ff(a.P95Wall), ff(a.MinWall), ff(a.MaxWall), ff(a.MedianClient),
			ff(a.MedianTaskTG), ff(a.MedianMetricTG), ff(a.MedianPromptTPS),
			ff(a.MedianMemRead), ff(a.P95MemRead), ff(a.MaxMemRead),
			ff(a.MedianGPUUtil), ff(a.P95GPUUtil), ff(a.MaxGPUUtil),
			ff(a.MedianPower), ff(a.P95Power), ff(a.MaxPower),
			ff(a.MedianCancelled), strconv.Itoa(a.TotalCancelled), ff(a.MedianStarvation), strconv.Itoa(a.SpillBlocks),
		}); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func writeSummaryMarkdown(path string, results []runResult) error {
	var b strings.Builder
	b.WriteString("# llama.cpp benchmark summary\n\n")
	b.WriteString("| configuration | threads | ok/runs | median wall | min–max wall | median task tg | wall gen tok/s | prompt tok/s | DDR read GB/s | GPU util | power | cancelled | starvation events | spill blocks |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, a := range aggregateResults(results) {
		fmt.Fprintf(&b,
			"| %s | %d | %d/%d | %.2fs | %.2f–%.2fs | %.2f | %.2f | %.1f | %.2f | %.1f%% | %.1fW | %.1f | %.1f | %d |\n",
			a.ConfigName, a.Threads, a.Success, a.Runs, a.MedianWall, a.MinWall, a.MaxWall,
			a.MedianTaskTG, a.MedianMetricTG, a.MedianPromptTPS,
			a.MedianMemRead, a.MedianGPUUtil, a.MedianPower,
			a.MedianCancelled, a.MedianStarvation, a.SpillBlocks,
		)
	}
	b.WriteString("\n`task tg` is weighted per-task decode throughput from llama.cpp final timing lines. `wall gen tok/s` is the server-wide generated-token counter delta divided by benchmark wall time, so it includes tool/idle gaps.\n")
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func printRunSummary(r runResult) {
	fmt.Printf("threads=%d run=%d wall=%.2fs", r.ThreadCount, r.Repetition, r.WallSeconds)
	if r.Client.Parsed {
		fmt.Printf(" client=%.2fs", r.Client.ReportedSeconds)
	}
	fmt.Printf(" task-tg=%.2f wall-gen=%.2f pp=%.1f", r.Llama.TaskTGWeighted, r.Metrics.WallGenerationTPS, r.Llama.PromptTPSWeighted)
	if r.Memory.Samples > 0 {
		fmt.Printf(" ddr-read=%.2fGB/s", r.Memory.MeanRead)
	}
	if r.GPU.Samples > 0 {
		fmt.Printf(" gpu=%.1f%% power=%.1fW", r.GPU.MeanUtil, r.GPU.MeanPower)
	}
	fmt.Printf(" cancels=%d spill-blocks=%d\n", r.Llama.CancelledTasks, r.FitSpillBlocks)
}

func printAggregateSummary(results []runResult) {
	fmt.Println("=== aggregate ===")
	fmt.Printf("%-22s %-8s %-8s %-12s %-12s %-12s %-12s %-10s %-10s\n",
		"configuration", "threads", "ok/runs", "median wall", "task tg", "wall gen", "DDR read", "GPU%", "power")
	for _, a := range aggregateResults(results) {
		fmt.Printf("%-22s %-8d %d/%-6d %-12.2f %-12.2f %-12.2f %-12.2f %-10.1f %-10.1f\n",
			a.ConfigName, a.Threads, a.Success, a.Runs, a.MedianWall, a.MedianTaskTG,
			a.MedianMetricTG, a.MedianMemRead, a.MedianGPUUtil, a.MedianPower)
	}
}

func writeEnvironmentSnapshot(cfg config) {
	var b strings.Builder
	fmt.Fprintf(&b, "generated_at=%s\n", time.Now().Format(time.RFC3339Nano))
	fmt.Fprintf(&b, "workdir=%s\n", cfg.WorkDir)
	fmt.Fprintf(&b, "client_cmd=%s\n", cfg.ClientCmd)
	fmt.Fprintf(&b, "go=%s\n\n", runtimeVersion())

	commands := [][]string{
		{"uname", "-a"},
		{"lscpu"},
		{cfg.ServerBinary, "--version"},
		{"git", "-C", cfg.WorkDir, "rev-parse", "HEAD"},
	}
	if cfg.AmdSMI != "" {
		commands = append(commands, []string{cfg.AmdSMI, "version"})
	}
	if cfg.UProf != "" {
		commands = append(commands, []string{cfg.UProf, "--version"})
	}
	commands = append(commands, []string{"hipconfig", "--version"})
	for _, argv := range commands {
		fmt.Fprintf(&b, "$ %s\n", strings.Join(argv, " "))
		cmd := exec.Command(argv[0], argv[1:]...)
		out, err := cmd.CombinedOutput()
		b.Write(out)
		if err != nil {
			fmt.Fprintf(&b, "[error: %v]\n", err)
		}
		b.WriteString("\n")
	}
	_ = os.WriteFile(filepath.Join(cfg.OutputDir, "environment.txt"), []byte(b.String()), 0o644)
}

func runtimeVersion() string {
	out, err := exec.Command("go", "version").CombinedOutput()
	if err == nil {
		return strings.TrimSpace(string(out))
	}
	return "unknown"
}

func gitCommit(workdir string) string {
	cmd := exec.Command("git", "-C", workdir, "rev-parse", "HEAD")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func envWith(base []string, extras []string) []string {
	env := append([]string(nil), base...)
	env = append(env, extras...)
	return env
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o644); err != nil {
		return err
	}
	return nil
}

func f64(v float64) string { return strconv.FormatFloat(v, 'f', 6, 64) }
func ff(v float64) string  { return strconv.FormatFloat(v, 'f', 3, 64) }

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}
