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
	if len(os.Args) > 1 && os.Args[1] == "analyze" {
		if err := runAnalysisCommand(os.Args[2:], os.Stdout); err != nil {
			fatalf("analyze: %v", err)
		}
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "plan" {
		if err := runPlanCommand(os.Args[2:], os.Stdout); err != nil {
			fatalf("plan: %v", err)
		}
		return
	}

	cfg := parseFlags()
	invocationDir, err := os.Getwd()
	if err != nil {
		fatalf("current directory: %v", err)
	}

	threadValues, err := parseThreadValues(cfg.ThreadValues)
	if err != nil {
		fatalf("invalid --threads: %v", err)
	}
	if cfg.Runs < 1 {
		fatalf("--runs must be >= 1")
	}
	if cfg.WarmupRuns < 0 {
		fatalf("--warmup-runs must be >= 0")
	}

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

	cfg.ServerBinary, err = resolveExecutable(cfg.ServerBinary, cfg.WorkDir)
	if err != nil {
		fatalf("server binary: %v", err)
	}
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
	cfg.Shell, err = resolveExecutable(cfg.Shell, cfg.WorkDir)
	if err != nil {
		fatalf("shell: %v", err)
	}

	if cfg.ProfileMemory && cfg.UProfSudo {
		fmt.Println("Acquiring sudo credentials for AMD uProf...")
		cmd := exec.Command("sudo", "-v")
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "warning: sudo credentials unavailable: %v; disabling uProf\n", err)
			cfg.ProfileMemory = false
		} else if cfg.ModprobeUncore {
			if err := exec.Command("sudo", "-n", "modprobe", "amd_uncore").Run(); err != nil {
				fmt.Fprintf(os.Stderr, "warning: sudo modprobe amd_uncore failed: %v\n", err)
			}
		}
	}

	writeEnvironmentSnapshot(cfg)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	experiments, err := expandExperiments(cfg)
	if err != nil {
		fatalf("experiment matrix: %v", err)
	}
	var results []runResult
	if cfg.Resume {
		results, err = loadResumableResults(cfg.OutputDir, experiments, threadValues, cfg.Runs)
		if err != nil {
			fatalf("load resumable results: %v", err)
		}
		fmt.Printf("Resuming %d successful counted runs from %s; failed or missing runs will be retried.\n", len(results), cfg.OutputDir)
	}
	fitByConfig := make(map[string]string)
	fitSpillByConfig := make(map[string]int)
	if cfg.InspectFit {
		for _, experiment := range experiments {
			select {
			case <-ctx.Done():
				fatalf("interrupted")
			default:
			}
			dir := filepath.Join(cfg.OutputDir, "configs", experiment.Name)
			if cfg.Resume {
				if out, readErr := os.ReadFile(filepath.Join(dir, "fit.txt")); readErr == nil && len(out) > 0 {
					fitByConfig[experiment.Key] = string(out)
					fitSpillByConfig[experiment.Key] = countFitSpillBlocks(string(out))
					continue
				}
			}
			if err := os.MkdirAll(dir, 0o755); err != nil {
				fmt.Fprintf(os.Stderr, "warning: create fit directory: %v\n", err)
				continue
			}
			out, fitErr := inspectFit(ctx, experiment.Config)
			if fitErr != nil {
				fmt.Fprintf(os.Stderr, "warning: fit inspection %s failed: %v\n", experiment.Name, fitErr)
			}
			fitByConfig[experiment.Key] = out
			fitSpillByConfig[experiment.Key] = countFitSpillBlocks(out)
			if err := os.WriteFile(filepath.Join(dir, "fit.txt"), []byte(out), 0o644); err != nil {
				fmt.Fprintf(os.Stderr, "warning: write fit output: %v\n", err)
			}
		}
	}
	for _, experiment := range experiments {
		for _, threads := range experimentThreadValues(experiment, threadValues) {
			if cfg.Resume && cellComplete(cfg.OutputDir, experiment, threads, cfg.Runs, cfg.WarmupRuns) {
				fmt.Printf("\n=== already complete %s threads=%d; skipping ===\n", experiment.Name, threads)
				continue
			}
			select {
			case <-ctx.Done():
				if err := writeAllSummaries(cfg.OutputDir, results); err != nil {
					fmt.Fprintf(os.Stderr, "write partial summaries: %v\n", err)
				}
				fmt.Println("Interrupted; partial results written.")
				return
			default:
			}

			fmt.Printf("\n=== starting %s threads=%d ===\n", experiment.Name, threads)
			session, startErr := startServerSession(ctx, experiment.Config, experiment.Name, threads)
			if startErr != nil {
				fatalf("start server for %s threads=%d: %v", experiment.Name, threads, startErr)
			}

			for warmup := 1; warmup <= cfg.WarmupRuns; warmup++ {
				if ctx.Err() != nil {
					break
				}
				// Every server session needs its own warm-up. A prior warm-up result
				// cannot warm the freshly restarted server used when resuming missing
				// counted repetitions, so rerun it for each incomplete cell.
				fmt.Printf("\n=== warm-up %d/%d %s threads=%d ===\n", warmup, cfg.WarmupRuns, experiment.Name, threads)
				warmupResult := executeRun(ctx, experiment.Config, experiment.Name, experiment.Key, threads, -warmup,
					fitByConfig[experiment.Key], fitSpillByConfig[experiment.Key], session)
				if warmupResult.ExitCode != 0 {
					session.stop()
					fatalf("warm-up failed in %s: %s", warmupResult.RunDir, warmupResult.Error)
				}
			}

			for rep := 1; rep <= cfg.Runs && ctx.Err() == nil; rep++ {
				if cfg.Resume && resumableResult(cfg.OutputDir, experiment, threads, rep) {
					fmt.Printf("\n=== %s threads=%d run=%d/%d already complete; skipping ===\n", experiment.Name, threads, rep, cfg.Runs)
					continue
				}
				fmt.Printf("\n=== %s threads=%d run=%d/%d ===\n", experiment.Name, threads, rep, cfg.Runs)
				res := executeRun(ctx, experiment.Config, experiment.Name, experiment.Key, threads, rep,
					fitByConfig[experiment.Key], fitSpillByConfig[experiment.Key], session)
				results = append(results, res)
				printRunSummary(res)
				if err := writeAllSummaries(cfg.OutputDir, results); err != nil {
					session.stop()
					fatalf("write summaries: %v", err)
				}
				if res.ExitCode != 0 && cfg.FailFast {
					session.stop()
					fatalf("client failed in %s", res.RunDir)
				}
			}
			session.stop()
			if ctx.Err() != nil {
				if err := writeAllSummaries(cfg.OutputDir, results); err != nil {
					fmt.Fprintf(os.Stderr, "write partial summaries: %v\n", err)
				}
				fmt.Println("Interrupted; partial results written.")
				return
			}
		}
	}

	fmt.Println()
	printAggregateSummary(results)
	fmt.Printf("\nResults: %s\n", cfg.OutputDir)
}

func parseFlags() config {
	var cfg config
	flag.StringVar(&cfg.WorkDir, "workdir", "../llama.cpp", "llama.cpp checkout root")
	flag.StringVar(&cfg.OutputDir, "out", "./benchmark-runs", "output directory")
	flag.StringVar(&cfg.ClientCmd, "client-cmd", "", "client benchmark command, executed through --shell (default: standalone prompt run.sh)")
	flag.StringVar(&cfg.PromptScript, "prompt-script", "standalone-prompt/run.sh", "standalone prompt script used when --client-cmd is empty")
	flag.StringVar(&cfg.OpenCodeConfig, "opencode-config", "standalone-prompt/opencode.json", "OpenCode config template; endpoint is set to this run's llama-server")
	flag.StringVar(&cfg.Shell, "shell", "/bin/bash", "shell used for --client-cmd")
	flag.IntVar(&cfg.Runs, "runs", 7, "runs per THREADS value")
	flag.IntVar(&cfg.WarmupRuns, "warmup-runs", 0, "warm-up runs per configuration, excluded from summaries")
	flag.StringVar(&cfg.ThreadValues, "threads", "6,8,12", "comma-separated llama.cpp CPU thread counts")
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
	flag.BoolVar(&cfg.FailFast, "fail-fast", false, "stop on first client failure")
	flag.BoolVar(&cfg.Resume, "resume", false, "resume an existing output directory, reusing only successful matching runs")
	flag.DurationVar(&cfg.ClientTimeout, "client-timeout", 30*time.Minute, "maximum duration of one client benchmark")
	flag.DurationVar(&cfg.HealthTimeout, "health-timeout", 10*time.Minute, "server startup/health timeout")
	flag.DurationVar(&cfg.ServerSettle, "server-settle", 500*time.Millisecond, "delay after server becomes healthy")

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

	flag.StringVar(&cfg.AmdSMI, "amd-smi", "amd-smi", "amd-smi executable")
	flag.IntVar(&cfg.GPU, "gpu", 0, "AMD GPU index")
	flag.BoolVar(&cfg.ProfileGPU, "profile-gpu", true, "capture amd-smi GPU metrics")
	flag.StringVar(&cfg.UProf, "uprof", "AMDuProfPcm", "AMDuProfPcm executable")
	flag.BoolVar(&cfg.ProfileMemory, "profile-memory", true, "capture AMD uProf memory bandwidth")
	flag.BoolVar(&cfg.UProfSudo, "uprof-sudo", true, "run AMDuProfPcm under sudo")
	flag.BoolVar(&cfg.ModprobeUncore, "modprobe-amd-uncore", true, "sudo modprobe amd_uncore before profiling")

	flag.Var(&cfg.ServerArgs, "server-arg", "extra llama-server argument; repeat for multiple arguments")
	flag.Var(&cfg.ServerEnv, "server-env", "extra server environment KEY=VALUE; repeatable")
	flag.Var(&cfg.ClientEnv, "client-env", "extra client environment KEY=VALUE; repeatable")
	flag.Parse()

	return cfg
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
	type axis struct {
		name   string
		values []string
		apply  func(*config, string) error
	}
	valuesWithBase := func(baseValue, values string) []string {
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
	intAxis := func(name, values string, baseValue int, set func(*config, int)) (axis, bool) {
		if strings.TrimSpace(values) == "" {
			return axis{}, false
		}
		return axis{
			name:   name,
			values: valuesWithBase(strconv.Itoa(baseValue), values),
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
	stringAxis := func(name, values, baseValue string, set func(*config, string)) (axis, bool) {
		if strings.TrimSpace(values) == "" {
			return axis{}, false
		}
		return axis{
			name:   name,
			values: valuesWithBase(baseValue, values),
			apply: func(c *config, value string) error {
				set(c, value)
				return nil
			},
		}, true
	}

	var axes []axis
	appendAxis := func(current axis, ok bool) {
		if ok {
			axes = append(axes, current)
		}
	}
	appendAxis(intAxis("parallel", base.ParallelValues, base.Parallel, func(c *config, v int) { c.Parallel = v }))
	appendAxis(intAxis("context", base.ContextValues, base.Context, func(c *config, v int) { c.Context = v }))
	appendAxis(intAxis("batch", base.BatchValues, base.Batch, func(c *config, v int) { c.Batch = v }))
	appendAxis(intAxis("ubatch", base.UBatchValues, base.UBatch, func(c *config, v int) { c.UBatch = v }))
	appendAxis(intAxis("fit-target", base.FitTargetValues, base.FitTarget, func(c *config, v int) { c.FitTarget = v }))
	appendAxis(stringAxis("kv-k", base.KVKValues, base.KVK, func(c *config, v string) { c.KVK = v }))
	appendAxis(stringAxis("kv-v", base.KVVValues, base.KVV, func(c *config, v string) { c.KVV = v }))
	appendAxis(stringAxis("model", base.ModelValues, base.Model, func(c *config, v string) { c.Model = v }))
	if strings.TrimSpace(base.CacheReuseValues) != "" {
		axes = append(axes, axis{
			name:   "cache-reuse",
			values: valuesWithBase(strconv.Itoa(base.CacheReuse), base.CacheReuseValues),
			apply: func(c *config, value string) error {
				n, err := strconv.Atoi(value)
				if err != nil || n < 0 {
					return errors.New("must be a non-negative integer")
				}
				c.CacheReuse = n
				return nil
			},
		})
	}
	if strings.TrimSpace(base.FlashAttentionValues) != "" {
		axes = append(axes, axis{
			name:   "flash-attention",
			values: valuesWithBase(base.FlashAttention, base.FlashAttentionValues),
			apply: func(c *config, value string) error {
				value = strings.ToLower(strings.TrimSpace(value))
				switch value {
				case "on", "off", "auto":
					c.FlashAttention = value
					return nil
				default:
					return errors.New("must be on, off, or auto")
				}
			},
		})
	}
	if strings.TrimSpace(base.PreserveThinkValues) != "" {
		axes = append(axes, axis{
			name:   "preserve-thinking",
			values: valuesWithBase(strconv.FormatBool(base.PreserveThink), base.PreserveThinkValues),
			apply: func(c *config, value string) error {
				v, err := strconv.ParseBool(value)
				if err != nil {
					return errors.New("must be true or false")
				}
				c.PreserveThink = v
				return nil
			},
		})
	}

	configs := []config{base}
	for _, currentAxis := range axes {
		next := make([]config, 0, len(configs)*len(currentAxis.values))
		for _, partial := range configs {
			for _, value := range currentAxis.values {
				candidate := partial
				if err := currentAxis.apply(&candidate, value); err != nil {
					return nil, fmt.Errorf("%s value %q: %w", currentAxis.name, value, err)
				}
				next = append(next, candidate)
			}
		}
		configs = next
	}

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
	return out, nil
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

func executeRun(parent context.Context, cfg config, configName, configKey string, threads, rep int, fitOut string, fitSpill int, session *serverSession) runResult {
	root := experimentRoot(cfg.OutputDir, configName)
	runDir := filepath.Join(root, fmt.Sprintf("threads-%02d", threads), repetitionDir(rep))
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return runResult{ConfigName: configName, ConfigKey: configKey, ThreadCount: threads, Repetition: rep, RunDir: runDir, ExitCode: -1, Error: err.Error()}
	}
	opencodeConfigPath := filepath.Join(runDir, "opencode.json")
	if err := writeRunOpenCodeConfig(cfg.OpenCodeConfig, opencodeConfigPath, session.baseURL+"/v1", cfg.Context); err != nil {
		return failedRun(configName, configKey, threads, rep, runDir, fitOut, fitSpill, fmt.Errorf("prepare OpenCode config: %w", err))
	}

	llamaLog := filepath.Join(runDir, "llama.log")
	serverExited, serverErr := session.proc.exited()
	if serverExited {
		if serverErr == nil {
			serverErr = errors.New("server exited")
		}
		return failedRun(configName, configKey, threads, rep, runDir, fitOut, fitSpill, fmt.Errorf("server unavailable: %w", serverErr))
	}
	serverCommand := append([]string{cfg.ServerBinary}, serverArgs(cfg, threads, session.logPath)...)
	meta := runMeta{
		StartedAt:     time.Now().Format(time.RFC3339Nano),
		ConfigName:    configName,
		ConfigKey:     configKey,
		Warmup:        rep < 0,
		Threads:       threads,
		Repetition:    rep,
		ClientCommand: cfg.ClientCmd,
		Shell:         cfg.Shell,
		ServerCommand: serverCommand,
		ServerEnv:     append([]string(nil), cfg.ServerEnv...),
		ClientEnv:     clientEnvironment(cfg, threads, rep, runDir, opencodeConfigPath),
		GitCommit:     gitCommit(cfg.WorkDir),
		Config: map[string]any{
			"model": cfg.Model, "ctx": cfg.Context, "parallel": cfg.Parallel,
			"reasoning": cfg.Reasoning, "preserve_thinking": cfg.PreserveThink,
			"prompt_session_type": promptSessionType(cfg.PreserveThink),
			"kv_k":                cfg.KVK, "kv_v": cfg.KVV, "batch": cfg.Batch, "ubatch": cfg.UBatch,
			"fit_target": cfg.FitTarget, "cache_reuse": cfg.CacheReuse, "unified_kv": cfg.UnifiedKV,
			"flash_attention":          cfg.FlashAttention,
			"kv_per_slot":              cfg.KVPerSlot,
			"starvation_threshold_tps": cfg.StarvationThreshold,
			"large_prefill_tokens":     cfg.LargePrefillTokens,
		},
	}
	if err := writeJSON(filepath.Join(runDir, "meta.json"), meta); err != nil {
		return failedRun(configName, configKey, threads, rep, runDir, fitOut, fitSpill, fmt.Errorf("write metadata: %w", err))
	}

	logStart, err := logFileSize(session.logPath)
	if err != nil {
		return failedRun(configName, configKey, threads, rep, runDir, fitOut, fitSpill, fmt.Errorf("server log: %w", err))
	}

	var gpuProc, uprofProc *managedProc
	gpuStatus, memoryStatus := "disabled", "disabled"
	if cfg.ProfileGPU {
		gpuStatus = "unavailable"
		gpuPath := filepath.Join(runDir, "gpu.csv")
		gpuErrPath := filepath.Join(runDir, "gpu.err.log")
		gpuFile, e := os.Create(gpuPath)
		if e == nil {
			gpuErr, e2 := os.Create(gpuErrPath)
			if e2 == nil {
				gpuCmd := exec.Command(cfg.AmdSMI, "monitor",
					"--gpu", strconv.Itoa(cfg.GPU),
					"--power-usage", "--gfx", "--mem", "--vram-usage",
					"--watch", "1", "--csv")
				gpuCmd.Dir = cfg.WorkDir
				gpuCmd.Stdout = gpuFile
				gpuCmd.Stderr = gpuErr
				gpuProc, err = startManaged(gpuCmd)
				if err == nil {
					gpuStatus = "running"
				}
				_ = gpuFile.Close()
				_ = gpuErr.Close()
				if err != nil {
					gpuStatus = "failed to start: " + err.Error()
					fmt.Fprintf(os.Stderr, "warning: amd-smi start failed: %v\n", err)
				}
			} else {
				_ = gpuFile.Close()
			}
		}
	}

	if cfg.ProfileMemory {
		memoryStatus = "unavailable"
		uprofPath := filepath.Join(runDir, "memory.csv")
		uprofConsolePath := filepath.Join(runDir, "uprof-console.log")
		uprofConsole, e := os.Create(uprofConsolePath)
		if e != nil {
			fmt.Fprintf(os.Stderr, "warning: uProf console log: %v\n", e)
		} else {
			var argv []string
			if cfg.UProfSudo {
				argv = []string{"sudo", "-n", cfg.UProf}
			} else {
				argv = []string{cfg.UProf}
			}
			argv = append(argv, "-m", "memory", "-a", "-A", "system,package", "-s", "-o", uprofPath)
			uprofCmd := exec.Command(argv[0], argv[1:]...)
			uprofCmd.Dir = cfg.WorkDir
			uprofCmd.Stdout = uprofConsole
			uprofCmd.Stderr = uprofConsole
			uprofProc, err = startManaged(uprofCmd)
			if err == nil {
				memoryStatus = "running"
			}
			_ = uprofConsole.Close()
			if err != nil {
				memoryStatus = "failed to start: " + err.Error()
				fmt.Fprintf(os.Stderr, "warning: AMDuProfPcm start failed: %v\n", err)
			}
		}
	}

	metricsCancel, metricsDone := startMetricSampler(parent, session.baseURL+"/metrics", filepath.Join(runDir, "metrics.csv"))

	start := time.Now()
	clientExit, clientErr := runClient(parent, cfg, threads, rep, runDir)
	end := time.Now()

	metricsCancel()
	metrics := <-metricsDone
	logEnd, logErr := logFileSize(session.logPath)
	if logErr == nil {
		logErr = copyLogRange(session.logPath, llamaLog, logStart, logEnd)
	}
	if logErr != nil {
		fmt.Fprintf(os.Stderr, "warning: capture server log for %s: %v\n", runDir, logErr)
	}
	if elapsed := end.Sub(start).Seconds(); elapsed > 0 {
		metrics.WallGenerationTPS = metrics.PredictedTokensDelta / elapsed
		metrics.WallPromptTPS = metrics.PromptTokensDelta / elapsed
	}

	if uprofProc != nil {
		uprofProc.stop(syscall.SIGINT, 5*time.Second, cfg.UProfSudo)
	}
	if gpuProc != nil {
		gpuProc.stop(syscall.SIGTERM, 3*time.Second, false)
	}
	if clientErr != nil {
		fmt.Fprintf(os.Stderr, "client run error: %v\n", clientErr)
	}

	clientLog := filepath.Join(runDir, "client.log")
	res := runResult{
		ConfigName:     configName,
		ConfigKey:      configKey,
		Warmup:         rep < 0,
		ThreadCount:    threads,
		Repetition:     rep,
		RunDir:         runDir,
		ExitCode:       clientExit,
		WallSeconds:    end.Sub(start).Seconds(),
		Client:         parseClientLog(clientLog),
		Llama:          parseLlamaLog(llamaLog, cfg.StarvationThreshold, cfg.LargePrefillTokens),
		Memory:         parseMemoryLog(filepath.Join(runDir, "memory.csv")),
		GPU:            parseGPULog(filepath.Join(runDir, "gpu.csv")),
		Metrics:        metrics,
		FitSpillBlocks: fitSpill,
		FitOutput:      strings.TrimSpace(fitOut),
	}
	if res.Client.Parsed && res.Client.ReportedSeconds == 0 {
		res.Client.ReportedSeconds = res.WallSeconds
	}
	if cfg.ProfileGPU && gpuStatus == "running" {
		gpuStatus = "unavailable: no parseable samples"
		if res.GPU.Samples > 0 {
			gpuStatus = "captured"
		}
	}
	if cfg.ProfileMemory && memoryStatus == "running" {
		memoryStatus = "unavailable: no parseable samples"
		if res.Memory.Samples > 0 {
			memoryStatus = "captured"
		}
	}
	res.GPUProfiler = gpuStatus
	res.MemoryProfiler = memoryStatus
	if res.ExitCode == 0 && !res.Client.Parsed {
		res.ExitCode = 1
		res.Error = "client benchmark result line was not parsed"
	} else if res.ExitCode == 0 && !res.Client.Passed {
		res.ExitCode = 1
		res.Error = "client benchmark reported FAIL"
	}
	if clientErr != nil {
		res.Error = clientErr.Error()
	}
	if err := writeJSON(filepath.Join(runDir, "result.json"), res); err != nil {
		fmt.Fprintf(os.Stderr, "write result for %s: %v\n", runDir, err)
	}
	return res
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
			return -1, fmt.Errorf("client timeout after %s", cfg.ClientTimeout)
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

func parseLlamaLog(path string, starvationThreshold float64, largePrefillTokens int64) llamaStats {
	f, err := os.Open(path)
	if err != nil {
		return llamaStats{}
	}
	defer f.Close()

	type timing struct {
		tokens int64
		sec    float64
	}
	prompts := map[string]timing{}
	evals := map[string]timing{}
	tasks := map[string]bool{}
	activeTasks := map[string]bool{}
	lowTasks := map[string]bool{}
	slots := map[string]bool{}
	var tg3 []float64
	var lcpSim, lcpKeep []float64
	var cancelled int
	var maxPrefill int64
	var maxConcurrent, largePrefillEvents, starvationEvents int

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if m := reLaunch.FindStringSubmatch(line); m != nil {
			slots[m[1]] = true
			tasks[m[2]] = true
			activeTasks[m[2]] = true
			maxConcurrent = max(maxConcurrent, len(activeTasks))
		}
		if m := rePromptEval.FindStringSubmatch(line); m != nil {
			ms, _ := strconv.ParseFloat(m[2], 64)
			n, _ := strconv.ParseInt(m[3], 10, 64)
			prompts[m[1]] = timing{tokens: n, sec: ms / 1000}
		}
		if m := reEval.FindStringSubmatch(line); m != nil {
			ms, _ := strconv.ParseFloat(m[2], 64)
			n, _ := strconv.ParseInt(m[3], 10, 64)
			evals[m[1]] = timing{tokens: n, sec: ms / 1000}
			delete(activeTasks, m[1])
			delete(lowTasks, m[1])
		}
		if m := reProgress.FindStringSubmatch(line); m != nil {
			v, _ := strconv.ParseFloat(m[4], 64)
			if v >= 0 {
				tg3 = append(tg3, v)
				if v < starvationThreshold && len(activeTasks) > 1 && !lowTasks[m[1]] {
					starvationEvents++
					lowTasks[m[1]] = true
				} else if v >= starvationThreshold {
					delete(lowTasks, m[1])
				}
			}
		}
		if m := reCancel.FindStringSubmatch(line); m != nil {
			// llama-server cancels queued work during teardown after the client is done.
			// Those shutdown cancellations are not workload cancellations.
			if reStopCancel.MatchString(line) {
				continue
			}
			cancelled++
			delete(activeTasks, m[1])
			delete(lowTasks, m[1])
		}
		if m := reLCP.FindStringSubmatch(line); m != nil {
			a, _ := strconv.ParseFloat(m[1], 64)
			b, _ := strconv.ParseFloat(m[2], 64)
			lcpSim = append(lcpSim, a)
			lcpKeep = append(lcpKeep, b)
		}
		if m := rePrefill.FindStringSubmatch(line); m != nil {
			n, _ := strconv.ParseInt(m[2], 10, 64)
			if n >= largePrefillTokens {
				largePrefillEvents++
			}
			if n > maxPrefill {
				maxPrefill = n
			}
		}
	}

	var out llamaStats
	out.TaskCount = len(tasks)
	out.SlotsUsed = len(slots)
	out.CancelledTasks = cancelled
	out.MaxPromptProcessingTokens = maxPrefill
	out.MaxConcurrentTasks = maxConcurrent
	out.LargePrefillEvents = largePrefillEvents
	out.DecodeStarvationEvents = starvationEvents

	for _, x := range prompts {
		out.PromptTokens += x.tokens
		out.PromptSeconds += x.sec
	}
	if out.PromptSeconds > 0 {
		out.PromptTPSWeighted = float64(out.PromptTokens) / out.PromptSeconds
	}
	for _, x := range evals {
		out.GeneratedTokens += x.tokens
		out.EvalSecondsSummed += x.sec
	}
	if out.EvalSecondsSummed > 0 {
		out.TaskTGWeighted = float64(out.GeneratedTokens) / out.EvalSecondsSummed
	}
	if len(tg3) > 0 {
		st := statsOf(tg3)
		out.MinTG3s = st.Min
		out.MaxTG3s = st.Max
		out.MeanTG3s = st.Mean
	}
	if len(lcpSim) > 0 {
		out.LCPCount = len(lcpSim)
		out.MeanLCPSimilarity = statsOf(lcpSim).Mean
		out.MeanLCPKeep = statsOf(lcpKeep).Mean
	}
	return out
}

func parseMemoryLog(path string) memoryStats {
	f, err := os.Open(path)
	if err != nil {
		return memoryStats{}
	}
	defer f.Close()

	var total, read, write []float64
	sc := bufio.NewScanner(f)
	headerFound := false
	totalIdx, readIdx, writeIdx := -1, -1, -1
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		rec, err := csv.NewReader(strings.NewReader(line)).Read()
		if err != nil {
			continue
		}
		if !headerFound {
			for i, field := range rec {
				n := normalizeHeader(field)
				// Use the first system-level occurrence. With -A system,package the
				// same three names occur twice; the first set is the system aggregate.
				switch n {
				case "TOTAL_MEM_BW_GB_S":
					if totalIdx < 0 {
						totalIdx = i
					}
				case "TOTAL_MEM_RDBW_GB_S":
					if readIdx < 0 {
						readIdx = i
					}
				case "TOTAL_MEM_WRBW_GB_S":
					if writeIdx < 0 {
						writeIdx = i
					}
				}
			}
			if totalIdx >= 0 && readIdx >= 0 && writeIdx >= 0 {
				headerFound = true
			}
			continue
		}
		maxIdx := totalIdx
		if readIdx > maxIdx {
			maxIdx = readIdx
		}
		if writeIdx > maxIdx {
			maxIdx = writeIdx
		}
		if len(rec) <= maxIdx {
			continue
		}
		a, e1 := strconv.ParseFloat(strings.TrimSpace(rec[totalIdx]), 64)
		b, e2 := strconv.ParseFloat(strings.TrimSpace(rec[readIdx]), 64)
		c, e3 := strconv.ParseFloat(strings.TrimSpace(rec[writeIdx]), 64)
		if e1 != nil || e2 != nil || e3 != nil {
			continue
		}
		total = append(total, a)
		read = append(read, b)
		write = append(write, c)
	}
	ts, rs, ws := statsOf(total), statsOf(read), statsOf(write)
	return memoryStats{
		ReadValues: append([]float64(nil), read...),
		Samples:    len(total), MeanTotal: ts.Mean, MaxTotal: ts.Max,
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

	var util, memUtil, power, vram []float64
	sc := bufio.NewScanner(f)
	var header []string
	index := map[string]int{}

	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "WARNING") {
			continue
		}
		rec, err := csv.NewReader(strings.NewReader(line)).Read()
		if err != nil || len(rec) < 2 {
			continue
		}
		if header == nil {
			hasGPU := false
			for _, x := range rec {
				u := normalizeHeader(x)
				if u == "GPU" {
					hasGPU = true
				}
			}
			if !hasGPU {
				continue
			}
			header = rec
			for i, x := range header {
				index[normalizeHeader(x)] = i
			}
			continue
		}

		if i := findHeaderIndex(index, "GFX_UTIL", "GFX%", "GFXUTIL", "GFX", "GFX_ACTIVITY", "GPU_UTIL", "GPU_UTILIZATION", "GPU%"); i >= 0 && i < len(rec) {
			if v, ok := parseNumber(rec[i]); ok {
				util = append(util, v)
			}
		}
		if i := findHeaderIndex(index, "MEM_UTIL", "MEM%", "MEMUTIL", "MEM_ACTIVITY", "MEM"); i >= 0 && i < len(rec) {
			if v, ok := parseNumber(rec[i]); ok {
				memUtil = append(memUtil, v)
			}
		}
		if i := findHeaderContains(index, "POWER"); i >= 0 && i < len(rec) {
			if v, ok := parseNumber(rec[i]); ok {
				power = append(power, v)
			}
		}
		if i := findHeaderIndex(index, "VRAM_USED", "VRAM_USED_MB", "VRAM_USAGE", "VRAM_USAGE_MB"); i >= 0 && i < len(rec) {
			field := rec[i]
			if strings.Contains(field, "/") {
				field = strings.SplitN(field, "/", 2)[0]
			}
			if v, ok := parseNumber(field); ok {
				// amd-smi monitor normally reports MB for discrete GPUs.
				vram = append(vram, v)
			}
		}
	}

	us, mus, ps, vs := statsOf(util), statsOf(memUtil), statsOf(power), statsOf(vram)
	return gpuStats{
		UtilValues: append([]float64(nil), util...), PowerValues: append([]float64(nil), power...),
		Samples: max(len(util), len(memUtil), len(power), len(vram)), MeanUtil: us.Mean, MaxUtil: us.Max,
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
	by := map[string][]runResult{}
	for _, r := range results {
		key := fmt.Sprintf("%s|%d", r.ConfigKey, r.ThreadCount)
		by[key] = append(by[key], r)
	}
	var keys []string
	for k := range by {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var out []aggregate
	for _, k := range keys {
		rs := by[k]
		var wall, client, taskTG, metricTG, pp, mem, gpu, power, cancelled, starvation []float64
		var memRaw, gpuRaw, powerRaw []float64
		success := 0
		spill := 0
		totalCancelled := 0
		for _, r := range rs {
			totalCancelled += r.Llama.CancelledTasks
			if r.FitSpillBlocks > spill {
				spill = r.FitSpillBlocks
			}
			if r.ExitCode != 0 {
				continue
			}
			success++
			wall = append(wall, r.WallSeconds)
			if r.Client.ReportedSeconds > 0 {
				client = append(client, r.Client.ReportedSeconds)
			}
			if r.Llama.TaskTGWeighted > 0 {
				taskTG = append(taskTG, r.Llama.TaskTGWeighted)
			}
			if r.Metrics.WallGenerationTPS > 0 {
				metricTG = append(metricTG, r.Metrics.WallGenerationTPS)
			}
			if r.Llama.PromptTPSWeighted > 0 {
				pp = append(pp, r.Llama.PromptTPSWeighted)
			}
			if r.Memory.MeanRead > 0 {
				mem = append(mem, r.Memory.MeanRead)
				memRaw = append(memRaw, r.Memory.ReadValues...)
			}
			gpuStats := r.GPU
			// Sample slices are excluded from result.json to keep the files small.
			// Rehydrate them from the raw telemetry when a resumed run is summarized.
			if gpuStats.Samples > 0 && len(gpuStats.UtilValues) == 0 && len(gpuStats.PowerValues) == 0 {
				gpuStats = parseGPULog(filepath.Join(r.RunDir, "gpu.csv"))
			}
			if gpuStats.Samples > 0 {
				if gpuStats.MeanUtil > 0 {
					gpu = append(gpu, gpuStats.MeanUtil)
				}
				gpuRaw = append(gpuRaw, gpuStats.UtilValues...)
			}
			if gpuStats.Samples > 0 && gpuStats.MeanPower > 0 {
				power = append(power, gpuStats.MeanPower)
				powerRaw = append(powerRaw, gpuStats.PowerValues...)
			}
			cancelled = append(cancelled, float64(r.Llama.CancelledTasks))
			starvation = append(starvation, float64(r.Llama.DecodeStarvationEvents))
		}
		ws := statsOf(wall)
		memSamples, gpuSamples, powerSamples := statsOf(memRaw), statsOf(gpuRaw), statsOf(powerRaw)
		out = append(out, aggregate{
			ConfigName: rs[0].ConfigName, ConfigKey: rs[0].ConfigKey,
			Threads: rs[0].ThreadCount, Runs: len(rs), Success: success,
			MeanWall: ws.Mean, MedianWall: ws.Median, P95Wall: ws.P95, MinWall: ws.Min, MaxWall: ws.Max,
			MedianClient:     statsOf(client).Median,
			MedianTaskTG:     statsOf(taskTG).Median,
			MedianMetricTG:   statsOf(metricTG).Median,
			MedianPromptTPS:  statsOf(pp).Median,
			MedianMemRead:    statsOf(mem).Median,
			P95MemRead:       memSamples.P95,
			MaxMemRead:       memSamples.Max,
			MedianGPUUtil:    statsOf(gpu).Median,
			P95GPUUtil:       gpuSamples.P95,
			MaxGPUUtil:       gpuSamples.Max,
			MedianPower:      statsOf(power).Median,
			P95Power:         powerSamples.P95,
			MaxPower:         powerSamples.Max,
			MedianCancelled:  statsOf(cancelled).Median,
			MedianStarvation: statsOf(starvation).Median,
			TotalCancelled:   totalCancelled,
			SpillBlocks:      spill,
		})
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
