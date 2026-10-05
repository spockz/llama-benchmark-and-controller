// Package main implements llama-modelctl, which switches models on a llama.cpp
// router and preserves compatible slot-zero prompt caches.
package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"
)

const defaultPollInterval = 500 * time.Millisecond

type options struct {
	serverURL   string
	host        string
	port        int
	token       string
	presetPath  string
	stateDir    string
	session     string
	waitTimeout time.Duration
	idleTimeout time.Duration
	poll        time.Duration
	modelHashes map[string]string
}

type preset struct {
	ID     string
	Values map[string]string
}

type modelStatus struct {
	Value    string   `json:"value"`
	Args     []string `json:"args"`
	Failed   bool     `json:"failed"`
	ExitCode int      `json:"exit_code"`
}

type modelRecord struct {
	ID        string      `json:"id"`
	Source    string      `json:"source"`
	Status    modelStatus `json:"status"`
	ModelPath string      `json:"model_path"`
	Path      string      `json:"path"`
}

type modelList struct {
	Data []modelRecord `json:"data"`
}

type slot struct {
	ID           int  `json:"id"`
	IsProcessing bool `json:"is_processing"`
}

type cacheManifest struct {
	Version          int               `json:"version"`
	ModelID          string            `json:"model_id"`
	Session          string            `json:"session"`
	Source           string            `json:"source"`
	ResolvedModel    string            `json:"resolved_model_path"`
	ModelSHA256      string            `json:"model_sha256"`
	Preset           map[string]string `json:"preset"`
	AdapterSettings  map[string]string `json:"adapter_settings"`
	ContextRoPESet   map[string]string `json:"context_rope_settings"`
	KVTypes          map[string]string `json:"kv_types"`
	BuildInfo        string            `json:"llama_cpp_build_info"`
	CacheFilename    string            `json:"cache_filename"`
	SavedAt          time.Time         `json:"saved_at"`
	Restorable       bool              `json:"restorable"`
	RestorationNotes string            `json:"restoration_notes,omitempty"`
}

type apiClient struct {
	base  *url.URL
	token string
	http  *http.Client
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "llama-modelctl:", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	return runWithHTTP(args, stdout, stderr, nil)
}

func runWithHTTP(args []string, stdout, stderr io.Writer, client *http.Client) error {
	opt, args, err := parseGlobalArgs(args, stderr)
	if err != nil {
		return err
	}
	if len(args) == 0 {
		printUsage(stderr)
		return errors.New("missing command")
	}
	if isHelp(args[0]) {
		printUsage(stdout)
		return nil
	}
	api, err := newAPIClient(opt, client)
	if err != nil {
		return err
	}
	return executeCommand(context.Background(), api, opt, args, stdout, stderr)
}

func parseGlobalArgs(args []string, stderr io.Writer) (options, []string, error) {
	fs := flag.NewFlagSet("llama-modelctl", flag.ContinueOnError)
	fs.SetOutput(stderr)
	opt := options{}
	fs.StringVar(&opt.serverURL, "server", envOr("LLAMA_SERVER_URL", ""), "server base URL (overrides host and port)")
	fs.StringVar(&opt.host, "host", envOr("LLAMA_SERVER_HOST", "127.0.0.1"), "server host")
	fs.IntVar(&opt.port, "port", envInt("LLAMA_SERVER_PORT", 1234), "server port")
	fs.StringVar(&opt.token, "token", os.Getenv("LLAMA_SERVER_TOKEN"), "optional bearer token (or LLAMA_SERVER_TOKEN)")
	fs.StringVar(&opt.presetPath, "preset", envOr("LLAMA_MODELS_PRESET", "models.ini"), "router models INI used to resolve preset settings")
	fs.StringVar(&opt.stateDir, "state-dir", defaultStateDir(), "local/shared directory for locks and cache manifests")
	fs.StringVar(&opt.session, "session", "default", "conversation/session identity for cache isolation")
	fs.DurationVar(&opt.waitTimeout, "wait-timeout", 30*time.Minute, "maximum time to wait for model load/unload")
	fs.DurationVar(&opt.idleTimeout, "idle-timeout", 30*time.Second, "maximum time to wait for slot 0 to become idle")
	fs.DurationVar(&opt.poll, "poll-interval", defaultPollInterval, "state polling interval")
	hashFlags := stringMapFlag{}
	fs.Var(&hashFlags, "model-sha256", "trusted model digest as model-id=hex-sha256 (repeatable, for remote model files)")
	if err := fs.Parse(args); err != nil {
		return options{}, nil, err
	}
	opt.modelHashes = hashFlags
	if opt.waitTimeout <= 0 || opt.idleTimeout <= 0 || opt.poll <= 0 {
		return options{}, nil, errors.New("wait and poll durations must be positive")
	}
	return opt, fs.Args(), nil
}

func isHelp(command string) bool {
	return command == "help" || command == "-h" || command == "--help"
}

func newAPIClient(opt options, client *http.Client) (*apiClient, error) {
	if opt.serverURL == "" {
		if opt.port < 1 || opt.port > 65535 {
			return nil, fmt.Errorf("invalid port %d", opt.port)
		}
		opt.serverURL = "http://" + net.JoinHostPort(opt.host, fmt.Sprint(opt.port))
	}
	parsedURL, err := url.Parse(opt.serverURL)
	if err != nil || (parsedURL.Scheme != "http" && parsedURL.Scheme != "https") || parsedURL.Host == "" {
		return nil, fmt.Errorf("invalid --server URL %q", opt.serverURL)
	}
	if client == nil {
		client = &http.Client{Timeout: 60 * time.Second}
	}
	return &apiClient{base: parsedURL, token: opt.token, http: client}, nil
}

func executeCommand(ctx context.Context, api *apiClient, opt options, args []string, stdout, stderr io.Writer) error {
	command := args[0]
	commandArgs := args[1:]
	known := map[string]bool{"status": true, "load": true, "save": true, "switch": true, "unload": true, "config": true}
	if !known[command] {
		printUsage(stderr)
		return fmt.Errorf("unknown command %q", command)
	}
	if command == "status" {
		if len(commandArgs) != 0 {
			return errors.New("status takes no arguments")
		}
		return commandStatus(ctx, api, stdout)
	}
	presets, err := readPresets(opt.presetPath)
	if err != nil {
		return fmt.Errorf("read presets %s: %w", opt.presetPath, err)
	}
	return dispatchConfiguredCommand(ctx, api, opt, presets, command, commandArgs, stdout)
}

func dispatchConfiguredCommand(ctx context.Context, api *apiClient, opt options, presets map[string]preset, command string, args []string, stdout io.Writer) error {
	switch command {
	case "config":
		if len(args) != 1 {
			return errors.New("usage: llama-modelctl [options] config <model-id>")
		}
		return commandConfig(ctx, api, presets, args[0], stdout)
	default:
		return withLock(opt.stateDir, func() error {
			return dispatchLockedCommand(ctx, api, opt, presets, command, args, stdout)
		})
	}
}

func dispatchLockedCommand(ctx context.Context, api *apiClient, opt options, presets map[string]preset, command string, args []string, stdout io.Writer) error {
	switch command {
	case "load":
		if len(args) != 1 {
			return errors.New("usage: llama-modelctl [options] load <model-id>")
		}
		return loadOnly(ctx, api, opt, presets, args[0], stdout)
	case "save":
		if len(args) != 0 {
			return errors.New("save takes no arguments")
		}
		return saveCurrent(ctx, api, opt, presets, stdout)
	case "unload":
		return unloadCommand(ctx, api, opt, presets, args, stdout)
	case "switch":
		return switchCommand(ctx, api, opt, presets, args, stdout)
	default:
		return errors.New("unreachable command")
	}
}

func saveCurrent(ctx context.Context, api *apiClient, opt options, presets map[string]preset, stdout io.Writer) error {
	models, err := api.models(ctx)
	if err != nil {
		return err
	}
	active, err := activeModel(models)
	if err != nil {
		return err
	}
	if active == nil {
		return errors.New("no model is active")
	}
	_, err = saveActive(ctx, api, opt, presets, *active, stdout)
	return err
}

type stringMapFlag map[string]string

func (m *stringMapFlag) String() string { return "model-id=sha256" }
func (m *stringMapFlag) Set(s string) error {
	parts := strings.SplitN(s, "=", 2)
	if len(parts) != 2 || strings.TrimSpace(parts[0]) == "" {
		return errors.New("expected model-id=hex-sha256")
	}
	digest := strings.ToLower(strings.TrimSpace(parts[1]))
	if len(digest) != 64 {
		return errors.New("SHA-256 digest must be 64 hexadecimal characters")
	}
	if _, err := hex.DecodeString(digest); err != nil {
		return fmt.Errorf("invalid SHA-256 digest: %w", err)
	}
	if *m == nil {
		*m = map[string]string{}
	}
	(*m)[parts[0]] = digest
	return nil
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, `Usage: llama-modelctl [global options] <command>

Commands:
  status                 list configured server models and states
  load <model-id>        load only when no other model is active
  save                   save active model's slot-0 prompt cache
  switch <model-id>      save, unload, load target, then restore compatible cache
  unload                 save and unload every loaded model, freeing the GPU
  config <model-id>      show preset settings and loaded runtime properties

Global options:
  --server URL           full server URL; overrides --host and --port
  --host HOST --port N   default 127.0.0.1:1234
  --token TOKEN          bearer token (or LLAMA_SERVER_TOKEN)
  --preset FILE          router INI (default ./models.ini)
  --state-dir DIR        local/shared lock and manifest directory
  --session ID           cache namespace (default "default")
  --model-sha256 ID=HEX  trusted weight digest for an inaccessible remote model
  --wait-timeout D       load/unload wait (default 30m)
  --idle-timeout D       slot-idle wait (default 30s)

Use --discard-cache with switch or unload only when you explicitly accept losing
the current slot cache if the save request fails.`)
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envInt(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	var parsed int
	if _, err := fmt.Sscan(value, &parsed); err != nil {
		return fallback
	}
	return parsed
}

func defaultStateDir() string {
	if xdg := os.Getenv("XDG_STATE_HOME"); xdg != "" {
		return filepath.Join(xdg, "llama-modelctl")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "llama-modelctl")
	}
	return filepath.Join(home, ".local", "state", "llama-modelctl")
}

func readPresets(path string) (map[string]preset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	out, err := readPresetSections(newLineScanner(file))
	if err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, errors.New("no named presets found")
	}
	merged := mergeGlobalPreset(out)
	if len(merged) == 0 {
		return nil, errors.New("no named presets found")
	}
	return merged, nil
}

func readPresetSections(scanner *bufio.Scanner) (map[string]preset, error) {
	out := map[string]preset{}
	var current string
	values := map[string]string{}
	flush := func() {
		if current == "" {
			return
		}
		copyValues := make(map[string]string, len(values))
		for key, value := range values {
			copyValues[key] = value
		}
		out[current] = preset{ID: current, Values: copyValues}
	}
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ";") || strings.HasPrefix(line, "#") || line == "version = 1" {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			flush()
			current = strings.TrimSpace(line[1 : len(line)-1])
			if current == "" {
				return nil, fmt.Errorf("line %d: empty section name", lineNo)
			}
			values = map[string]string{}
			continue
		}
		if current == "" {
			continue
		}
		index := strings.IndexByte(line, '=')
		if index < 1 {
			return nil, fmt.Errorf("line %d: expected key = value", lineNo)
		}
		key := strings.ToLower(strings.TrimSpace(line[:index]))
		value := strings.TrimSpace(line[index+1:])
		if key == "" {
			return nil, fmt.Errorf("line %d: empty key", lineNo)
		}
		values[key] = strings.Trim(value, "\"'")
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	flush()
	return out, nil
}

func mergeGlobalPreset(presets map[string]preset) map[string]preset {
	global, ok := presets["*"]
	if !ok {
		return presets
	}
	delete(presets, "*")
	for id, model := range presets {
		merged := cloneMap(global.Values)
		for key, value := range model.Values {
			merged[key] = value
		}
		model.Values = merged
		presets[id] = model
	}
	return presets
}

// newLineScanner permits long adapter and template values in the INI file.
func newLineScanner(r io.Reader) *bufio.Scanner {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 4096), 1024*1024)
	return s
}

func commandStatus(ctx context.Context, api *apiClient, out io.Writer) error {
	models, err := api.models(ctx)
	if err != nil {
		return err
	}
	active := 0
	fmt.Fprintf(out, "%-46s %-12s %s\n", "MODEL", "STATE", "ACTIVE")
	for _, model := range models {
		isActive := model.Status.Value == "loaded" || model.Status.Value == "sleeping"
		if isActive {
			active++
		}
		marker := ""
		if isActive {
			marker = "*"
		}
		if model.Status.Failed {
			fmt.Fprintf(out, "%-46s %-12s %s (exit %d)\n", model.ID, model.Status.Value, marker, model.Status.ExitCode)
		} else {
			fmt.Fprintf(out, "%-46s %-12s %s\n", model.ID, model.Status.Value, marker)
		}
	}
	if active > 1 {
		return fmt.Errorf("router reports %d active models; expected at most one", active)
	}
	return nil
}

func commandConfig(ctx context.Context, api *apiClient, presets map[string]preset, id string, out io.Writer) error {
	modelPreset, ok := presets[id]
	if !ok {
		return fmt.Errorf("model %q is not a section in the preset file", id)
	}
	fmt.Fprintf(out, "Preset: %s\n", id)
	printMap(out, modelPreset.Values)
	models, err := api.models(ctx)
	if err != nil {
		return fmt.Errorf("read server state (preset shown above): %w", err)
	}
	model, ok := findModel(models, id)
	if !ok {
		return fmt.Errorf("preset %q is not currently listed by the server", id)
	}
	fmt.Fprintf(out, "\nServer state: %s\n", model.Status.Value)
	if model.Status.Value != "loaded" && model.Status.Value != "sleeping" {
		return nil
	}
	props, err := api.props(ctx, id)
	if err != nil {
		return fmt.Errorf("read loaded runtime properties: %w", err)
	}
	encoded, err := json.MarshalIndent(props, "", "  ")
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "Runtime properties:\n%s\n", encoded)
	return nil
}

func printMap(out io.Writer, values map[string]string) {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		fmt.Fprintf(out, "  %s = %s\n", key, values[key])
	}
}

func loadOnly(ctx context.Context, api *apiClient, opt options, presets map[string]preset, target string, out io.Writer) error {
	if _, ok := presets[target]; !ok {
		return fmt.Errorf("model %q is not configured in %s", target, opt.presetPath)
	}
	models, err := api.models(ctx)
	if err != nil {
		return err
	}
	if _, ok := findModel(models, target); !ok {
		return fmt.Errorf("model %q is not available in the router; verify its INI section", target)
	}
	active, err := activeModel(models)
	if err != nil {
		return err
	}
	if active != nil && active.ID != target {
		return fmt.Errorf("refusing to load %q while %q is %s; use switch to save and unload first", target, active.ID, active.Status.Value)
	}
	if active != nil {
		if active.Status.Value == "loaded" {
			fmt.Fprintf(out, "Model %q is already loaded.\n", target)
			return nil
		}
		if err := api.postModel(ctx, "/models/load", target); err != nil {
			return err
		}
		return waitForState(ctx, api, target, opt.waitTimeout, opt.poll, "loaded")
	}
	if err := api.postModel(ctx, "/models/load", target); err != nil {
		return err
	}
	if err := waitForState(ctx, api, target, opt.waitTimeout, opt.poll, "loaded"); err != nil {
		return fmt.Errorf("load %q: %w; actual state: %s", target, err, stateSummary(api, target))
	}
	fmt.Fprintf(out, "Loaded %q.\n", target)
	return nil
}

func switchCommand(ctx context.Context, api *apiClient, opt options, presets map[string]preset, args []string, out io.Writer) error {
	target, discard, err := parseSwitchArgs(args)
	if err != nil {
		return err
	}
	if err := verifyTarget(ctx, api, opt, presets, target); err != nil {
		return err
	}
	models, err := api.models(ctx)
	if err != nil {
		return err
	}
	active, err := activeModel(models)
	if err != nil {
		return err
	}
	if active != nil && active.ID == target {
		return activateSameModel(ctx, api, opt, target, active.Status.Value, out)
	}
	if active != nil {
		if err := unloadCurrent(ctx, api, opt, presets, *active, discard, out); err != nil {
			return err
		}
	}
	return loadAndRestore(ctx, api, opt, presets, target, out)
}

func parseSwitchArgs(args []string) (string, bool, error) {
	fs := flag.NewFlagSet("switch", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	discard := fs.Bool("discard-cache", false, "continue if saving active cache fails")
	if err := fs.Parse(args); err != nil {
		return "", false, err
	}
	if fs.NArg() != 1 {
		return "", false, errors.New("usage: llama-modelctl [options] switch [--discard-cache] <model-id>")
	}
	return fs.Arg(0), *discard, nil
}

func verifyTarget(ctx context.Context, api *apiClient, opt options, presets map[string]preset, target string) error {
	if _, ok := presets[target]; !ok {
		return fmt.Errorf("model %q is not configured in %s", target, opt.presetPath)
	}
	models, err := api.models(ctx)
	if err != nil {
		return err
	}
	if _, ok := findModel(models, target); !ok {
		return fmt.Errorf("model %q is not available in router; verify its INI section", target)
	}
	return nil
}

func activateSameModel(ctx context.Context, api *apiClient, opt options, target, state string, out io.Writer) error {
	if state == "loaded" {
		fmt.Fprintf(out, "Model %q is already active.\n", target)
		return nil
	}
	if err := api.postModel(ctx, "/models/load", target); err != nil {
		return err
	}
	if err := waitForState(ctx, api, target, opt.waitTimeout, opt.poll, "loaded"); err != nil {
		return err
	}
	fmt.Fprintf(out, "Model %q is active.\n", target)
	return nil
}

func unloadCurrent(ctx context.Context, api *apiClient, opt options, presets map[string]preset, active modelRecord, discard bool, out io.Writer) error {
	if err := waitSlotIdle(ctx, api, opt, active.ID); err != nil {
		return fmt.Errorf("refusing to interrupt active generation: %w", err)
	}
	if _, err := saveActive(ctx, api, opt, presets, active, out); err != nil {
		if !discard {
			return fmt.Errorf("cache save failed; model remains loaded. Use --discard-cache only if losing this cache is acceptable: %w", err)
		}
		fmt.Fprintf(out, "WARNING: discarding active cache after save failure: %v\n", err)
	}
	if err := waitSlotIdle(ctx, api, opt, active.ID); err != nil {
		return fmt.Errorf("model %q received work again after cache save; refusing to unload: %w", active.ID, err)
	}
	if err := api.postModel(ctx, "/models/unload", active.ID); err != nil {
		return fmt.Errorf("unload %q: %w; actual state: %s", active.ID, err, stateSummary(api, active.ID))
	}
	if err := waitForState(ctx, api, active.ID, opt.waitTimeout, opt.poll, "unloaded"); err != nil {
		return fmt.Errorf("unload %q: %w; actual state: %s", active.ID, err, stateSummary(api, active.ID))
	}
	fmt.Fprintf(out, "Unloaded %q.\n", active.ID)
	return nil
}

func loadAndRestore(ctx context.Context, api *apiClient, opt options, presets map[string]preset, target string, out io.Writer) error {
	if err := api.postModel(ctx, "/models/load", target); err != nil {
		return fmt.Errorf("load %q after unload: %w; actual state: %s", target, err, stateSummary(api, target))
	}
	if err := waitForState(ctx, api, target, opt.waitTimeout, opt.poll, "loaded"); err != nil {
		return fmt.Errorf("load %q: %w; actual state: %s", target, err, stateSummary(api, target))
	}
	model, ok := findModelFromAPI(ctx, api, target)
	if !ok {
		return fmt.Errorf("%q loaded but disappeared from GET /models", target)
	}
	manifest, key, err := makeManifest(ctx, api, opt, presets, model)
	if err != nil {
		return fmt.Errorf("%q loaded, but cache compatibility could not be checked: %w", target, err)
	}
	path := manifestPath(opt.stateDir, key)
	previous, err := readManifest(path)
	if errors.Is(err, os.ErrNotExist) {
		fmt.Fprintf(out, "Loaded %q with an empty cache (no saved compatible cache for session %q).\n", target, opt.session)
		return nil
	}
	if err != nil {
		return fmt.Errorf("%q loaded, but cache manifest %s is unreadable: %w", target, path, err)
	}
	if !previous.Restorable || !manifest.Restorable || !compatible(*previous, manifest) {
		fmt.Fprintf(out, "Loaded %q with an empty cache; saved cache was rejected as incompatible or unverifiable.\n", target)
		return nil
	}
	if err := waitSlotIdle(ctx, api, opt, target); err != nil {
		return fmt.Errorf("%q loaded, but slot 0 did not become idle for restore: %w", target, err)
	}
	result, err := api.slotAction(ctx, target, "restore", previous.CacheFilename)
	if err != nil {
		if isClientError(err) {
			fmt.Fprintf(out, "Loaded %q with an empty cache; saved cache is missing or rejected by llama-server: %v\n", target, err)
			return nil
		}
		return fmt.Errorf("%q loaded, but cache restore failed: %w", target, err)
	}
	if result.NRestored == 0 {
		fmt.Fprintf(out, "Loaded %q; saved cache had no reusable tokens, continuing cold.\n", target)
		return nil
	}
	fmt.Fprintf(out, "Loaded %q and restored %d slot-0 cache tokens for session %q.\n", target, result.NRestored, opt.session)
	return nil
}

func unloadCommand(ctx context.Context, api *apiClient, opt options, presets map[string]preset, args []string, out io.Writer) error {
	fs := flag.NewFlagSet("unload", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	discard := fs.Bool("discard-cache", false, "continue if saving active cache fails")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("usage: llama-modelctl [options] unload [--discard-cache]")
	}
	models, err := api.models(ctx)
	if err != nil {
		return err
	}
	active, err := activeModels(models)
	if err != nil {
		return err
	}
	if len(active) == 0 {
		fmt.Fprintln(out, "No model is active; GPU is already free of router models.")
		return nil
	}
	for _, model := range active {
		if err := unloadOne(ctx, api, opt, presets, model, *discard, out); err != nil {
			return err
		}
	}
	remaining, err := api.models(ctx)
	if err != nil {
		return fmt.Errorf("models were unloaded, but final GPU state could not be verified: %w", err)
	}
	if stillActive, err := activeModels(remaining); err != nil {
		return fmt.Errorf("models were unloaded, but final GPU state is transitional: %w", err)
	} else if len(stillActive) > 0 {
		return fmt.Errorf("unload incomplete; models still active: %s", modelIDs(stillActive))
	}
	fmt.Fprintln(out, "All router models are unloaded; GPU is available for other tasks.")
	return nil
}

func unloadOne(ctx context.Context, api *apiClient, opt options, presets map[string]preset, model modelRecord, discard bool, out io.Writer) error {
	if err := waitSlotIdle(ctx, api, opt, model.ID); err != nil {
		return fmt.Errorf("refusing to interrupt generation for %q: %w", model.ID, err)
	}
	cacheSaved, err := saveBeforeUnload(ctx, api, opt, presets, model, discard, out)
	if err != nil {
		return err
	}
	if err := unloadActive(ctx, api, opt, model); err != nil {
		return err
	}
	if cacheSaved {
		fmt.Fprintf(out, "Unloaded %q; cache saved.\n", model.ID)
	} else {
		fmt.Fprintf(out, "Unloaded %q; cache was discarded.\n", model.ID)
	}
	return nil
}

func modelIDs(models []modelRecord) string {
	ids := make([]string, len(models))
	for i, model := range models {
		ids[i] = model.ID
	}
	return strings.Join(ids, ", ")
}

func saveBeforeUnload(ctx context.Context, api *apiClient, opt options, presets map[string]preset, active modelRecord, discard bool, out io.Writer) (bool, error) {
	if _, err := saveActive(ctx, api, opt, presets, active, out); err != nil {
		if !discard {
			return false, fmt.Errorf("cache save failed; model remains loaded. Use --discard-cache only if losing this cache is acceptable: %w", err)
		}
		fmt.Fprintf(out, "WARNING: discarding active cache after save failure: %v\n", err)
		return false, nil
	}
	return true, nil
}

func unloadActive(ctx context.Context, api *apiClient, opt options, active modelRecord) error {
	if err := waitSlotIdle(ctx, api, opt, active.ID); err != nil {
		return fmt.Errorf("model %q received work again after cache save; refusing to unload: %w", active.ID, err)
	}
	if err := api.postModel(ctx, "/models/unload", active.ID); err != nil {
		return fmt.Errorf("unload %q: %w; actual state: %s", active.ID, err, stateSummary(api, active.ID))
	}
	if err := waitForState(ctx, api, active.ID, opt.waitTimeout, opt.poll, "unloaded"); err != nil {
		return fmt.Errorf("unload %q: %w; actual state: %s", active.ID, err, stateSummary(api, active.ID))
	}
	return nil
}

type slotActionResult struct {
	NSaved    int    `json:"n_saved"`
	NRestored int    `json:"n_restored"`
	Filename  string `json:"filename"`
}

func saveActive(ctx context.Context, api *apiClient, opt options, presets map[string]preset, active modelRecord, out io.Writer) (string, error) {
	if _, ok := presets[active.ID]; !ok {
		return "", fmt.Errorf("active model %q has no configured INI section; refusing to create an untracked cache", active.ID)
	}
	if err := waitSlotIdle(ctx, api, opt, active.ID); err != nil {
		return "", err
	}
	model, ok := findModelFromAPI(ctx, api, active.ID)
	if !ok {
		return "", errors.New("active model is no longer listed by GET /models")
	}
	manifest, key, err := makeManifest(ctx, api, opt, presets, model)
	if err != nil {
		return "", err
	}
	filename := "slot0-" + key + ".bin"
	result, err := api.slotAction(ctx, active.ID, "save", filename)
	if err != nil {
		return "", err
	}
	manifest.CacheFilename = filename
	manifest.SavedAt = time.Now().UTC()
	if err := writeManifest(opt.stateDir, key, manifest); err != nil {
		return filename, fmt.Errorf("server saved cache %q but writing local manifest failed: %w", filename, err)
	}
	if !manifest.Restorable {
		fmt.Fprintf(out, "Saved slot 0 as %q (%d tokens), but marked it non-restorable: %s\n", filename, result.NSaved, manifest.RestorationNotes)
	} else {
		fmt.Fprintf(out, "Saved slot 0 as %q (%d tokens).\n", filename, result.NSaved)
	}
	return filename, nil
}

func waitSlotIdle(ctx context.Context, api *apiClient, opt options, modelID string) error {
	deadline := time.Now().Add(opt.idleTimeout)
	for {
		slots, err := api.slots(ctx, modelID)
		if err != nil {
			return fmt.Errorf("cannot inspect slot 0: %w", err)
		}
		found := false
		for _, slot := range slots {
			if slot.ID == 0 {
				found = true
				if !slot.IsProcessing {
					return nil
				}
			}
		}
		if !found {
			return fmt.Errorf("model %q has no slot 0", modelID)
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("slot 0 remained busy for %s", opt.idleTimeout)
		}
		time.Sleep(minDuration(opt.poll, time.Until(deadline)))
	}
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

func waitForState(ctx context.Context, api *apiClient, id string, timeout, poll time.Duration, desired string) error {
	deadline := time.Now().Add(timeout)
	for {
		models, err := api.models(ctx)
		if err != nil {
			return err
		}
		model, ok := findModel(models, id)
		if !ok {
			return fmt.Errorf("model %q disappeared from GET /models", id)
		}
		if desired == "loaded" && model.Status.Value == "loaded" {
			return nil
		}
		if desired == "unloaded" && (model.Status.Value == "unloaded" || model.Status.Value == "downloaded") {
			return nil
		}
		if model.Status.Failed {
			return fmt.Errorf("model entered failed state (exit %d)", model.Status.ExitCode)
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("timed out after %s waiting for state %q (current %q)", timeout, desired, model.Status.Value)
		}
		time.Sleep(minDuration(poll, time.Until(deadline)))
	}
}

func stateSummary(api *apiClient, id string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	models, err := api.models(ctx)
	if err != nil {
		return "unavailable: " + err.Error()
	}
	model, ok := findModel(models, id)
	if !ok {
		return "not-listed"
	}
	return model.Status.Value
}

func findModelFromAPI(ctx context.Context, api *apiClient, id string) (modelRecord, bool) {
	models, err := api.models(ctx)
	if err != nil {
		return modelRecord{}, false
	}
	return findModel(models, id)
}

func findModel(models []modelRecord, id string) (modelRecord, bool) {
	for _, model := range models {
		if model.ID == id {
			return model, true
		}
	}
	return modelRecord{}, false
}

func activeModel(models []modelRecord) (*modelRecord, error) {
	active, err := activeModels(models)
	if err != nil {
		return nil, err
	}
	if len(active) > 1 {
		return nil, fmt.Errorf("multiple models are active: %q and %q", active[0].ID, active[1].ID)
	}
	if len(active) == 0 {
		return nil, nil
	}
	return &active[0], nil
}

func activeModels(models []modelRecord) ([]modelRecord, error) {
	var active []modelRecord
	for _, model := range models {
		if model.Status.Value == "loading" || model.Status.Value == "downloading" || model.Status.Value == "unloading" {
			return nil, fmt.Errorf("model %q is in transitional state %q; wait for it to finish before managing models", model.ID, model.Status.Value)
		}
		if model.Status.Value != "loaded" && model.Status.Value != "sleeping" {
			continue
		}
		active = append(active, model)
	}
	return active, nil
}

func (api *apiClient) models(ctx context.Context) ([]modelRecord, error) {
	var response modelList
	if err := api.requestJSON(ctx, http.MethodGet, "/models", nil, &response); err != nil {
		return nil, err
	}
	return response.Data, nil
}

func (api *apiClient) props(ctx context.Context, modelID string) (map[string]any, error) {
	var response map[string]any
	path := "/props?model=" + url.QueryEscape(modelID) + "&autoload=false"
	if err := api.requestJSON(ctx, http.MethodGet, path, nil, &response); err != nil {
		return nil, err
	}
	return response, nil
}

func (api *apiClient) slots(ctx context.Context, modelID string) ([]slot, error) {
	var response []slot
	path := "/slots?model=" + url.QueryEscape(modelID)
	if err := api.requestJSON(ctx, http.MethodGet, path, nil, &response); err != nil {
		return nil, err
	}
	return response, nil
}

func (api *apiClient) postModel(ctx context.Context, path, modelID string) error {
	var response map[string]any
	if err := api.requestJSON(ctx, http.MethodPost, path, map[string]string{"model": modelID}, &response); err != nil {
		return err
	}
	if success, ok := response["success"].(bool); ok && !success {
		return fmt.Errorf("%s reported success=false", path)
	}
	return nil
}

func (api *apiClient) slotAction(ctx context.Context, modelID, action, filename string) (slotActionResult, error) {
	var response slotActionResult
	path := fmt.Sprintf("/slots/0?action=%s", url.QueryEscape(action))
	if err := api.requestJSON(ctx, http.MethodPost, path, map[string]string{"model": modelID, "filename": filename}, &response); err != nil {
		return response, err
	}
	return response, nil
}

type apiError struct {
	Status int
	Body   string
}

func (e *apiError) Error() string { return fmt.Sprintf("HTTP %d: %s", e.Status, e.Body) }
func isClientError(err error) bool {
	var apiErr *apiError
	return errors.As(err, &apiErr) && apiErr.Status >= 400 && apiErr.Status < 500
}

func (api *apiClient) requestJSON(ctx context.Context, method, path string, body any, output any) error {
	requestURL := *api.base
	endpoint, err := url.Parse(path)
	if err != nil {
		return err
	}
	requestURL.Path = strings.TrimRight(api.base.Path, "/") + endpoint.Path
	requestURL.RawPath = ""
	requestURL.RawQuery = endpoint.RawQuery
	var requestBody io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return err
		}
		requestBody = strings.NewReader(string(encoded))
	}
	request, err := http.NewRequestWithContext(ctx, method, requestURL.String(), requestBody)
	if err != nil {
		return err
	}
	request.Header.Set("Accept", "application/json")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if api.token != "" {
		request.Header.Set("Authorization", "Bearer "+api.token)
	}
	response, err := api.http.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(io.LimitReader(response.Body, 8*1024*1024))
	if err != nil {
		return err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return &apiError{Status: response.StatusCode, Body: strings.TrimSpace(string(responseBody))}
	}
	if output == nil || len(responseBody) == 0 {
		return nil
	}
	if err := json.Unmarshal(responseBody, output); err != nil {
		return fmt.Errorf("decode %s %s response: %w (body %q)", method, path, err, string(responseBody))
	}
	return nil
}

func makeManifest(ctx context.Context, api *apiClient, opt options, presets map[string]preset, model modelRecord) (cacheManifest, string, error) {
	modelPreset, ok := presets[model.ID]
	if !ok {
		return cacheManifest{}, "", fmt.Errorf("model %q is not configured in INI", model.ID)
	}
	props, err := api.props(ctx, model.ID)
	if err != nil {
		return cacheManifest{}, "", err
	}
	resolvedPath, _ := props["model_path"].(string)
	if resolvedPath == "" {
		resolvedPath = model.ModelPath
	}
	if resolvedPath == "" {
		resolvedPath = model.Path
	}
	modelHash := opt.modelHashes[model.ID]
	if modelHash == "" && resolvedPath != "" {
		modelHash, _ = sha256Model(resolvedPath)
	}
	if modelHash == "" && modelPreset.Values["hf"] != "" {
		modelHash = hfSnapshotHash(resolvedPath)
	}
	buildInfo, _ := props["build_info"].(string)
	if buildInfo == "" {
		return cacheManifest{}, "", errors.New("/props did not provide build_info")
	}
	manifest := cacheManifest{
		Version:         1,
		ModelID:         model.ID,
		Session:         opt.session,
		Source:          firstNonEmpty(modelPreset.Values["hf"], modelPreset.Values["model"]),
		ResolvedModel:   resolvedPath,
		ModelSHA256:     modelHash,
		Preset:          cloneMap(modelPreset.Values),
		AdapterSettings: selectKeys(modelPreset.Values, func(key string) bool { return strings.Contains(key, "lora") || strings.Contains(key, "adapter") }),
		ContextRoPESet: selectKeys(modelPreset.Values, func(key string) bool {
			return key == "c" || strings.Contains(key, "ctx") || strings.Contains(key, "rope") || key == "yarn"
		}),
		KVTypes: selectKeys(modelPreset.Values, func(key string) bool {
			return strings.Contains(key, "cache-type") || key == "ctk" || key == "ctv" || strings.Contains(key, "cache-k") || strings.Contains(key, "cache-v")
		}),
		BuildInfo:  buildInfo,
		Restorable: modelHash != "",
	}
	if !manifest.Restorable {
		manifest.RestorationNotes = "no cryptographic model hash is available to this client; provide --model-sha256 for this model"
	}
	keyBytes, err := json.Marshal(struct {
		Version         int               `json:"version"`
		ModelID         string            `json:"model_id"`
		Session         string            `json:"session"`
		Source          string            `json:"source"`
		ModelHash       string            `json:"model_sha256"`
		Preset          map[string]string `json:"preset"`
		AdapterSettings map[string]string `json:"adapter_settings"`
		ContextRoPESet  map[string]string `json:"context_rope_settings"`
		KVTypes         map[string]string `json:"kv_types"`
		BuildInfo       string            `json:"build_info"`
	}{manifest.Version, manifest.ModelID, manifest.Session, manifest.Source, manifest.ModelSHA256, manifest.Preset, manifest.AdapterSettings, manifest.ContextRoPESet, manifest.KVTypes, manifest.BuildInfo})
	if err != nil {
		return cacheManifest{}, "", err
	}
	sum := sha256.Sum256(keyBytes)
	return manifest, hex.EncodeToString(sum[:]), nil
}

func hfSnapshotHash(path string) string {
	for _, marker := range []string{"/snapshots/", "/snapshot/"} {
		if at := strings.LastIndex(path, marker); at >= 0 {
			revision := strings.SplitN(path[at+len(marker):], string(os.PathSeparator), 2)[0]
			if len(revision) >= 32 {
				if _, err := hex.DecodeString(revision); err == nil {
					return "hf-revision:" + revision
				}
			}
		}
	}
	return ""
}

func sha256File(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func sha256Model(path string) (string, error) {
	base := filepath.Base(path)
	const marker = "-00001-of-"
	if index := strings.Index(base, marker); index >= 0 {
		pattern := filepath.Join(filepath.Dir(path), base[:index+1]+"*-of-*.gguf")
		parts, err := filepath.Glob(pattern)
		if err != nil {
			return "", err
		}
		if len(parts) == 0 {
			return "", os.ErrNotExist
		}
		hash := sha256.New()
		for _, part := range parts {
			if _, err := io.WriteString(hash, filepath.Base(part)+"\x00"); err != nil {
				return "", err
			}
			file, err := os.Open(part)
			if err != nil {
				return "", err
			}
			_, copyErr := io.Copy(hash, file)
			closeErr := file.Close()
			if copyErr != nil {
				return "", copyErr
			}
			if closeErr != nil {
				return "", closeErr
			}
		}
		return hex.EncodeToString(hash.Sum(nil)), nil
	}
	return sha256File(path)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func cloneMap(values map[string]string) map[string]string {
	out := make(map[string]string, len(values))
	for key, value := range values {
		out[key] = value
	}
	return out
}

func selectKeys(values map[string]string, match func(string) bool) map[string]string {
	out := map[string]string{}
	for key, value := range values {
		if match(key) {
			out[key] = value
		}
	}
	return out
}

func compatible(saved, current cacheManifest) bool {
	if !saved.Restorable || !current.Restorable || saved.ModelSHA256 == "" || current.ModelSHA256 == "" {
		return false
	}
	saved.CacheFilename = ""
	current.CacheFilename = ""
	saved.SavedAt = time.Time{}
	current.SavedAt = time.Time{}
	saved.RestorationNotes = ""
	current.RestorationNotes = ""
	left, _ := json.Marshal(saved)
	right, _ := json.Marshal(current)
	return string(left) == string(right)
}

func manifestPath(stateDir, key string) string { return filepath.Join(stateDir, "cache-"+key+".json") }

func readManifest(path string) (*cacheManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var result cacheManifest
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	if result.Version != 1 || result.CacheFilename == "" {
		return nil, errors.New("unsupported or incomplete cache manifest")
	}
	return &result, nil
}

func writeManifest(stateDir, key string, manifest cacheManifest) error {
	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	path := manifestPath(stateDir, key)
	tmp, err := os.CreateTemp(stateDir, ".manifest-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

func withLock(stateDir string, action func() error) error {
	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		return fmt.Errorf("create state directory %s: %w", stateDir, err)
	}
	lockPath := filepath.Join(stateDir, "switch.lock")
	file, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer file.Close()
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		return errors.New("another llama-modelctl operation holds the state lock; use a shared --state-dir to coordinate multiple clients")
	}
	defer syscall.Flock(int(file.Fd()), syscall.LOCK_UN) //nolint:errcheck
	return action()
}
